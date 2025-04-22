import argparse, sys, yaml, gc, shutil, random
import numpy as np
try:
    from mypath import *
except:
    import os
    P_CWD="/".join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1])
    sys.path.append(P_CWD)
    from mypath import *
import warnings
from tqdm import tqdm
    
from utils import file_io, tensor_utils, network_utils, vis_utils
from same.skel_pose_graph import SkelPoseGraph
from asmr.model import Model as SAUS, out_post_fwd
from same.model import Model as SAME

from asmr.mesh_graph import MeshGraph
from asmr.dataset import PairedDataset, get_paired_data_loader
from asmr.loss import compute_loss
from asmr.metric import compute_metric

def load_trainer(load_cfg, model, optimizer, scheduler, device, log_path):
    if (load_cfg is None) or (load_cfg["dir"] is None):
        return

    load_dir, load_epoch = load_cfg["dir"], load_cfg["epoch"]
    load_model_name = (
        "last_model.pt" if load_epoch is None else "model_{}.pt".format(load_epoch)
    )
    load_abs_dir = os.path.join(RESULT_DIR, load_dir)
    load_path = os.path.join(load_abs_dir, load_model_name)

    saved = torch.load(load_path, map_location=device)

    model.load_state_dict(saved["model"])
    optimizer.load_state_dict(saved["optimizer"])
    if "scheduler" in saved and scheduler:
        scheduler.load_state_dict(saved["scheduler"])
    epoch_cnt, iter_cnt = saved["epoch"], saved["iter"]

    prev_log_path = os.path.join(load_abs_dir, "logs")
    if not os.path.samefile(prev_log_path, log_path):
        import shutil

        shutil.rmtree(log_path)
        shutil.copytree(prev_log_path, log_path)
    print("continue training from ", prev_log_path)
    return epoch_cnt, iter_cnt

def set_seed(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    tensor_utils.set_device(args.device)
    np.set_printoptions(precision=5, suppress=True)
    torch.set_printoptions(precision=5, sci_mode=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True)
    parser.add_argument("--cfg", type=str, required=True, default="saus")
    parser.add_argument("--val_cfg", type=str, default="saus_val")
    parser.add_argument("--same_exp", type=str, default="ckpt0")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    cfg = file_io.load_cfg(args.cfg)

    # ignore warnings
    warnings.filterwarnings("ignore")
    
    ## seed, device, printoptions
    import torch
    set_seed(args)
    torch.multiprocessing.set_start_method('spawn')

    train_cfg = cfg["train"]
    if "copy_orig_contact" in cfg["train"]:
        PairedDataset.copy_orig_contact = cfg["train"]["copy_orig_contact"]

    ## Dataset
    data_dir = os.path.join(DATA_DIR, cfg["train_data"]["dir"])
    dl = get_paired_data_loader(
        data_dir,
        train_cfg["batch_size"],
        shuffle=True,
        data_config=cfg["train_data"],
        rep_config=cfg["representation"],
        get_dfn_op=(cfg["model"]["MeshEncoder"]["type"] == "DiffusionNet"),
        train=True,
        device=args.device,
    )
    if args.val_cfg is not None:
        val_cfg = file_io.load_cfg(args.val_cfg)
        val_data_dir = os.path.join(DATA_DIR, val_cfg["val_data"]["dir"])
        val_dl = get_paired_data_loader(
            val_data_dir,
            val_cfg["batch_size"],
            shuffle=True,
            data_config=val_cfg["val_data"],
            rep_config=cfg["representation"],
            get_dfn_op=(cfg["model"]["MeshEncoder"]["type"] == "DiffusionNet"),
            train=False,
            device=args.device,
        )

    ## Pre-trained SAME model
    same_cfg = file_io.load_model_cfg(args.same_exp)
    same_model = SAME(same_cfg["model"], same_cfg["representation"]).to(device=args.device)
    same_model.load_params(args.same_exp, freeze=True)
    same_model.eval()

    ## Model, Optimizer, Scheduler
    saus_model = SAUS(cfg["model"], cfg["representation"]).to(device=args.device)
    optimizer = torch.optim.AdamW(saus_model.parameters(), lr=train_cfg["learning_rate"])
    scheduler = network_utils.get_scheduler(optimizer, train_cfg["lr_schedule"], train_cfg["epoch_num"])

    ## Log setup
    from torch.utils.tensorboard import SummaryWriter

    save_dir = os.path.join(RESULT_DIR, args.exp)
    log_path = os.path.join(save_dir, "logs")
    writer = SummaryWriter(log_path)

    with open(os.path.join(save_dir, "para.txt"), "w") as para_file:
        para_file.write(" ".join(sys.argv))
    with open(os.path.join(save_dir, "config.yaml"), "w") as config_file:
        yaml.dump(cfg, config_file)  # save all cfg (not only cfg(==cfg['train]))
    print("SAVE DIR: ", save_dir)

    # write config to tensorboard
    writer.add_text("Config", yaml.dump(cfg), 0)

    ## Training setup
    # copy training ms_dict to model directory (so that it loads correctly during test time)
    shutil.copyfile(os.path.join(data_dir, "same_ms_dict.pt"), os.path.join(save_dir, "same_ms_dict.pt"))
    shutil.copyfile(os.path.join(data_dir, "saus_ms_dict.pt"), os.path.join(save_dir, "saus_ms_dict.pt"))
    same_ms_dict = torch.load(os.path.join(data_dir, "same_ms_dict.pt"))
    saus_ms_dict = torch.load(os.path.join(data_dir, "saus_ms_dict.pt"))

    # set SkelPoseGraph class variables
    SkelPoseGraph.skel_cfg = cfg["representation"]["skel"]
    SkelPoseGraph.pose_cfg = cfg["representation"]["pose"]
    SkelPoseGraph.ms_dict = same_ms_dict
    SkelPoseGraph.saus_ms_dict = saus_ms_dict

    # set MeshGraph class variables
    shutil.copyfile(os.path.join(data_dir, "mesh_ms_dict.pt"), os.path.join(save_dir, "mesh_ms_dict.pt"))
    mesh_ms_dict = torch.load(os.path.join(data_dir, "mesh_ms_dict.pt"))
    MeshGraph.ms_dict = mesh_ms_dict

    ## Load checkpoint
    epoch_init, iter_init = 0, 0
    if "load" in train_cfg:
        epoch_init, iter_init = load_trainer(
            train_cfg["load"], saus_model, optimizer, scheduler, args.device, log_path
        )

    print("======================= READY TO TRAIN ===================== ")

    ## Train Loop
    iter_cnt = iter_init
    epoch_loss = {loss: 0 for loss in list(train_cfg["loss"].keys()) + ["total"]}
    iter_loss = {loss: 0 for loss in list(train_cfg["loss"].keys()) + ["total"]}
    for epoch_cnt in tqdm(range(epoch_init, train_cfg["epoch_num"]), desc="Epoch", leave=False):
        # training loop
        saus_model.train()
        epoch_loss = {loss: 0 for loss in list(train_cfg["loss"].keys()) + ["total"]}
        for bi, (src_skel_pose, tgt_skel_pose, tgt_mesh) in enumerate(tqdm(dl, desc="Batch", leave=False)):
            torch.cuda.empty_cache()
            optimizer.zero_grad()

            # forward
            """
                Args:
                    src_skel_pose (SkelPoseGraph)
                        - skel_x    [sumJ, 6]  (S in paper: lo, go)
                        - pose_x    [sumJ, 26] (D in paper: q, p, r, pv, qv, pprev, c)
                        - src_x     [sumJ, 32] includes both skel_x and pose_x
                    
                    tgt_mesh (MeshGraph)
                        - tpose_v, posed_v, diff3f, f, nverts
                        - mass, L, evals, evexs, grad_X, grag_Y, batch_size, faces
            """
            pred_skinning, pred_skel_pose = saus_model.forward(tgt_mesh, src_skel_pose)     # [B, V, J] or [sumV, sumJ] / [sumJ, 6]   
            
            # get retargeted motion with same model 
            """
                Args:
                    - src_graph (SkelPoseGraph): need skel and pose
                    - tgt_graph (SkelPoseGraph): need skel only
                    
                Return:
                    - z
                    - hatD (torch.tensor): retargeted pose 
            """
            _, hatD = same_model.forward(src_skel_pose, pred_skel_pose)   # [sumJ, 11]

            # construct out
            out = {
                "pred_skinning": pred_skinning, 
                "pred_pose": hatD,
            }
            
            # update out, get gt 
            out, gt = out_post_fwd(
                out,
                src_skel_pose,
                pred_skel_pose,
                tgt_mesh,
                same_ms_dict, 
                cfg["representation"]["out"],
                tgt_skel_pose=tgt_skel_pose,
            )
        
            # compute loss
            loss = compute_loss(train_cfg["loss"], out, gt)

            # backward
            loss["total"].backward()
            optimizer.step()

            iter_cnt += 1

            # log - loss
            for k, v in loss.items():
                v_cdn = tensor_utils.cdn(v)
                epoch_loss[k] += v_cdn
                iter_loss[k] += v_cdn

            if iter_cnt % train_cfg["log_per_iter"] == 0:
                for k, v in iter_loss.items():
                    writer.add_scalar("Loss-Iter/" + k, v / train_cfg["log_per_iter"], iter_cnt)
                iter_loss = {loss: 0 for loss in list(train_cfg["loss"].keys()) + ["total"]}
            
        # end of batch loop
                
        # log - loss, lr
        for k, v in epoch_loss.items():
            writer.add_scalar("Loss-Epoch/" + k, v / len(dl), epoch_cnt)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch_cnt)
        
        # save checkpoint
        if (epoch_cnt % train_cfg["save_per"] == 0) or (
            epoch_cnt + 1 == train_cfg["epoch_num"]
        ):
            save_state = {
                "epoch": epoch_cnt,
                "model": saus_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "iter": iter_cnt,
            }
            torch.save(
                save_state, os.path.join(save_dir, f"model_{epoch_cnt:03d}.pt")
            )
            torch.save(save_state, os.path.join(save_dir, "last_model.pt"))

        if scheduler is not None:
            scheduler.step()
        
        torch.cuda.empty_cache()
        gc.collect()
        
        # validation
        if (epoch_cnt + 1) % val_cfg["val_per"] == 0:
            with torch.no_grad():
                saus_model.eval()
                epoch_metric = {metric: list() for metric in val_cfg["metric"]}
                epoch_loss = {loss: 0 for loss in list(train_cfg["loss"].keys()) + ["total"]}
                for bi, (src_skel_pose, tgt_skel_pose, tgt_mesh) in enumerate(tqdm(val_dl, desc="Val-Batch", leave=False)):
                    pred_skinning, pred_skel_pose = saus_model.forward(tgt_mesh, src_skel_pose)
                    _, hatD = same_model.forward(src_skel_pose, pred_skel_pose)
                    out = {
                        "pred_skinning": pred_skinning, 
                        "pred_pose": hatD,
                    }
                    out, gt = out_post_fwd(
                        out,
                        src_skel_pose,
                        pred_skel_pose,
                        tgt_mesh,
                        same_ms_dict, 
                        cfg["representation"]["out"],
                        tgt_skel_pose=tgt_skel_pose,
                    )

                    metric = compute_metric(val_cfg["metric"], out, gt)
                    loss = compute_loss(train_cfg["loss"], out, gt)

                    for name, value in metric.items():
                        epoch_metric[name].append(value)
                    for name, value in loss.items():
                        epoch_loss[name] += tensor_utils.cdn(value)
                
                # log - metric
                for name, values in epoch_metric.items():
                    res = torch.cat(values, dim=0).mean().item()
                    writer.add_scalar("Metric/" + name, res, epoch_cnt)

                # log - loss
                for name, value in epoch_loss.items():
                    writer.add_scalar("Loss-Val/" + name, value / len(val_dl), epoch_cnt)

                # log - mesh at the end of epoch
                tgt_mesh = gt["tgt_mesh"]
                src_skel_pose = gt["src_skel_pose"]
                pred_skel_pose = out["pred_skel_pose"]
                nverts = tgt_mesh.nverts[0]
                nfaces = tgt_mesh.nfaces[0]
                njoints = src_skel_pose.njoints[0]

                gt_mesh = vis_utils.get_char_mesh(tensor_utils.cdn(tgt_mesh.posed_v[:nverts]),
                                                        tensor_utils.cdn(tgt_mesh.f[:nfaces]))
                pred_mesh = vis_utils.get_char_mesh(tensor_utils.cdn(out["vtx"][:nverts]),
                                                    tensor_utils.cdn(tgt_mesh.f[:nfaces]))
                
                writer.add_mesh("Mesh/GT", vertices=gt_mesh.vertices[None], faces=gt_mesh.faces[None], global_step=epoch_cnt)
                writer.add_mesh("Mesh/Pred", vertices=pred_mesh.vertices[None], faces=pred_mesh.faces[None], global_step=epoch_cnt)

                gt_mesh.export(os.path.join(log_path, f"gt_mesh_{epoch_cnt:03d}.obj"))
                pred_mesh.export(os.path.join(log_path, f"pred_mesh_{epoch_cnt:03d}.obj"))

                # log - skinning
                skinning_maxjoint = vis_utils.get_skinning_weights_mesh(tensor_utils.cdn(tgt_mesh.tpose_v[:nverts]),
                                                                        tensor_utils.cdn(tgt_mesh.f[:nfaces]),
                                                                        tensor_utils.cdn(pred_skinning[0, :nverts, :njoints]),
                                                                        perjoint=False,
                                                                        tpose_xform=tensor_utils.cdn(out["fk_tpose"][:njoints]),
                                                                        merge_skel_mesh=True)
                skinning_perjoint = vis_utils.get_skinning_weights_mesh(tensor_utils.cdn(tgt_mesh.tpose_v[:nverts]),
                                                                        tensor_utils.cdn(tgt_mesh.f[:nfaces]),
                                                                        tensor_utils.cdn(pred_skinning[0, :nverts, :njoints]),
                                                                        perjoint=True,
                                                                        tpose_xform=tensor_utils.cdn(out["fk_tpose"][:njoints]),
                                                                        merge_skel_mesh=True)
                writer.add_mesh("Skinning-MaxJoint",
                                vertices=skinning_maxjoint.vertices[None],
                                faces=skinning_maxjoint.faces[None],
                                colors=torch.torch.from_numpy(skinning_maxjoint.visual.vertex_colors[..., :3]).float()[None],
                                global_step=epoch_cnt)
                for sk_idx, sk in enumerate(skinning_perjoint):
                    writer.add_mesh(f"Skinning-PerJoint-Epoch{epoch_cnt:03d}",
                                    vertices=sk.vertices[None],
                                    faces=sk.faces[None],
                                    colors=torch.torch.from_numpy(sk.visual.vertex_colors[..., :3]).float()[None],
                                    global_step=sk_idx)

                # log - pose
                gt_pose = vis_utils.get_pose_mesh(tensor_utils.cdn(gt["fk"][:njoints]), add_plane=True)
                pred_pose = vis_utils.get_pose_mesh(tensor_utils.cdn(out["fk"][:njoints]), add_plane=True)
                tgt_pose = vis_utils.get_pose_mesh(tensor_utils.cdn(gt["fk_tgt"][:njoints]), add_plane=True)
                writer.add_mesh("Pose/GT", vertices=gt_pose.vertices[None], faces=gt_pose.faces[None], global_step=epoch_cnt)
                writer.add_mesh("Pose/Pred", vertices=pred_pose.vertices[None], faces=pred_pose.faces[None], global_step=epoch_cnt)
                writer.add_mesh("Pose/Target", vertices=tgt_pose.vertices[None], faces=tgt_pose.faces[None], global_step=epoch_cnt)

                # log - T-pose
                gt_tpose = vis_utils.get_pose_mesh(tensor_utils.cdn(gt["fk_tpose"][:njoints]))
                pred_tpose = vis_utils.get_pose_mesh(tensor_utils.cdn(out["fk_tpose"][:njoints]))
                tgt_tpose = vis_utils.get_pose_mesh(tensor_utils.cdn(gt["fk_tgt_tpose"][:njoints]))
                writer.add_mesh("Skeleton/GT", vertices=gt_tpose.vertices[None], faces=gt_tpose.faces[None], global_step=epoch_cnt)
                writer.add_mesh("Skeleton/Pred", vertices=pred_tpose.vertices[None], faces=pred_tpose.faces[None], global_step=epoch_cnt)
                writer.add_mesh("Skeleton/Target", vertices=tgt_tpose.vertices[None], faces=tgt_tpose.faces[None], global_step=epoch_cnt)

                del gt_mesh, pred_mesh
                del skinning_maxjoint, skinning_perjoint
                del gt_pose, pred_pose, tgt_pose
                del gt_tpose, pred_tpose, tgt_tpose

                gc.collect()
                torch.cuda.empty_cache()