import argparse, sys
import trimesh
import torch
from tqdm import tqdm
import numpy as np
try:
    from mypath import *
except:
    import os
    P_CWD="/".join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1])
    sys.path.append(P_CWD)
    from mypath import *
import warnings

from utils import file_io, tensor_utils, network_utils, vis_utils
from same.skel_pose_graph import SkelPoseGraph
from asmr.model import (
    Model as SAUS,
    make_load_model as load_saus,
    out_post_fwd
)
from same.model import (
    Model as SAME,
    make_load_model as load_same,
)

from asmr.mesh_graph import MeshGraph
from asmr.dataset import PairedDataset, get_paired_data_loader
from asmr.metric import compute_metric
from asmr.loss import compute_loss

def prepare_model_test(saus_exp, same_exp, device):
    # device, printoptions
    tensor_utils.set_device(device)
    np.set_printoptions(precision=5, suppress=True)
    torch.set_printoptions(precision=5, sci_mode=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # SAME
    same_model, same_cfg = load_same(same_exp, device)
    same_model.eval()

    # SAUS
    saus_model, saus_cfg = load_saus(saus_exp, device)
    saus_model.eval()

    load_dir = os.path.join(RESULT_DIR, saus_exp.split("/")[0])
    same_ms_dict = torch.load(os.path.join(load_dir, "same_ms_dict.pt"))
    saus_ms_dict = torch.load(os.path.join(load_dir, "saus_ms_dict.pt"))
    mesh_ms_dict = torch.load(os.path.join(load_dir, "mesh_ms_dict.pt"))

    # set SkelPoseGraph class variables
    SkelPoseGraph.skel_cfg = saus_cfg["representation"]["skel"]
    SkelPoseGraph.pose_cfg = saus_cfg["representation"]["pose"]
    SkelPoseGraph.ms_dict = same_ms_dict
    SkelPoseGraph.saus_ms_dict = saus_ms_dict
    
    MeshGraph.ms_dict = mesh_ms_dict

    return same_model, saus_model, same_cfg, saus_cfg, same_ms_dict, saus_ms_dict, mesh_ms_dict

# python saus/train.py --exp test --same_exp ckpt0 --cfg saus
# python saus/train.py --cfg saus --exp test_000

def set_seed(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    tensor_utils.set_device(args.device)
    np.set_printoptions(precision=5, suppress=True)
    torch.set_printoptions(precision=5, sci_mode=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True)
    parser.add_argument("--same_exp", type=str, required=True)
    parser.add_argument("--test_cfg", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()
    test_cfg = file_io.load_cfg(args.test_cfg)
    
    # ignore warnings
    warnings.filterwarnings("ignore")
    
    # load
    same_model, saus_model, same_cfg, saus_cfg, same_ms_dict, saus_ms_dict, mesh_ms_dict \
        = prepare_model_test(args.exp, args.same_exp, args.device)
    
    ## Dataset
    # train_cfg = saus_cfg["train"]
    data_dir = os.path.join(DATA_DIR, test_cfg["test_data"]["dir"])
    dl = get_paired_data_loader(
        data_dir,
        test_cfg["batch_size"],
        shuffle=False,
        data_config=test_cfg["test_data"],
        rep_config=saus_cfg["representation"],
        get_dfn_op=(saus_cfg["model"]["MeshEncoder"]["type"] == "DiffusionNet"),
        train=False,
        device=args.device,
    )

    ## Test
    with torch.no_grad():
        test_metrics = {metric: list() for metric in test_cfg["metric"]}
        vtx_list = []
        for mi, (src_skel_pose, tgt_skel_pose, tgt_mesh) in enumerate(tqdm(dl)):
            # forward
            pred_skinning, pred_skel_pose = saus_model(tgt_mesh, src_skel_pose)     # [B, V, J] or [sumV, sumJ] / [sumJ, 6]

            # get retargeted motion with same model
            _, hatD = same_model(src_skel_pose, pred_skel_pose)   # [sumJ, 11]
            
            # construct out
            out = {
                "pred_skinning": pred_skinning,
                "pred_pose": hatD,
            }

            # update out, get gt
            out, gt = out_post_fwd(
                out,
                src_skel_pose,
                pred_skel_pose,  # ! src_skel_pose -> pred_skel_pose
                tgt_mesh,
                same_ms_dict,
                saus_cfg["representation"]["out"],
                tgt_skel_pose=tgt_skel_pose,
            )
            
            metric = compute_metric(test_cfg["metric"], out, gt)

            for name, value in metric.items():
                test_metrics[name].append(value)

        # log - metric
        for name, values in test_metrics.items():
            res = torch.cat(values, dim=0).mean().item()
            print(f"{name}: {res:.2f}")