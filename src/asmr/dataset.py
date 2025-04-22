import os, torch, random
from os.path import join as pjoin
import numpy as np
from torch.utils.data import Dataset, Sampler, DataLoader

# from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from functools import partial
from dataclasses import dataclass
import pickle
from tqdm import tqdm

import trimesh
from utils.mesh_utils import map_vertices, decimate_mesh_vertex
from utils.dfn_utils import get_dfn_info2

from mypath import *
from same.skel_pose_graph import SkelPoseGraph
from asmr.mesh_graph import MeshGraph

from functools import partial
import gc


@dataclass
class SkelData:
    # [nJ, nDim]
    lo: torch.Tensor
    go: torch.Tensor
    qb: torch.BoolTensor
    # [nE, 2]
    edge_index: torch.LongTensor
    edge_feature: torch.LongTensor
    # [nJ]
    sym_idx: torch.LongTensor


@dataclass
class PoseData:
    # [nJ, nDim]
    q: torch.Tensor
    p: torch.Tensor
    qv: torch.Tensor
    pv: torch.Tensor
    pprev: torch.Tensor
    c: torch.BoolTensor
    # [nDim]
    r: torch.Tensor


@dataclass
class MeshData:
    # [nV, nDim]
    tpose_v: torch.Tensor
    posed_v: torch.Tensor
    diff3f: torch.Tensor
    # [nF, 3]
    f: torch.LongTensor
    # [nE, 2]    
    edge_index: torch.LongTensor


def npz_2_skel_data(lo, go, qb, edges, sym_idx):
    if not (np.arange(edges.shape[0]) == edges[:, 1]).all():
        edges = edges[np.argsort(edges[:, 1])]  # sort by child idx - just in case ...
    
    skel_data = SkelData(
        torch.Tensor(lo),
        torch.Tensor(go),
        torch.BoolTensor(qb),
        torch.LongTensor(edges[:, :2]).transpose(1, 0),
        torch.LongTensor(edges[:, 2:]),
        torch.LongTensor(sym_idx),
    )

    return skel_data


def npz_2_pose_data(q, p, qv, pv, pprev, c, r, offset=1):
    nF = q.shape[0]
    pose_data_list = [
        PoseData(
            torch.Tensor(q[i]),
            torch.Tensor(p[i]),
            torch.Tensor(qv[i]),
            torch.Tensor(pv[i]),
            torch.Tensor(pprev[i]),
            torch.BoolTensor(c[i]).reshape(-1, 1),
            torch.Tensor(r[i]).reshape(1, -1),
        )
        for i in range(0, nF, offset)
    ]

    return pose_data_list


def npz_2_mesh_data(tpose_mesh, tpose_mesh_diff3f, verts, edge_index):
    tpose_v = torch.Tensor(tpose_mesh["c_rest_pos"])
    posed_v = torch.Tensor(verts)
    diff3f = torch.Tensor(tpose_mesh_diff3f) if tpose_mesh_diff3f is not None else None
    f = torch.LongTensor(tpose_mesh["fid_to_cids"])
    edge_index = torch.LongTensor(edge_index).transpose(1, 0)
    return MeshData(tpose_v, posed_v, diff3f, f, edge_index)


def apply_decimation(tpose_v, faces, nverts):
    if tpose_v.shape[0] <= nverts:
        return torch.arange(tpose_v.shape[0]), faces
    
    original_mesh=trimesh.Trimesh(
        vertices=tpose_v,
        faces=faces,
        process=False
    )

    # decimate mesh
    decim_mesh = decimate_mesh_vertex(original_mesh, nverts, 2)
    _, vertex_map = map_vertices(original_mesh, decim_mesh)
    
    return vertex_map, decim_mesh.faces
    

def save_decimation(orig_mesh_dir, mesh_dir, vmap, fmap, tpose_v, tpose_mesh_diff3f):
    # make symbolic link when decimated is not needed
    if len(vmap) == tpose_v.shape[0]:
        os.symlink(orig_mesh_dir, mesh_dir)
    else:
        os.makedirs(mesh_dir, exist_ok=True)
        
        # save
        np.savez(pjoin(mesh_dir, "tpose_mesh.npz"), c_rest_pos=tpose_v[vmap], fid_to_cids=fmap)
        np.save(pjoin(mesh_dir, "tpose_mesh_diff3f.npy"), tpose_mesh_diff3f[vmap])
        
        motion_dirs = sorted([d for d in os.listdir(orig_mesh_dir) if os.path.isdir(pjoin(orig_mesh_dir, d))])

        for md in tqdm(motion_dirs, desc="Decimating"):
            os.makedirs(pjoin(mesh_dir, md), exist_ok=True)
            vert_files = sorted([f for f in os.listdir(pjoin(orig_mesh_dir, md)) if f.endswith(".npy")])
            for vf in vert_files:
                verts = np.load(pjoin(orig_mesh_dir, md, vf))
                np.save(pjoin(mesh_dir, md, vf), verts[vmap])


def edges_from_faces(face_idx):
    e0 = face_idx[:, [0, 1]]
    e1 = face_idx[:, [1, 2]]
    e2 = face_idx[:, [2, 0]]
    edges = np.concatenate([e0, e1, e2], axis=0)
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0) # [nE, 2]
    return edges

class PairedDataset(Dataset):
    copy_orig_contact = False

    def __init__(self, data_config, rep_config, get_dfn_op, train=True, eval_rig=False):
        # train or test
        self.train = train
        self.eval_rig = eval_rig

        # skel
        self.skel_list = []
        self.pose_list = []
        self.tgt_skel_list = []
        self.tgt_pose_list = []

        # mesh
        self.tpose_mesh_list = []       # tpose mesh features
        self.tpose_mesh_dfnp_list = []  # tpose mesh operator (save file names and load during training)
        self.tpose_diff3f_list = []     # tpose mesh diff3f features
        self.posed_mesh_list = []       # posed mesh files (save file names and load during training)
        self.mesh_edge_list = []

        self.vmap_list = []
        self.fmap_list = []

        # config
        self.data_config = data_config
        self.rep_config = rep_config
        # self.use_diff3f = "diff3f" in rep_config["mesh"]

        # DiffusionNet operators
        self.get_dfn_op = get_dfn_op

    def load_mesh_data(self, mesh_dir):
        assert os.path.exists(mesh_dir), f"{mesh_dir} does not exist"
        assert os.path.exists(pjoin(mesh_dir, "tpose_mesh.npz")), f"{mesh_dir} does not have tpose_mesh.npz"

        vmap, fmap = None, None
        
        # decimate mesh if needed
        if self.data_config.get("use_decimate", False):
            # directory names
            orig_mesh_dir = mesh_dir
            nverts = np.load(pjoin(orig_mesh_dir, "tpose_mesh.npz"))["c_rest_pos"].shape[0]
            while nverts > self.data_config["num_vertex"]:
                nverts = int(nverts * 0.9)

            # load data
            tpose_mesh_path = pjoin(orig_mesh_dir, "tpose_mesh.npz")
            tpose_mesh = np.load(tpose_mesh_path)
            tpose_mesh_diff3f = np.load(tpose_mesh_path.replace(".npz", "_diff3f.npy"))
            tpose_v = tpose_mesh["c_rest_pos"]
            faces = tpose_mesh["fid_to_cids"]

            # decimation
            vmap, fmap = apply_decimation(tpose_v, faces, nverts)

            # save
            if not os.path.exists(mesh_dir):
                save_decimation(orig_mesh_dir, mesh_dir, vmap, fmap, tpose_v, tpose_mesh_diff3f)
                    
        # paths
        tpose_path = pjoin(mesh_dir, "tpose_mesh.npz")
        tpose_op_path = pjoin(mesh_dir, "tpose_mesh_dfnp.pkl")

        motion_dirs = sorted([d for d in os.listdir(mesh_dir) if os.path.isdir(pjoin(mesh_dir, d)) and d not in [".ipynb_checkpoints"]])
        posed_path_list = []
        if self.eval_rig:
            if len(motion_dirs) > 0:
                md = motion_dirs[0]
                vert_files = sorted([f for f in os.listdir(pjoin(mesh_dir, md)) if f.endswith(".npy")])
                posed_path_list.append(pjoin(mesh_dir, md, vert_files[1]))
        elif self.data_config.get("seq_idx", False):
            seq_idx = self.data_config["seq_idx"]
            if len(motion_dirs) > 0:
                for i in seq_idx:
                    md = motion_dirs[i]
                    vert_files = sorted([f for f in os.listdir(pjoin(mesh_dir, md)) if f.endswith(".npy")])
                    for vf in vert_files[1::self.data_config["frame_offset"]]:
                        posed_path_list.append(pjoin(mesh_dir, md, vf))
        else:
            for md in motion_dirs:
                vert_files = sorted([f for f in os.listdir(pjoin(mesh_dir, md)) if f.endswith(".npy")])
                for vf in vert_files[1::self.data_config["frame_offset"]]:
                    posed_path_list.append(pjoin(mesh_dir, md, vf))
        
        # pre-processing for edge indices
        face_idx = np.load(tpose_path)["fid_to_cids"]
        edges = edges_from_faces(face_idx)
        
        return tpose_path, tpose_op_path, posed_path_list, edges
    

    def load_mesh_stats_and_dfnp(self):
        # load mesh data
        tpose_verts, faces = [], []
        for tpose_mesh in self.tpose_mesh_list:
            # tpose_mesh = np.load(tpose_path)
            tpose_verts.append(tpose_mesh["c_rest_pos"])
            faces.append(tpose_mesh["fid_to_cids"])
        tpose_verts_cat = np.concatenate(tpose_verts, axis=0)

        # normailzation constants
        mean, std = tpose_verts_cat.mean(axis=0), tpose_verts_cat.std(axis=0)
        
        # extract DiffusionNet operators
        dfn_info_list = []
        if self.get_dfn_op:
            for i in range(len(tpose_verts)):
                dfn_info = get_dfn_info2(torch.from_numpy(tpose_verts[i]).float(), torch.from_numpy(faces[i]).long(), map_location="cpu")
                dfn_info_list.append(dfn_info)
            # for dfn_path, dfn_info in zip(self.tpose_mesh_dfnp_list, dfn_info_list):
            #     pickle.dump(dfn_info, open(dfn_path, 'wb'))
        
        return mean, std, dfn_info_list


    def load_skel_data(self, skel_dir):
        assert os.path.exists(skel_dir), f"{skel_dir} does not exist"
        assert os.path.exists(pjoin(skel_dir, "skel.npz")), f"{skel_dir} does not have skel.npz"
        
        # source skeleton
        skel_npz = np.load(pjoin(skel_dir, "skel.npz"))
        sd = npz_2_skel_data(**skel_npz)
        self.skel_list.append(sd)

        motion_npz = sorted([f for f in os.listdir(skel_dir) if f.endswith(".npz") and f != "skel.npz"])
        pose_list = []

        if self.eval_rig:
            npz = np.load(pjoin(skel_dir, motion_npz[0]))
            pdl = npz_2_pose_data(**npz, offset=self.data_config["frame_offset"])
            pose_list.extend([ pdl[0] ])
        elif self.data_config.get("seq_idx", False):
            seq_idx = self.data_config["seq_idx"]
            for i in seq_idx:
                npz = np.load(pjoin(skel_dir, motion_npz[i]))
                pdl = npz_2_pose_data(**npz, offset=self.data_config["frame_offset"])
                pose_list.extend(pdl)
        else:
            for mi, motion in enumerate(motion_npz):
                npz = np.load(pjoin(skel_dir, motion))
                pdl = npz_2_pose_data(**npz, offset=self.data_config["frame_offset"])
                pose_list.extend(pdl)

        self.pose_list.append(pose_list)
    
    
    def load_tgt_skel_data(self, skel_dir):
        assert os.path.exists(skel_dir), f"{skel_dir} does not exist"
        assert os.path.exists(pjoin(skel_dir, "skel.npz")), f"{skel_dir} does not have skel.npz"
        
        # target skeleton
        skel_npz = np.load(pjoin(skel_dir, "skel.npz"))
        sd = npz_2_skel_data(**skel_npz)
        self.tgt_skel_list.append(sd)

        motion_npz = sorted([f for f in os.listdir(skel_dir) if f.endswith(".npz") and f != "skel.npz"])
        pose_list = []

        if self.eval_rig:
            npz = np.load(pjoin(skel_dir, motion_npz[0]))
            pdl = npz_2_pose_data(**npz, offset=self.data_config["frame_offset"])
            pose_list.extend([ pdl[0] ])
        elif self.data_config.get("seq_idx", False):
            seq_idx = self.data_config["seq_idx"]
            for i in seq_idx:
                npz = np.load(pjoin(skel_dir, motion_npz[i]))
                pdl = npz_2_pose_data(**npz, offset=self.data_config["frame_offset"])
                pose_list.extend(pdl)
        else:
            for mi, motion in enumerate(motion_npz):
                npz = np.load(pjoin(skel_dir, motion))
                pdl = npz_2_pose_data(**npz, offset=self.data_config["frame_offset"])
                pose_list.extend(pdl)

        self.tgt_pose_list.append(pose_list)


    def load_data(self, data_dir):
        char_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(pjoin(data_dir, d))])
        bar = tqdm(total=len(self.data_config["char_names"]) * (len(self.data_config["skel_names"]) + 1))
        for ci, char_dir in enumerate(char_dirs):
            if char_dir in [".DS_Store", ".ipynb_checkpoints"]:
                continue
            if char_dir not in self.data_config["char_names"]:
                continue

            # load mesh data
            bar.set_description(f"Loading {char_dir} Mesh")
            tpose_path, tpose_op_path, posed_mesh_list, edges = self.load_mesh_data(pjoin(data_dir, char_dir, "Mesh"))
            self.tpose_mesh_list.append(np.load(tpose_path))
            if "diff3f" in self.rep_config["mesh"]:
                self.tpose_diff3f_list.append(np.load(tpose_path.replace(".npz", "_diff3f.npy")))
            elif "diff3f_pca" in self.rep_config["mesh"]:
                self.tpose_diff3f_list.append(np.load(tpose_path.replace(".npz", "_diff3f_pca.npy")))
            else:
                self.tpose_diff3f_list.append(None)
            # self.tpose_mesh_dfnp_list.append(tpose_op_path)
            if len(posed_mesh_list) > 0:
                self.posed_mesh_list.append([np.load(pm) for pm in posed_mesh_list])
            self.mesh_edge_list.append(edges)
            bar.update(1)

            # load skeleton data
            skel_dirs = sorted([d for d in os.listdir(pjoin(data_dir, char_dir)) if os.path.isdir(pjoin(data_dir, char_dir, d)) and d.startswith("Skeleton")])
            for si, skel_dir in enumerate(skel_dirs):
                bar.set_description(f"Loading {char_dir} {skel_dir}")
                if skel_dir == self.data_config["tgt_skel"]:
                    if self.data_config.get("skel_char", None) is not None:
                        self.load_tgt_skel_data(pjoin(data_dir, self.data_config["skel_char"], skel_dir))
                    else:
                        self.load_tgt_skel_data(pjoin(data_dir, char_dir, skel_dir))
                if skel_dir not in self.data_config["skel_names"]:
                    continue
                if self.data_config.get("skel_char", None) is not None:
                    self.load_skel_data(pjoin(data_dir, self.data_config["skel_char"], skel_dir))
                else:
                    self.load_skel_data(pjoin(data_dir, char_dir, skel_dir))
                bar.update(1)
            
            if len(posed_mesh_list) == 0:
                self.posed_mesh_list.append([np.load(tpose_path)["c_rest_pos"] for _ in range(len(self.pose_list[-1]))])
        bar.close()
                
        # load mesh stats and dfnp
        mesh_mean, mesh_std, self.tpose_mesh_dfnp_list = self.load_mesh_stats_and_dfnp()
        if self.train:
            torch.save({
                "vtx_m": torch.from_numpy(mesh_mean).float(),
                "vtx_s": torch.from_numpy(mesh_std).float()
            }, pjoin(data_dir, "mesh_ms_dict.pt"))

        # # (opt.) load DiffusionNet operators
        # if self.get_dfn_op:
        #     for i in range(len(self.tpose_mesh_dfnp_list)):
        #         self.tpose_mesh_dfnp_list[i] = pickle.load(open(self.tpose_mesh_dfnp_list[i], mode='rb'))
        
        # check data consistency
        assert len(self.skel_list) == len(self.pose_list), f"mismatched number of skel and pose: {len(self.skel_list)} != {len(self.pose_list)}"
        assert len(self.skel_list) == len(self.tpose_mesh_list) * len(self.data_config["skel_names"]), f"mismatched number of skel and mesh data: {len(self.skel_list)} != {len(self.tpose_mesh_list) * len(self.data_config['skel_names'])}"
        for i in range(len(self.pose_list)):
            assert len(self.pose_list[i]) == len(self.posed_mesh_list[i // len(self.data_config["skel_names"])]), f"pose_list and posed_mesh_list must have same length, but mismatched at {i}: {len(self.pose_list[i])} != {len(self.posed_mesh_list[i // len(self.data_config['skel_names'])])}"

        # log data
        print(f"Loaded {len(self.skel_list)} characters with {len(self.pose_list[0])} frames each (total {len(self.skel_list) * len(self.pose_list[0])} frames)")
        gc.collect()


    def __getitem__(self, idx):
        # number of skels per character
        nskel = len(self.data_config["skel_names"])

        # character index
        src_skel_idx = idx // self.nframes
        if self.train:
            tgt_skel_idx = random.randint(0, self.nchars - 1)
        else:
            tgt_skel_idx = idx // self.nframes # same character
        tgt_char_idx = tgt_skel_idx // nskel

        # frame index
        frame_idx = idx % self.nframes

        # source skel and pose
        src_skel = self.skel_list[src_skel_idx]
        src_pose = self.pose_list[src_skel_idx][frame_idx]
        src_skel_pose = SkelPoseGraph(src_skel, src_pose)

        # target skel and pose with Skeleton00
        tgt_skel = self.tgt_skel_list[tgt_char_idx]
        tgt_pose = self.tgt_pose_list[tgt_char_idx][frame_idx]
        tgt_skel_pose = SkelPoseGraph(tgt_skel, tgt_pose)

        # target mesh
        tpose_mesh = self.tpose_mesh_list[tgt_char_idx]
        tpose_mesh_diff3f = self.tpose_diff3f_list[tgt_char_idx]
        # verts = np.load(self.posed_mesh_list[tgt_mesh_idx])
        # verts = np.load(self.posed_mesh_list[tgt_id][frame_idx])
        verts = self.posed_mesh_list[tgt_char_idx][frame_idx]
        edge_index = self.mesh_edge_list[tgt_char_idx]

        # make meshGraph
        tgt_mesh = npz_2_mesh_data(tpose_mesh, tpose_mesh_diff3f, verts, edge_index)#, self.vmap_list[tgt_id], self.fmap_list[tgt_id])
        tgt_mesh = MeshGraph(tgt_mesh)
            
        # set DiffusionNet operators
        if self.get_dfn_op:
            tgt_mesh.set_mesh_dfnp(self.tpose_mesh_dfnp_list[tgt_char_idx])
                
        return (src_skel_pose, tgt_skel_pose, tgt_mesh)

    def __len__(self):
        return len(self.pose_list[0]) * len(self.pose_list)

    @property
    def nframes(self):
        return len(self.pose_list[0])
    
    @property
    def nchars(self):
        return len(self.pose_list)


def PairedGraph_collate_fn(batch, device="cuda:0"):
    src_skel_pose_batch = Batch.from_data_list([item[0] for item in batch])
    tgt_skel_pose_batch = Batch.from_data_list([item[1] for item in batch])
    tgt_mesh_batch = Batch.from_data_list([item[2] for item in batch])

    if hasattr(src_skel_pose_batch, "sym_idx"):
        src_skel_pose_batch.sym_idx = src_skel_pose_batch.sym_idx + src_skel_pose_batch.batch

    src_skel_pose_batch = src_skel_pose_batch.to(device, non_blocking=True)
    tgt_skel_pose_batch = tgt_skel_pose_batch.to(device, non_blocking=True)
    tgt_mesh_batch = tgt_mesh_batch.to(device, non_blocking=True)

    # post-process sym_idx
    src_skel_pose_batch.sym_idx = src_skel_pose_batch.sym_idx - src_skel_pose_batch.batch + src_skel_pose_batch.njoints.cumsum(0)[src_skel_pose_batch.batch] - src_skel_pose_batch.njoints[src_skel_pose_batch.batch]
    tgt_skel_pose_batch.sym_idx = tgt_skel_pose_batch.sym_idx - tgt_skel_pose_batch.batch + tgt_skel_pose_batch.njoints.cumsum(0)[tgt_skel_pose_batch.batch] - tgt_skel_pose_batch.njoints[tgt_skel_pose_batch.batch]

    return src_skel_pose_batch, tgt_skel_pose_batch, tgt_mesh_batch


def get_paired_data_loader(data_dir, batch_size, shuffle, data_config, rep_config, get_dfn_op, train=True, eval_rig=False, device="cpu"):
    ds = PairedDataset(data_config, rep_config, get_dfn_op, train=train, eval_rig=eval_rig)
    ds.load_data(data_dir)
    
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        collate_fn=partial(
            PairedGraph_collate_fn,
            device=device,
        ),
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
    )
    return dl

# if __name__ == "__main__":
#     dataset = PairedDataset()
#     # import pdb;pdb.set_trace()
