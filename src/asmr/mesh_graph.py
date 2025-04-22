import torch
import torch_geometric

rep_dim = {
    "v": 3,  # vertex position
}

class MeshGraph(torch_geometric.data.Data):
    # class variable : should be set right after loading cfg, before using any of the data ...
    ms_dict = {}

    def __init__(self, mesh_data):
        super(MeshGraph, self).__init__()
        # mesh data
        if mesh_data is not None: # essential !
            self.tpose_v = mesh_data.tpose_v
            self.posed_v = mesh_data.posed_v
            self.diff3f = mesh_data.diff3f
            self.f = mesh_data.f
            self.edge_index = mesh_data.edge_index
            self.nverts = self.tpose_v.shape[0]
            self.nfaces = self.f.shape[0]
            self.nedges = self.edge_index.shape[1]
            
            # # mesh operator (dummy)
            # self.mass = torch.zeros(1)
            # self.L = torch.zeros(1)
            # self.evals = torch.zeros(1)
            # self.evecs = torch.zeros(1)
            # self.grad_X = torch.zeros(1)
            # self.grad_Y = torch.zeros(1)
            # self.batch_size = torch.ones(1, dtype=torch.int)

    # def set_mesh_dfnp(self, mesh_operators):
    #     # mesh operator
    #     self.mass, self.L, self.evals, self.evecs, self.grad_X, self.grad_Y, _ = mesh_operators
        
    # @property
    # def dfnp(self):
    #     """get mesh operators for DiffusionNet"""
    #     return [self.mass, self.L, self.evals, self.evecs, self.grad_X, self.grad_Y] #, self.f]
    
    # @property
    # def batch_dfnp(self):
    #     """given the 'batch_size', resize the operator tensors"""
    #     batch_mass = self.mass.reshape(self.batch_size, -1)
    #     batch_L = reshape_sparse_coo(self.L, (self.batch_size, -1, self.L.shape[-1]))
    #     batch_evals = self.evals.reshape(self.batch_size, -1)
    #     batch_evecs = self.evecs.reshape(self.batch_size, -1, self.evecs.shape[-1])
    #     batch_grad_X = reshape_sparse_coo(self.grad_X, (self.batch_size, -1, self.grad_X.shape[-1]))
    #     batch_grad_Y = reshape_sparse_coo(self.grad_Y, (self.batch_size, -1, self.grad_Y.shape[-1]))
    #     # batch_f = self.f.reshape(self.batch_size, -1, self.f.shape[-1])
    #     return [batch_mass, batch_L, batch_evals, batch_evecs, batch_grad_X, batch_grad_Y] #, list_f]
    
    # @property
    # def batch_m_input(self):
    #     """given the 'batch_size', reshape and preprocess data for mesh encoder"""
    #     B_tpose_v = self.tpose_v.reshape(self.batch_size, -1, self.tpose_v.shape[-1])   # [B, V, D=3]
    #     B_diff3f = self.diff3f.reshape(self.batch_size, -1, self.diff3f.shape[-1])      # [B, V, D=2048]
    #     return torch.cat([B_tpose_v, B_diff3f], dim=-1); 
    
    # @property
    # def batch_posed_v(self):
    #     """given the 'batch_size', reshape and preprocess data for mesh encoder"""
    #     B_posed_v = self.posed_v.reshape(self.batch_size, -1, self.tpose_v.shape[-1])
    #     return B_posed_v
    
    @property
    def edge_index_bidirection(self):
        return torch.hstack((self.edge_index, self.edge_index[[1, 0]]))
    
    def normalize_x(self, key):
        val = getattr(self, key)
        if key + "_m" in self.ms_dict:
            m = self.ms_dict[key + "_m"].to(device=val.device)
            s = self.ms_dict[key + "_s"].to(device=val.device)
            val = (val - m) / s
        return val
    
    def ms_from_key(self, key):
        m = self.ms_dict[key + "_m"].to(device=self.tpose_v.device)
        s = self.ms_dict[key + "_s"].to(device=self.tpose_v.device)
        return m, s

# def reshape_sparse_coo(tensor, new_shape):
#     """
#     Args
#         tensor (torch.Tensor): torch sparse COO tensor
#         new_shape (tuple): new shape to reshape tensor
#     Return
#         new_sparse_tensor (torch.Tensor): reshaped torch sparse COO tensor
#     """
#     # Ensure the tensor is a sparse COO tensor
#     assert tensor.is_sparse, "Input tensor must be a sparse COO tensor"
    
#     # Coalesce the tensor to ensure its indices are unique and sorted
#     tensor = tensor.coalesce()
    
#     old_shape = tensor.shape
#     num_elements = torch.prod(torch.tensor(old_shape))
    
#     # Calculate the inferred dimension if -1 is present
#     inferred_shape = list(new_shape)
#     if -1 in inferred_shape:
#         inferred_index = inferred_shape.index(-1)
#         inferred_shape[inferred_index] = num_elements // -torch.prod(torch.tensor(inferred_shape))
#     new_shape = tuple(inferred_shape)
    
#     new_num_elements = torch.prod(torch.tensor(new_shape))
#     assert num_elements == new_num_elements, "The total number of elements must remain the same"
    
#     # Flatten the indices of the original sparse tensor
#     flat_indices = tensor.indices()[0] * old_shape[1] + tensor.indices()[1]
    
#     # Calculate new indices
#     new_indices = torch.stack([
#         (flat_indices // (new_shape[1] * new_shape[2])) % new_shape[0],
#         (flat_indices // new_shape[2]) % new_shape[1],
#         flat_indices % new_shape[2]
#     ])
    
#     # Create the new sparse tensor with the new shape
#     new_sparse_tensor = torch.sparse_coo_tensor(new_indices, tensor.values(), new_shape)
    
#     return new_sparse_tensor

def rnd_mask(B_skel, consq_n, mask_prob=0.5, edge_thres=4, demo=None):
    # mask single frame and repeat (to avoid flickering masks for the same joints among consecutive frames)
    device = B_skel.lo.device
    nV_sf = int(B_skel.lo.shape[0] / consq_n)
    mask_sf = torch.zeros((nV_sf,), device=device, dtype=torch.bool)

    if demo == "no_mask":
        return mask_sf.repeat(consq_n)

    nB = B_skel.batch.max() + 1
    nB_sf = int(nB / consq_n)
    edge_batch = B_skel.batch[B_skel.edge_index[0]]
    edge_index_sf = B_skel.edge_index[:, edge_batch < nB_sf]

    # find end-effector
    ee = find_ee(B_skel)
    n_limb = 5
    assert len(ee) == n_limb * nB
    ee_sf = ee[: nB_sf * n_limb].reshape(nB_sf, n_limb)

    # randomly select to mask or not
    do_mask = torch.rand(nB_sf) < mask_prob  # 50 %

    # randomly select one limb per skeleton (and the corresponding end-effectors)
    rnd_ith_limb = torch.randint(0, n_limb - 1, (nB_sf, 1)).to(device=device)
    rnd_ee_sf = torch.gather(ee_sf, 1, rnd_ith_limb).flatten()

    # randomly select mask depth
    # e.g.) 0: just end-effector, 1: end-effector and its parent, 2: end-effector, parent, and grandparent, ...
    mask_ee_reach = torch.randint(0, edge_thres, (len(rnd_ee_sf),))

    # find all joints to be masked
    mask_joints = []
    ee_ascend = rnd_ee_sf
    for i in range(max(mask_ee_reach) + 1):
        mask_joints.append(ee_ascend[do_mask & (i <= mask_ee_reach)])
        ee_ascend = edge_index_sf[0, ee_ascend]  # ee-> up to parent
    mask_joints = torch.concat(mask_joints)
    mask_sf[mask_joints] = True
    return mask_sf.repeat(consq_n)


def find_ee(skel_graph):
    edge_feature, edge_index = skel_graph.edge_feature, skel_graph.edge_index
    return edge_index[1, edge_feature[:, 1] == 0]


def find_feet(skel_graph):
    # CAUTION; this function assumes a single skel
    ee = find_ee(skel_graph)
    go = skel_graph.go
    left_foot, right_foot = None, None
    for foot in ee[torch.argsort(go[ee, 1])[:2]]:
        if go[foot, 0] > 0:
            left_foot = foot
        else:
            right_foot = foot
    assert left_foot != right_foot
    assert (left_foot is not None) and (right_foot is not None)
    return left_foot, right_foot
