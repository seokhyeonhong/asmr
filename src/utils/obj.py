import numpy as np
import torch

class ObjMesh():
    def __init__(self, file_name=None):
        self.v = None
        self.vn = None
        self.vt = None
        
        self.f = None
        self.fn = None
        self.ft = None
        
        #self.edges = None
        
        if file_name is not None:
            self.load_file(file_name)
        else:
            print("empty ObjMesh!")
            
    def convert(self, mode='torch'):
        if mode=='torch' or mode == 'th':
            self.v  = torch.tensor(self.v )
            self.vn = torch.tensor(self.vn)
            self.vt = torch.tensor(self.vt)
            self.f  = torch.tensor(self.f )
            self.fn = torch.tensor(self.fn)
            self.ft = torch.tensor(self.ft)
        elif mode =='numpy' or mode=='np':
            self.v  = np.array(self.v )
            self.vn = np.array(self.vn)
            self.vt = np.array(self.vt)
            self.f  = np.array(self.f )
            self.fn = np.array(self.fn)
            self.ft = np.array(self.ft)
        
    def load_file(self, obj_file):
        vertex_position = []
        vertex_normal = []
        vertex_UV = []
        face_indices = []
        face_normal = []
        face_UV = []
        for line in open(obj_file, "r"):
            if line.startswith('#'):
                continue
            values = line.split()
            if not values:
                continue
            if values[0] == 'v':
                v = list(map(float, values[1:]))
                vertex_position.append(v)
            if values[0] == 'vn':
                vn = list(map(float, values[1:]))
                vertex_normal.append(vn)
            if values[0] == 'vt':
                vt = list(map(float, values[1:]))
                vertex_UV.append(vt)
            if values[0] == 'f':
                # import pdb; pdb.set_trace()
                f = list(map(lambda x: int(x.split('/')[0]),  values[1:]))
                face_indices.append(f)
                if len(values[1].split('/')) >=2:
                    ft = list(map(lambda x: int(x.split('/')[1]),  values[1:]))
                    face_UV.append(ft)
                if len(values[1].split('/')) >=3:
                    ft = list(map(lambda x: int(x.split('/')[2]),  values[1:]))
                    face_normal.append(ft)
            
        self.v = np.array(vertex_position)
        self.vt = np.array(vertex_UV)
        self.vn = np.array(vertex_normal)
        
        self.f = np.array(face_indices) -1
        self.ft = np.array(face_UV) -1
        self.fn = np.array(face_normal) -1