from third_party.pointnet2.pointnet2_modules import PointnetSAModuleVotes
from third_party.pointnet2.pointnet2_utils import furthest_point_sample

def build_preencoder():
    mlp_dims = [0, 64, 128, 256]
    preencoder = PointnetSAModuleVotes(
        radius=0.2,
        nsample=64,
        npoint=1024,
        mlp=mlp_dims,
        normalize_xyz=True,
    )
    return preencoder

build_preencoder()