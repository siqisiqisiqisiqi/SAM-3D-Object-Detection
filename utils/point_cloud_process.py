from typing import Tuple
import numpy as np
from numpy import ndarray

def point_cloud_process(point_cloud: ndarray)-> Tuple[ndarray, ndarray]:
    """point cloud coordinate transform and center calculation

    Parameters
    ----------
    point_cloud : ndarray
        The point cloud in the world coordinate system
        size [batchsize, point number, 3]

    Returns
    -------
    Tuple[ndarray, ndarray]
        point_cloud_trans: transformed point cloud
        size [batchsize, point number, 3]
        xyz_mean: point cloud mean point
        size [batchsize, 3]
    """
    num_point = point_cloud.shape[2]
    xyz_sum = point_cloud.sum(2, keepdim=True)
    xyz_mean = xyz_sum / num_point
    point_cloud_trans = point_cloud - xyz_mean.repeat(1, 1, num_point)
    xyz_mean = xyz_mean.squeeze()
    return point_cloud_trans, xyz_mean
