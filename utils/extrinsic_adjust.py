from scipy.spatial.transform import Rotation as R
import numpy as np
from configparser import ConfigParser


def main():
    """Slightly change the camera extrinsic parameters
    """
    with np.load('camera_params/E.npz') as X:
        mtx, dist, Mat, tvecs = [X[i] for i in ('mtx', 'dist', 'Mat', 'tvecs')]

    tvecs[1, 0] = -1 * tvecs[1, 0]
    tvecs[2, 0] = -1 * tvecs[2, 0]

    print(tvecs)

    r = R.from_euler('y', -2.0, degrees=True)
    R_euler = r.as_matrix()
    Mat2 = Mat @ R_euler

    r = R.from_euler('x', -3.0, degrees=True)
    R_euler = r.as_matrix()
    Mat3 = Mat2 @ R_euler

    r = R.from_euler('z', 2.5, degrees=True)
    R_euler = r.as_matrix()
    Mat4 = Mat3 @ R_euler

    config_object = ConfigParser()
    config_object.read(f"camera_params/SN36077403.conf")
    intri_param = config_object["LEFT_CAM_HD"]

    fx = float(intri_param["fx"])
    fy = float(intri_param["fy"])
    cx = float(intri_param["cx"])
    cy = float(intri_param["cy"])
    mtx = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    np.savez('camera_params/Ext2.npz', mtx=mtx,
             dist=dist, Mat=Mat4, tvecs=tvecs)


if __name__ == "__main__":
    main()
