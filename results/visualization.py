import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
PARENT_DIR = os.path.dirname(ROOT_DIR)
sys.path.append(ROOT_DIR)

import torch
import numpy as np
from numpy import ndarray
from torch.utils.data import DataLoader
import cv2

from models.amodal_3D_model import Amodal3DModel
from utils.stereo_custom_dataset import StereoCustomDataset
from src.params import *

BS = 32
LABEL_VISUAL = False

pc_path = os.path.join(PARENT_DIR, "datasets", "pointclouds")
label_path = os.path.join(PARENT_DIR, "datasets", "labels")
save_path = os.path.join(ROOT_DIR, "results")


def visaulization(img_dir_tuple: tuple, corners: ndarray):
    """Draw the 3D bounding box in the 2D image

    Parameters
    ----------
    img_dir_tuple : tuple
        Tuple contains the image directory
    corners : ndarray  
        Coordinates of corner in world reference frame. 
        Size BSx8x3
    """
    # load the camera parameters
    with np.load('camera_params/Ext2.npz') as X:
        mtx, _, Mat, tvecs = [X[i] for i in ('mtx', 'dist', 'Mat', 'tvecs')]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
              (0, 255, 255), (255, 0, 255)]
    for i in range(BS):
        img_dir = img_dir_tuple[i]
        img = cv2.imread(img_dir)
        corner_world = corners[i]
        corner_camera = Mat @ (corner_world.T) + tvecs
        corner_image = (mtx @ corner_camera).T
        corner = corner_image[:, :2] / corner_image[:, 2:3]
        corner = corner.astype(int)
        # TODO: debug why the forward is not right
        corner[:, 0] = 1293 - corner[:, 0]

        corner1 = corner[:4, :]
        corner2 = corner[4:8, :]
        pt1 = corner1.reshape((-1, 1, 2))
        pt2 = corner2.reshape((-1, 1, 2))
        # rospy.loginfo(f"the corners are {corner}")

        index = i
        while index > 4:
            index = index - 5
        color = colors[index]
        thickness = 2
        cv2.polylines(img, [pt1], True, color, thickness)
        cv2.polylines(img, [pt2], True, color, thickness)
        for i, j in zip(range(4), range(4, 8)):
            cv2.line(img, tuple(corner[i]), tuple(corner[j]), color, thickness)

        # option 2 drawing
        index1 = [1, 0, 4, 5]
        index2 = [0, 3, 7, 4]
        index3 = [2, 3, 7, 6]
        index4 = [1, 2, 6, 5]
        zero1 = np.zeros((img.shape), dtype=np.uint8)
        zero2 = np.zeros((img.shape), dtype=np.uint8)
        zero3 = np.zeros((img.shape), dtype=np.uint8)
        zero4 = np.zeros((img.shape), dtype=np.uint8)
        zero_mask1 = cv2.fillConvexPoly(zero1, corner[index1, :], color)
        zero_mask2 = cv2.fillConvexPoly(zero2, corner[index2, :], color)
        zero_mask3 = cv2.fillConvexPoly(zero3, corner[index3, :], color)
        zero_mask4 = cv2.fillConvexPoly(zero4, corner[index4, :], color)
        zeros_mask = np.array(
            (zero_mask1 + zero_mask2 + zero_mask3 + zero_mask4))

        alpha = 1
        beta = 0.55
        gamma = 0
        mask_img = cv2.addWeighted(img, alpha, zeros_mask, beta, gamma)
        cv2.imshow("Image", mask_img)
        cv2.waitKey(0)


def calculate_corner(centers: ndarray, sizes: ndarray) -> ndarray:
    """calculate the corners coordinate from the center and size

    Parameters
    ----------
    centers : ndarray
        The center cooridate of the 3D bounding box
        Size BSx3
    sizes : ndarray
        Length, width, and height of the 3D bounding box
        Size BSx3
    Returns
    -------
    ndarray
        The corner coordinates in the world reference frame
    """
    corners = []
    for i in range(BS):
        center = centers[i]
        size = sizes[i]
        l, w, h = size[0], size[1], size[2]
        corner = []
        s1 = [-1 / 2 * l, 1 / 2 * l, 1 / 2 * l, -1 / 2 * l]
        s2 = [-1 / 2 * w, -1 / 2 * w, 1 / 2 * w, 1 / 2 * w]
        for delta_z in (-1 / 2 * h, 1 / 2 * h):
            for delta_x, delta_y in zip(s1, s2):
                corner_point = center + np.array([delta_x, delta_y, delta_z])
                corner.append(corner_point)
        corners.append(corner)
    corners = np.array(corners)
    return corners


def visualize_labeled_data(label_dicts: dict, img_dir_tuple: tuple):
    """Visualize the labeled 3D bounding box in the 2D image

    Parameters
    ----------
    label_dicts : dict
        Labeled 3D bounding box  
    img_dir_tuple : tuple
        Tuple contains the image directory
    """
    box3d_center_label = label_dicts.get('box3d_center')
    size_class_label = label_dicts.get('size_class')
    size_residual_label = label_dicts.get('size_residual')
    box3d_center = box3d_center_label.detach().cpu().numpy()
    size_class = size_class_label.detach().cpu().numpy()
    size_residual = size_residual_label.detach().cpu().numpy()
    size_label = []
    for i in range(BS):
        size = g_type_mean_size[g_class2type[size_class[i, 0]]]
        size = size + size_residual[i]
        size_label.append(size)
    corners_label = calculate_corner(
        box3d_center / 100, np.array(size_label) / 100)
    visaulization(img_dir_tuple, corners_label)


def main():
    print("Press SPACE to switch the image!")
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dataset = StereoCustomDataset(pc_path, label_path)
    dataloader = DataLoader(
        dataset, batch_size=BS, shuffle=True, num_workers=1, drop_last=True)

    model = Amodal3DModel()
    model.to(device)

    result_path = f"{save_path}/1015/1015_epoch40.pth"
    result = torch.load(result_path)
    model_state_dict = result['model_state_dict']

    model.load_state_dict(model_state_dict)
    model.eval()

    features, label_dicts, img_dir_tuple = next(iter(dataloader))
    features = features.to(device, dtype=torch.float)
    data_dicts_var = {key: value.cuda() for key, value in label_dicts.items()}

    with torch.no_grad():
        _, metrics = model(features, data_dicts_var)
    corners = metrics['corners']

    if LABEL_VISUAL is not True:
        visaulization(img_dir_tuple, corners / 100)
        print("Visualize the labeled data.")
    else:
        visualize_labeled_data(label_dicts, img_dir_tuple)
        print("Visualize the predict result.")


if __name__ == "__main__":
    main()
