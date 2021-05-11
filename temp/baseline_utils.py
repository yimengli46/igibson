import collections
import copy
import json
import os
import networkx as nx
import numpy as np
import numpy.linalg as LA
import scipy.io as sio
import cv2
import math
from math import cos, sin, acos, atan2, pi
from io import StringIO

def minus_theta_fn(previous_theta, current_theta):
  result = current_theta - previous_theta
  if result < -math.pi:
    result += 2*math.pi
  if result > math.pi:
    result -= 2*math.pi
  return result

def project_pixels_to_camera_coords (sseg_img, current_depth, current_pose, gap=2, focal_length=128, resolution=256, ignored_classes=[]):
  ## camera intrinsic matrix
  K = np.array([[focal_length, 0, focal_length], [0, focal_length, focal_length], [0, 0, 1]])
  inv_K = LA.inv(K)
  ## first compute the rotation and translation from current frame to goal frame
  ## then compute the transformation matrix from goal frame to current frame
  ## thransformation matrix is the camera2's extrinsic matrix
  tx, tz, theta = current_pose
  R = np.array([[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]])
  T = np.array([tx, 0, tz])
  transformation_matrix = np.empty((3, 4))
  transformation_matrix[:3, :3] = R
  transformation_matrix[:3, 3] = T
  
  # build the point matrix
  coords_range = range(0, resolution, gap)
  xv, yv = np.meshgrid(np.array(coords_range), np.array(coords_range))
  Z = current_depth[yv.flatten(), xv.flatten()].reshape(len(coords_range), len(coords_range))
  points_4d = np.ones((len(coords_range), len(coords_range), 4), np.float32)
  points_4d[:, :, 0] = xv
  points_4d[:, :, 1] = yv
  points_4d[:, :, 2] = Z
  points_4d = np.transpose(points_4d, (2, 0, 1)).reshape((4, -1)) # 4 x N

  # apply intrinsic matrix
  points_4d[[0, 1, 3], :] = inv_K.dot(points_4d[[0, 1, 3], :])
  points_4d[0, :] = points_4d[0, :] * points_4d[2, :]
  points_4d[1, :] = points_4d[1, :] * points_4d[2, :]

  ## transform kp1_4d from camera1(current) frame to camera2(goal) frame through transformation matrix
  print('points_4d.shape = {}'.format(points_4d.shape))
  points_3d = points_4d[:3, :]
  print('points_3d.shape = {}'.format(points_3d.shape))

  ## pick x-row and z-row
  sseg_points = sseg_img[yv.flatten(), xv.flatten()].flatten()

  # ignore some classes points
  #print('sseg_points.shape = {}'.format(sseg_points.shape))
  for c in ignored_classes:
    good = (sseg_points != c)
    sseg_points = sseg_points[good]
    points_3d = points_3d[:, good]
  #print('after: sseg_points.shape = {}'.format(sseg_points.shape))
  #print('after: points_3d.shape = {}'.format(points_3d.shape))

  return points_3d, sseg_points.astype(int)


def project_pixels_to_world_coords (sseg_img, current_depth, current_pose, gap=2, focal_length=128, resolution=256, ignored_classes=[]):
  ## camera intrinsic matrix
  K = np.array([[focal_length, 0, focal_length], [0, focal_length, focal_length], [0, 0, 1]])
  inv_K = LA.inv(K)
  ## first compute the rotation and translation from current frame to goal frame
  ## then compute the transformation matrix from goal frame to current frame
  ## thransformation matrix is the camera2's extrinsic matrix
  tx, tz, theta = current_pose
  theta = -(theta + 1.5 * pi)
  R = np.array([[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]])
  T = np.array([tx, 0, tz])
  transformation_matrix = np.empty((3, 4))
  transformation_matrix[:3, :3] = R
  transformation_matrix[:3, 3] = T
  
  # build the point matrix
  coords_range = range(0, resolution, gap)
  xv, yv = np.meshgrid(np.array(coords_range), np.array(coords_range))
  Z = current_depth[yv.flatten(), xv.flatten()].reshape(len(coords_range), len(coords_range))
  points_4d = np.ones((len(coords_range), len(coords_range), 4), np.float32)
  points_4d[:, :, 0] = xv
  points_4d[:, :, 1] = yv
  points_4d[:, :, 2] = Z
  points_4d = np.transpose(points_4d, (2, 0, 1)).reshape((4, -1)) # 4 x N

  # apply intrinsic matrix
  points_4d[[0, 1, 3], :] = inv_K.dot(points_4d[[0, 1, 3], :])
  points_4d[0, :] = points_4d[0, :] * points_4d[2, :]
  points_4d[1, :] = points_4d[1, :] * points_4d[2, :]

  ## transform kp1_4d from camera1(current) frame to camera2(goal) frame through transformation matrix
  points_3d = transformation_matrix.dot(points_4d)

  ## pick x-row and z-row
  sseg_points = sseg_img[yv.flatten(), xv.flatten()].flatten()

  # ignore some classes points
  #print('sseg_points.shape = {}'.format(sseg_points.shape))
  for c in ignored_classes:
    good = (sseg_points != c)
    sseg_points = sseg_points[good]
    points_3d = points_3d[:, good]
  #print('after: sseg_points.shape = {}'.format(sseg_points.shape))
  #print('after: points_3d.shape = {}'.format(points_3d.shape))

  return points_3d, sseg_points.astype(int)

def convertInsSegToSSeg (InsSeg, ins2cat_dict):
  ins_id_list = list(ins2cat_dict.keys())
  SSeg = np.zeros(InsSeg.shape, dtype=np.int32)
  for ins_id in ins_id_list:
    SSeg = np.where(InsSeg==ins_id, ins2cat_dict[ins_id], SSeg)

  return SSeg


d3_41_colors_rgb: np.ndarray = np.array(
    [
      [0, 0, 0],
      [255, 255, 255],
      [180, 120, 120],
      [6, 230, 230],
      [80, 50, 50],
      [4, 200, 3],
      [120, 120, 80],
      [140, 140, 140],
      [204, 5, 255],
      [230, 230, 230],
      [4, 250, 7],
      [224, 5, 255],
      [235, 255, 7],
      [150, 5, 61],
      [120, 120, 70],
      [8, 255, 51],
      [255, 6, 82],
      [143, 255, 140],
      [204, 255, 4],
      [255, 51, 7],
      [204, 70, 3],
      [0, 102, 200],
      [61, 230, 250],
      [255, 6, 51],
      [11, 102, 255],
      [255, 7, 71],
      [255, 9, 224],
      [9, 7, 230],
      [220, 220, 220],
      [255, 9, 92],
      [112, 9, 255],
      [8, 255, 214],
      [7, 255, 224],
      [255, 184, 6],
      [10, 255, 71],
      [255, 41, 10],
      [7, 255, 255],
      [224, 255, 8],
      [102, 8, 255],
      [255, 61, 6],
      [255, 194, 7],
      [255, 122, 8],
      [0, 255, 20],
      [255, 8, 41],
      [255, 5, 153],
      [6, 51, 255],
      [235, 12, 255],
      [160, 150, 20],
      [0, 163, 255],
      [140, 140, 140],
      [250, 10, 15],
      [20, 255, 0],
      [31, 255, 0],
      [255, 31, 0],
      [255, 224, 0],
      [153, 255, 0],
      [0, 0, 255],
      [255, 71, 0],
      [0, 235, 255],
      [0, 173, 255],
      [31, 0, 255],
      [11, 200, 200],
      [255, 82, 0],
      [0, 255, 245],
      [0, 61, 255],
      [0, 255, 112],
      [0, 255, 133],
      [255, 0, 0],
      [255, 163, 0],
      [255, 102, 0],
      [194, 255, 0],
      [0, 143, 255],
      [51, 255, 0],
      [0, 82, 255],
      [0, 255, 41],
      [0, 255, 173],
      [10, 0, 255],
      [173, 255, 0],
      [0, 255, 153],
      [255, 92, 0],
      [255, 0, 255],
      [255, 0, 245],
      [255, 0, 102],
      [255, 173, 0],
      [255, 0, 20],
      [255, 184, 184],
      [0, 31, 255],
      [0, 255, 61],
      [0, 71, 255],
      [255, 0, 204],
      [0, 255, 194],
      [0, 255, 82],
      [0, 10, 255],
      [0, 112, 255],
      [51, 0, 255],
      [0, 194, 255],
      [0, 122, 255],
      [0, 255, 163],
      [255, 153, 0],
      [0, 255, 10],
      [255, 112, 0],
      [143, 255, 0],
      [82, 0, 255],
      [163, 255, 0],
      [255, 235, 0],
      [8, 184, 170],
      [133, 0, 255],
      [0, 255, 92],
      [184, 0, 255],
      [255, 0, 31],
      [0, 184, 255],
      [0, 214, 255],
      [255, 0, 112],
      [92, 255, 0],
      [0, 224, 255],
      [112, 224, 255],
      [70, 184, 160],
      [163, 0, 255],
      [153, 0, 255],
      [71, 255, 0],
      [255, 0, 163],
      [255, 204, 0],
      [255, 0, 143],
      [0, 255, 235],
      [133, 255, 0],
      [255, 0, 235],
      [245, 0, 255],
      [255, 0, 122],
      [255, 245, 0],
      [10, 190, 212],
      [214, 255, 0],
      [0, 204, 255],
      [20, 0, 255],
      [255, 255, 0],
      [0, 153, 255],
      [0, 41, 255],
      [0, 255, 204],
      [41, 0, 255],
      [41, 255, 0],
      [173, 0, 255],
      [0, 245, 255],
      [71, 0, 255],
      [122, 0, 255],
      [0, 255, 184],
      [0, 92, 255],
      [184, 255, 0],
      [0, 133, 255],
      [255, 214, 0],
      [25, 194, 194],
      [102, 255, 0],
      [92, 0, 255],
      [120, 120, 120],
    ],
    dtype=np.uint8,
)

def apply_color_to_map (semantic_map, num_classes=41):
  assert len(semantic_map.shape) == 2
  H, W = semantic_map.shape
  color_semantic_map = np.zeros((H, W, 3), dtype='uint8')
  for i in range(num_classes):
    color_semantic_map[semantic_map==i] = d3_41_colors_rgb[i]
  return color_semantic_map

def apply_color_to_pointCloud (sseg_points, num_classes=41):
  assert len(sseg_points.shape) == 1
  N = sseg_points.shape[0]
  color_sseg_points = np.zeros((N, 3), dtype='uint8')
  for i in range(num_classes):
    color_sseg_points[sseg_points==i] = d3_41_colors_rgb[i]
  return color_sseg_points