import collections
import copy
import json
import os
import networkx as nx
import numpy as np
import numpy.linalg as LA
import scipy.io as sio
import cv2

def get_scene_list():
	return ['Rs_int', 'Beechwood_0_int', 'Beechwood_1_int', 'Benevolence_0_int', 'Benevolence_1_int', 
	'Benevolence_2_int', 'Ihlen_0_int', 'Ihlen_1_int', 'Merom_0_int', 'Merom_1_int', 'Pomaria_0_int', 
	'Pomaria_1_int', 'Pomaria_2_int', 'Wainscott_0_int', 'Wainscott_1_int']

# 100 pixel per meter
def pixelToWorldCoords(pixel, world_H, world_W, resolution=100):
	xv, yv = pixel
	x_coord = ((xv - world_W/2)/resolution)
	y_coord = ((yv - world_H/2)/resolution)
	return x_coord, y_coord


def create_folder(folder_name, clean_up=False):
  flag_exist = os.path.isdir(folder_name)
  if not flag_exist:
    print('{} folder does not exist, so create one.'.format(folder_name))
    os.makedirs(folder_name)
    #os.makedirs(os.path.join(test_case_folder, 'observations'))
  else:
    print('{} folder already exists, so do nothing.'.format(folder_name))

def cylindrical_panorama(img, focal_length=256):
	K = np.array([[focal_length, 0, focal_length], [0, focal_length, focal_length], [0, 0, 1]])

	"""This function returns the cylindrical warp for a given image and intrinsics matrix K"""
	h, w = img.shape[:2]
	# pixel coordinates
	# y_i: [[0, 0, 0, ..., 0],
	#		[1, 1, 1, ..., 1],
	#		...
	#		[511, 511, 511, ..., 511]]
	# x_i: [[0, 1, 2, ..., 511],
	#		...
	# 		[0, 1, 2, ..., 511]]
	y_i, x_i = np.indices((h, w))
	X = np.stack([x_i, y_i, np.ones_like(x_i)], axis=-1).reshape(h * w, 3) # to homog
	Kinv = np.linalg.inv(K) 
	X = Kinv.dot(X.T).T # normalized coords
	# calculate cylindrical coords (sin\theta, h, cos\theta)
	A = np.stack([np.sin(X[:, 0]), X[:, 1], np.cos(X[:, 0])], axis=-1).reshape(w * h, 3)
	# project back to image-pixels plane
	B = K.dot(A.T).T 
	# back from homog coords
	B = B[:, :-1] / B[:, [-1]]
	# make sure warp coords only within image bounds
	B[(B[:, 0] < 0) | (B[:, 0] >= w) | (B[:, 1] < 0) | (B[:, 1] >= h)] = -1
	B = B.reshape(h, w, -1)

	img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA) # for transparent borders...
	# warp the image according to cylindrical coords
	result =  cv2.remap(img_rgba, B[:, :, 0].astype(np.float32), B[:, :, 1].astype(np.float32), cv2.INTER_AREA, 
		borderMode=cv2.BORDER_TRANSPARENT)

	result = result[:, 56:512-56, :3]

	return result