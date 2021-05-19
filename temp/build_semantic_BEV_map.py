'''
build top-down-view semantic map from depth and sseg egocentric observations.
'''

import numpy as np
import numpy.linalg as LA
import cv2
import matplotlib.pyplot as plt
import math
from math import cos, sin, acos, atan2, pi, floor
from utils import project_pixels_to_world_coords, convertInsSegToSSeg, apply_color_to_map, create_folder


scene_id = 1
scene_list = ['Rs_int', 'Beechwood_0_int']
scene = scene_list[scene_id]

dataset_dir = '/home/yimeng/Datasets/iGibson/my_data'
cell_size = 0.1
trav_map_cell_size = 0.01
proportion_cell_size = round(trav_map_cell_size / cell_size, 2)

class_mapper = np.load('{}/{}/class_mapper.npy'.format(dataset_dir, scene), allow_pickle=True).item()
num_classes = class_mapper[list(class_mapper.keys())[-1]] + 1
max_height = 5.0 # maximum height is 5.0 meter

#UNIGNORED_CLASS = [3, 4, 6, 7, 8, 9, 10]
saved_folder = '{}/{}/sem_occupancy_map_results'.format(dataset_dir, scene)
create_folder(saved_folder, clean_up=True)

step_size = 50
map_boundary = 10

IGNORED_CLASS = [5]
'''
for i in range(41):
	if i not in UNIGNORED_CLASS:
		IGNORED_CLASS.append(i)
'''

# load img list
poses_list = np.load('{}/{}/poses.npy'.format(dataset_dir, scene), allow_pickle=True)
img_names = list(range(poses_list.shape[0]))

cat_dict = np.load('{}/{}/class_mapper.npy'.format(dataset_dir, scene), allow_pickle=True).item()

# decide size of the grid
trav_map = cv2.imread('{}/{}/layout/{}.png'.format(
	'/home/yimeng/Datasets/iGibson/gibson2/data/ig_dataset/scenes',
	scene,
	'floor_trav_0'), 0)
H, W = trav_map.shape
min_X = -(W/2)*trav_map_cell_size
max_X = (W/2)*trav_map_cell_size
min_Z = -(H/2)*trav_map_cell_size
max_Z = (H/2)*trav_map_cell_size
four_dim_grid = np.zeros((int(H*proportion_cell_size), int(max_height/cell_size), int(W*proportion_cell_size), num_classes), dtype=np.int32) # x, y, z, C
H, W = four_dim_grid.shape[0], four_dim_grid.shape[2]


for idx, img_name in enumerate(img_names):
	#if idx == 100:
	#	break

	print('idx = {}'.format(idx))
	# load rgb image, depth and sseg
	rgb_img = cv2.imread('{}/{}/rgb/{}_rgb.png'.format(dataset_dir, scene, img_name), 1)[:, :, ::-1]
	depth_img = np.load('{}/{}/depth/{}_depth.npy'.format(dataset_dir, scene, img_name), allow_pickle=True)
	sseg_img = cv2.imread('{}/{}/sseg/{}_sseg.png'.format(dataset_dir, scene, img_name), 0)
	
	pose = poses_list[img_name] # x, z, theta
	print('pose = {}'.format(pose))

	if idx % step_size == 0:
		#'''
		fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 6))
		ax[0].imshow(rgb_img)
		ax[0].get_xaxis().set_visible(False)
		ax[0].get_yaxis().set_visible(False)
		ax[0].set_title("rgb")
		ax[1].imshow(apply_color_to_map(sseg_img, num_classes=num_classes))
		ax[1].get_xaxis().set_visible(False)
		ax[1].get_yaxis().set_visible(False)
		ax[1].set_title("sseg")
		ax[2].imshow(depth_img)
		ax[2].get_xaxis().set_visible(False)
		ax[2].get_yaxis().set_visible(False)
		ax[2].set_title("depth")
		fig.tight_layout()
		#plt.show()
		fig.savefig('{}/step_{}.jpg'.format(saved_folder, idx))
		plt.close()
		#assert 1==2
		#'''

	xyz_points, sseg_points = project_pixels_to_world_coords(sseg_img, depth_img, pose, gap=4, focal_length=256, resolution=512, ignored_classes=IGNORED_CLASS)

	mask_X = np.logical_and(xyz_points[0, :] > min_X, xyz_points[0, :] < max_X) 
	mask_Y = np.logical_and(xyz_points[1, :] > 0.0, xyz_points[1, :] < max_height)
	mask_Z = np.logical_and(xyz_points[2, :] > min_Z, xyz_points[2, :] < max_Z)  
	mask_XYZ = np.logical_and.reduce((mask_X, mask_Y, mask_Z))
	xyz_points = xyz_points[:, mask_XYZ]
	sseg_points = sseg_points[mask_XYZ]

	x_coord = np.floor((xyz_points[0, :] - min_X) / cell_size).astype(int)
	y_coord = np.floor(xyz_points[1, :] / cell_size).astype(int)
	z_coord = np.floor((xyz_points[2, :] - min_Z) / cell_size).astype(int)
	mask_y_coord = y_coord < int(max_height/cell_size)
	x_coord = x_coord[mask_y_coord]
	y_coord = y_coord[mask_y_coord]
	z_coord = z_coord[mask_y_coord]
	sseg_points = sseg_points[mask_y_coord]
	four_dim_grid[z_coord, y_coord, x_coord, sseg_points] += 1
	#assert 1==2

	# sum over the height axis
	grid_sum_height = np.sum(four_dim_grid, axis=1)

	# argmax over the category axis
	semantic_map = np.argmax(grid_sum_height, axis=2)
	#assert 1==2

	color_semantic_map = apply_color_to_map(semantic_map, num_classes=num_classes)

	if idx % step_size == 0:
		color_semantic_map = cv2.resize(color_semantic_map, (int(H/proportion_cell_size), int(W/proportion_cell_size)), interpolation = cv2.INTER_NEAREST)
		cv2.imwrite('{}/step_{}_semantic.jpg'.format(saved_folder, idx), color_semantic_map[:,:,::-1])

# save final color_semantic_map and semantic_map
color_semantic_map = apply_color_to_map(semantic_map, num_classes=num_classes)
color_semantic_map = cv2.resize(color_semantic_map, (int(H/proportion_cell_size), int(W/proportion_cell_size)), interpolation = cv2.INTER_NEAREST)
cv2.imwrite('{}/step_{}_semantic.jpg'.format(saved_folder, idx), color_semantic_map[:,:,::-1])

semantic_map = semantic_map.astype('uint')
semantic_map = cv2.resize(semantic_map, (int(H/proportion_cell_size), int(W/proportion_cell_size)), interpolation = cv2.INTER_NEAREST)
cv2.imwrite('{}/BEV_semantic_map.png'.format(saved_folder), semantic_map)