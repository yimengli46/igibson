'''
	generate egocentric rgb, depth, sseg given an occupancy map
'''

from gibson2.envs.igibson_env import iGibsonEnv
import argparse
import numpy as np
import gym
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
import matplotlib.pyplot as plt
from math import pi
import cv2
from utils import create_folder

scene_id = 0
igibson_folder = '/home/yimeng/Datasets/iGibson'

scene_list = ['Rs_int', 'Beechwood_0_int', 'Beechwood_1_int', 'Benevolence_0_int', 'Benevolence_1_int', 
	'Benevolence_2_int', 'Ihlen_0_int', 'Ihlen_1_int', 'Merom_0_int', 'Merom_1_int', 'Pomaria_0_int', 
	'Pomaria_1_int', 'Pomaria_2_int', 'Wainscott_0_int', 'Wainscott_1_int']
scene = scene_list[scene_id]

theta_lst = [0, pi/2, pi, 1.5*pi]
saved_folder = f'{igibson_folder}/my_data/{scene}'
create_folder(saved_folder)
create_folder(f'{saved_folder}/rgb', clean_up=True)
create_folder(f'{saved_folder}/depth', clean_up=True)
create_folder(f'{saved_folder}/sseg', clean_up=True)

# load trav_map
trav_map = cv2.imread('{}/{}/layout/{}.png'.format(
	f'{igibson_folder}/gibson2/data/ig_dataset/scenes',
	scene,
	'floor_trav_0_occupancy'), 0)
H, W = trav_map.shape
kernel = np.ones((5, 5),np.uint8)
trav_map = cv2.erode(trav_map, kernel, iterations=1)
x = np.linspace(0, W-1, W)
y = np.linspace(0, H-1, H)
xv, yv = np.meshgrid(x, y)
# map resolution is 1pixel per 0.1m
x_coord = ((xv - W/2)*0.01)
y_coord = ((yv - H/2)*0.01)
assert x_coord.shape[0] == y_coord.shape[0]

# generate observations
mode = 'headless'
config = f'{igibson_folder}/my_code/configs/config_{scene}.yaml'
nav_env = iGibsonEnv(config_file=config,
                     mode=mode,
                     action_timestep=1.0 / 120.0,
                     physics_timestep=1.0 / 120.0)

nav_env.reset()

poses = []

count = 0
# 30 means 30*0.01m = 0.3m
for i in range(0, x_coord.shape[0], 30):
	for j in range(0, x_coord.shape[1], 30):
		print(f'i = {i}, j = {j}')
		camera_pose = np.array([x_coord[i][j], y_coord[i][j], 0.0])
		#flag_free = nav_env.test_valid_position(nav_env.robots[0], camera_pose)
		flag_free = (trav_map[i][j] > 0)

		# if the position is free
		if flag_free:
			for i_th, theta in enumerate(theta_lst):
				orn = (0, 0, theta)
				nav_env.set_pos_orn_with_z_offset(nav_env.robots[0], camera_pose, orn)

				state, _, _, _ = nav_env.step(4)
				img = state['rgb']
				depth = state['depth']
				sseg = state['seg']

				sum_img = np.sum((img[:,:,0] > 0))
				h, w = img.shape[:2]
				
				if sum_img > h*w*0.75:
					img = (img*255).astype('uint')
					depth = depth * 50.0
					sseg = (sseg[:,:,0]*255).astype('uint')

					cv2.imwrite(f'{saved_folder}/rgb/{count}_rgb.png', img[:,:,::-1])
					np.save(f'{saved_folder}/depth/{count}_depth.npy', depth)
					cv2.imwrite(f'{saved_folder}/sseg/{count}_sseg.png', sseg)
				
					count += 1

					pose = (x_coord[i][j], y_coord[i][j], theta)
					poses.append(pose)

					#plt.imshow(img)
					#plt.show()
					#assert 1==2

poses = np.array(poses)
np.save(f'{saved_folder}/poses.npy', poses)


from gibson2.utils.semantics_utils import get_class_name_to_class_id
dict_class2id = get_class_name_to_class_id()
dict_class2id['walls'] = 3
np.save(f'{saved_folder}/class_mapper.npy', dict_class2id)