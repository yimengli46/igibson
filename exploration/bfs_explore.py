from gibson2.envs.igibson_env import iGibsonEnv
import argparse
import numpy as np
import gym
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
import matplotlib.pyplot as plt
from math import pi
import cv2
import random
import networkx as nx
from utils import create_folder, get_scene_list, pixelToWorldCoords, cylindrical_panorama

scene_id = 0
igibson_folder = '/home/yimeng/Datasets/iGibson'

scene_list = get_scene_list()
scene = scene_list[scene_id]
my_data_folder = f'{igibson_folder}/my_data/{scene}'

theta_lst = [0, pi/2, pi, -0.5*pi]
#======================================== load the topo map ==========================================
topo_V_E = np.load(f'{my_data_folder}/v_and_e.npy', allow_pickle=True).item()
v_lst, e_lst = topo_V_E['vertices'], topo_V_E['edges']

# load trav_map
trav_map = cv2.imread('{}/{}/layout/{}.png'.format(
	f'{igibson_folder}/gibson2/data/ig_dataset/scenes',
	scene,
	'floor_trav_0_occupancy'), 0)
trav_map_H, trav_map_W = trav_map.shape

#===================================== BFS traverse the env ===========================================
start_v_idx = random.randint(0, len(v_lst)-1) 

# build a graph
G = nx.Graph()
for e in e_lst:
	u, v = e
	G.add_edge(u, v)

edges = nx.bfs_edges(G, start_v_idx)
# get the nodes in a bfs order
nodes = [start_v_idx] + [v for u, v in edges]

#================================= sample panorama observation at each node ============================
# generate observations
mode = 'headless'
config = f'{igibson_folder}/my_code/configs/config_{scene}.yaml'
nav_env = iGibsonEnv(config_file=config,
                     mode=mode,
                     action_timestep=1.0 / 120.0,
                     physics_timestep=1.0 / 120.0)

nav_env.reset()

count = 0
# 30 means 30*0.01m = 0.3m
for i in range(len(nodes)):
		vertex = v_lst[nodes[i]]
		x_coord, y_coord = pixelToWorldCoords(vertex, trav_map_H, trav_map_W)

		camera_pose = np.array([x_coord, y_coord, 0.0])

		panorama = np.zeros((512, 400*4, 3), dtype='uint8')
		imgs = np.zeros((512, 512*4, 3), dtype='uint8')

		for i_th, theta in enumerate(theta_lst):
			orn = (0, 0, theta)
			nav_env.set_pos_orn_with_z_offset(nav_env.robots[0], camera_pose, orn)

			state, _, _, _ = nav_env.step(4)
			img = state['rgb']
			
			img = (img*255).astype('uint8')
			
			j = 3 - i_th
			
			panorama[:, 400*j:400*(j+1), :] = cylindrical_panorama(img)
			imgs[:, 512*j:512*(j+1), :] = img
		#assert 1==2

		#plt.subplot(1, 2)
		#plt.imshow(panorama)
		#plt.show()
		fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(20, 8))
		ax[0].imshow(panorama)
		ax[1].imshow(imgs)
		fig.tight_layout()
		plt.show()

