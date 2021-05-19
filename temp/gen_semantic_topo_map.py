'''
combine topological map with semantic map.
'''

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from utils import apply_color_to_map
import skimage.measure
from math import floor

scene_id = 0
scene_list = ['Rs_int', 'Beechwood_0_int']
scene = scene_list[scene_id]

saved_folder = '/home/yimeng/Datasets/iGibson/my_data/{}'.format(scene)

cat_dict = np.load('{}/class_mapper.npy'.format(saved_folder), allow_pickle=True).item()
num_classes = cat_dict[list(cat_dict.keys())[-1]] + 1


#======================================== load scene occupancy map ====================================
# load trav_map
occupancy_map = cv2.imread('{}/{}/layout/{}.png'.format(
	'/home/yimeng/Datasets/iGibson/gibson2/data/ig_dataset/scenes',
	scene,
	'floor_trav_0_occupancy'), 0)
H, W = occupancy_map.shape
x = np.linspace(0, W-1, W)
y = np.linspace(0, H-1, H)
xv, yv = np.meshgrid(x, y)

#======================================== load the semantic map =======================================
semantic_map = cv2.imread('{}/sem_occupancy_map_results/BEV_semantic_map.png'.format(saved_folder), 0)

semantic_occupancy_map = occupancy_map.copy()
semantic_occupancy_map[semantic_occupancy_map == 255] = 1 # change free space into label 1
mask_semantics = np.logical_and(semantic_map > 0, semantic_map != 4) # 0 is back ground, 4 is floor
semantic_occupancy_map[mask_semantics] = semantic_map[mask_semantics]

#======================================== load the topo map ==========================================
topo_V_E = np.load('{}/v_and_e.npy'.format(saved_folder), allow_pickle=True).item()
v_lst, e_lst = topo_V_E['vertices'], topo_V_E['edges']

color_semantic_map = apply_color_to_map(semantic_occupancy_map, num_classes=num_classes)

#'''
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
ax.imshow(color_semantic_map)
x, y = [], []
for ed in e_lst:
	v1 = v_lst[ed[0]]
	v2 = v_lst[ed[1]]
	y.append(v1[1])
	x.append(v1[0])
	y.append(v2[1])
	x.append(v2[0])
	ax.plot([v1[0], v2[0]], [v1[1], v2[1]], 
            'k-', lw=1)
ax.scatter(x=x, y=y, c='r', s=2)
fig.tight_layout()
#'''

#====================================== compute centers of semantic classes =====================================
idx2cat_dict = {v: k for k, v in cat_dict.items()}
IGNORED_CLASS = [0,1,2, 3,4,5]
cat_binary_map = semantic_occupancy_map.copy()
for cat in IGNORED_CLASS:
	cat_binary_map = np.where(cat_binary_map==cat, -1, cat_binary_map)
# run skimage to find the number of objects belong to this class
instance_label, num_ins = skimage.measure.label(cat_binary_map, background=-1, connectivity=1, return_num=True)

list_instances = []
for idx_ins in range(1, num_ins+1):
	mask_ins = (instance_label==idx_ins)
	if np.sum(mask_ins) > 50: # should have at least 50 pixels
		print('idx_ins = {}'.format(idx_ins))
		x_coords = xv[mask_ins]
		y_coords = yv[mask_ins]
		ins_center = (floor(np.median(x_coords)), floor(np.median(y_coords)))
		ins_cat = semantic_occupancy_map[int(y_coords[0]), int(x_coords[0])]
		ins = {}
		ins['center'] = ins_center
		ins['cat'] = ins_cat
		list_instances.append(ins)

#================================== link the instances to the topo nodes ======================================
v_arr = np.array(v_lst)
x, y = [], []
for ins in list_instances:
	center = ins['center']
	cat = ins['cat']

	x.append(center[0])
	y.append(center[1])

	dist = np.sqrt((center[0] - v_arr[:, 0])**2 + (center[1] - v_arr[:, 1])**2)
	closest_v_idx = np.argmin(dist)
	vertex = v_lst[closest_v_idx]

	ax.plot([center[0], vertex[0]], [center[1], vertex[1]], 
            'k-', lw=1)

	try:
		cat_name = idx2cat_dict[cat]
	except:
		cat_name = 'unknown'
	ax.text(center[0], center[1], cat_name)

ax.scatter(x=x, y=y, c='b', s=5)

#assert 1==2
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
fig.savefig('{}/topo_semantic_map.png'.format(saved_folder))
plt.close()
#plt.show() 
