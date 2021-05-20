'''
	generate topological map vertices and edges given an occupancy map
'''

import numpy as np
import matplotlib.pyplot as plt
import os
from topo_map.img_to_skeleton import img2skeleton
from topo_map.skeleton_to_topoMap import skeleton2topoMap
from topo_map.utils import drawtoposkele_with_VE,  build_VE_from_graph

scene_id = 0
igibson_folder = '/home/yimeng/Datasets/iGibson'

scene_list = ['Rs_int', 'Beechwood_0_int', 'Beechwood_1_int', 'Benevolence_0_int', 'Benevolence_1_int', 
	'Benevolence_2_int', 'Ihlen_0_int', 'Ihlen_1_int', 'Merom_0_int', 'Merom_1_int', 'Pomaria_0_int', 
	'Pomaria_1_int', 'Pomaria_2_int', 'Wainscott_0_int', 'Wainscott_1_int']
scene = scene_list[scene_id]

saved_folder = f'{igibson_folder}/my_data/{scene}'

print(f'scene = {scene}')
gray1, gau1, skeleton = img2skeleton('{}/{}/layout/{}.png'.format(
	f'{igibson_folder}/gibson2/data/ig_dataset/scenes',
	scene,
	'floor_trav_0_occupancy'))
graph = skeleton2topoMap(skeleton)

v_lst, e_lst = build_VE_from_graph(graph, skeleton, vertex_dist=10)

fig = plt.figure(frameon=False)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.xaxis.set_major_locator(plt.NullLocator())
ax.yaxis.set_major_locator(plt.NullLocator())
drawtoposkele_with_VE(graph, skeleton + (1 - gray1[:, :] / 255) * 2, v_lst, e_lst, ax=ax)
fig.savefig(f'{saved_folder}/{scene}_topo_map.png', format='png', dpi=500, bbox_inches='tight', pad_inches=0)

result = {}
result['vertices'] = v_lst
result['edges'] = e_lst

np.save(f'{saved_folder}/v_and_e.npy', result)
