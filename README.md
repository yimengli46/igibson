## Build a Semantic Topological Map on iGibson

1. First download the github repo under branch 'jana'. Put it directly under iGibson folder.

2. Create another folder under iGibson folder called 'my_data'.
```bash
mkdir my_data
```

3. Go to 'my_code' folder. 

4. `temp/gen_observations.py` will generate the following things and saved in `my_data/Rs_int` folder. 
- `*.png`, egocentric rgb, depth and semantic segmentation observations
- `class_mapper.npy`, semantic label mapping from index to name
- `poses.npy`, 3d pose `(x,y,z)` in the world coordinates system where the observations are taken

Change line 16 `igibson_folder` to be where the iGibson folder is on your machine.
```bash
python temp/gen_observations.py
```

5. `temp/gen_topo_maps.py` will generate the following things and saved in `my_data/Rs_int` folder.
- `topo_map.png`, visualization of the built topological map
- `v_and_e.npy`, vertices and edges of the built topological map

Change line 13 `igibson_folder` to be where the iGibson folder is on your machine.
```bash
python temp/gen_topo_maps.py.
```


6. `temp/build_semantic_BEV_map.py` will generatethe following things and saved in `my_data/Rs_int/sem_occupancy_map_results` folder.
- `step_xx.jpg`, visualization of the egocentric observations at step xx.
- `step_xx_semantic.jpg`, visualization of the built top-down view semantic map at step xx.
- `BEV_semantic_map.png`, built top-down view semantic map. pixel value denotes the class label.

Change line 15 `igibson_folder` to be where the iGibson folder is on your machine.
```bash
python temp/build_semantic_BEV_map.py.
```

7. `temp/gen_semantic_topo_map.py` will generatethe following things and saved in `my_data/Rs_int` folder.
- `topo_semantic_map.png`, visualization of the built topological semantic map
Change line 15 `igibson_folder` to be where the iGibson folder is on your machine.
```bash
python temp/gen_semantic_topo_map.py.
```
