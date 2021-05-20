## Build a Semantic Topological Map on iGibson

1. First download the github repo under branch 'jana'. Put it directly under iGibson folder.

2. Create another folder under iGibson folder called 'my_data'.
```bash
mkdir my_data
```

3. Go to 'my_code' folder.
Generate observations at densely sampled locations,

Change line 16 igibson_folder to be where the iGibson folder is on your machine.
```bash
python temp/gen_observations.py
```
4. Generate topological map on an occupancy map.
 
Change line 13 igibson_folder to be where the iGibson folder is on your machine.
```bash
python temp/gen_topo_maps.py.
```
5. Generate top-down view semantic map.

Change line 15 igibson_folder to be where the iGibson folder is on your machine.
```bash
python temp/build_semantic_BEV_map.py.
```
6. Generate topological semantic map.

Change line 15 igibson_folder to be where the iGibson folder is on your machine.
```bash
python temp/gen_semantic_topo_map.py.
```
