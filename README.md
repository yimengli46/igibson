## Build a Semantic Topological Map on iGibson

1. First download the github repo under branch 'jana'.

Put it directly under iGibson folder.

2. Create another folder under iGibson folder called 'my_data'.

    mkdir my_data

3. Go to 'my_code' folder.

Generate observations at densely sampled locations,

Change line 16 igibson_folder to be where the iGibson folder is on your machine.
    
    python temp/gen_observations.py

4. Generate topological map on an occupancy map

    python temp/gen_topo_maps.py.
