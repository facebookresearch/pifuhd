# PIFuHD: Multi-Level Pixel-Aligned Implicit Function for High-Resolution 3D Human Digitization

This repository contains a pytorch implementation of "Multi-Level Pixel-Aligned Implicit Function for High-Resolution 3D Human Digitization".

This codebase provides: 
- test code
- training code
- visualization code
- evaluation code

## Requirements
- Python 3
- [PyTorch](https://pytorch.org/) tested on 1.1.0
- json
- PIL
- skimage
- tqdm
- [trimesh](https://trimsh.org/) with pyembree
- cv2
- tensorboard
- tqdm
- OpenGL (for rendering)
- tinyobjloader (need to compile from source, but only for rendering)



## Download Pre-trained model and test data

Pre-trained model: https://www.dropbox.com/s/8qsvbeq9tfbq3ji/checkpoints.zip?dl=0

Test data (Images + Keypoint detections): https://www.dropbox.com/s/2s9nk51ebvb5ibl/input_test.zip?dl=0

## A Quick Testing
Open apps/simple_test.py and set `input_path` (the test data folder you downloaded above with jpg and json files), `output_path`, and `checkpoint_path`. Run the code:
```
python -m apps.simple_test
```

Your output will be saved in `output_path`.  
You may use meshlab (http://www.meshlab.net/) to visualize the 3D mesh output (obj file). 


## Testing
1. run the following script to get joints for each image for testing (joints are used for image cropping only.). Make sure you correctly set the location of OpenPose binary.
```
cd utils
python process_openpose.py -i {path_of_images} -o {path_of_images}
```

2. run the following script to run evaluation. Make sure to set `--dataroot` to `path_of_images`, `--results_path` to where you want to dump out results, and `--load_netMR_checkpoint_path` to the checkpoint.
```
python -m apps.submit_eval
```

3. optionally, you can remove artifacts by keeping only the biggest connected component with the following script. (Warning: the script will overwrite the original obj files.)
```
cd utils
python connected_comp.py -f {path_of_objs}
```

For visualizaton, you have two options. One is to render results in the original image space to demonstrate pixel-aligned reconstructions with the following code. The rendered animation (.mp4) will be stored under `{path_of_objs}`.
```
cd utils
python -m render_aligned -f {path_of_objs} -d {path_of_original_images} -ww {image_width} -hh {image_height} 
# add -g for geometry rendering. default is normal visualization 
```
The other option is to render results with turn-table with the following code. The rendered animation (.mp4) will be stored under `{path_of_objs}`.
```
cd utils
python -m render_normalized -f {path_of_objs} -ww {rendering_width} -hh {rendering_height} 
# add -g for geometry rendering. default is normal visualization.
```

## Training 
1. run the following script after commenting out/in relevant training. Make sure you set dataroot properly to `pifu_data`. You can choose whether to use faircluster or local. 
```
python -m apps.submit
```


## Evaluation
1. comment out under Upper PIFu and comment in under Ablation Study in `./apps/submit_eval` and configure the name of models in model_names. Make sure to set `--dataroot` to `eval_dataset` properly. Then run
```
python -m apps.submit_eval
```

2. configure `exp_list` in `./utils/evaluator.py`. Note that this script requires OpenGL, so be careful when running this on server. The statistics will be saved as `buff-item.npy`, `buff-vals.npy`, `rp_item.npy`, and `rp_vals.npy`. The values will be also printed out. The normal rendering will be saved in the result folder. Run the following command.
```
cd utils
python -m evaluator -r {path_of_results} -t {path_of_eval_dataset}
```

## Precompute Occupancy Labels
for occupancy learning, the code offers several ways to obtain a pair of points and in/out labels. 

`--sampling_otf` enables on-the-fly sampling, which is slower but eliminates the need for large storage. This option supports lower PIFu training, but unfortunately this mode is currently not supported for upper PIFu training.

To generate training samples based on gaussian ball perturbation, please follow the instruction below. This data generation is necessary to train upper PIFu.

1. set `--dataroot` to `pifu_data` directory in `./apps/generate_points.py` and run the following command. It will distribute data generation using cluster and should be done with in 1-2 hours.
```
python -m apps.generate_points
```

To generate training samples based on SDF, which cannot be computed on-the-fly, please follow the instruction below. This data generation may be troublesome due to the involvement of multiple programs. As this is used for only lower PIFu to improve the robustness, you could ignore this data generation. In this case, please set `--num_sample_inout` to `0`.
1. install [openvdb](https://www.openvdb.org/download/). This is probably the hardest part especially for Fedora... for ubuntu, apt-get should be sufficient.

2. compile `utils/SDF_generation/main.cpp`. I have included configuration for vscode. Open `utils/SDF_generation` with vscode as a new window and build with CTRL+SHIFT+B. You may get some linker errors including OpenVDB. Make sure you install all the dependency and library path is linked properly.  `main.out` is generated under `utils/SDF_generation`.

3. run the following code to convert meshes into signed distance field after setting mesh data path as well as `pifu_data`. The code generates `{subject_name}_sdf.obj` and `{subject_name}_sdf.data` under `renderpeople/{subject_name}`.
```
cd utils/SDF_generation
python process_rp.py
```

4. move the generated sdf data.
```
mkdir {pifu_data_path}/GEO/SAMPLE/data
mv {renderpeople_path}/*/*sdf.data {pifu_data_path}/GEO/SAMPLE/data/
```

5. to load the sdf data on-the-fly, the following code will split them into multiple files.
```
cd utils/SDF_generation
python convert_numpy.py -i {pifu_data_path}/GEO/SAMPLE/data -o {pifu_data_path}/GEO/SAMPLE 
```
