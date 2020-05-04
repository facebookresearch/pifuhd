# Multi-Level Pixel-Aligned Implicit Function for High-Resolution 3D Human Digitization

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

## Training 
1. run the following script after commenting out/in relevant training. Make sure you set dataroot properly to `pifu_data`. You can choose whether to use faircluster or local. 
```
python -m apps.submit
```

## Testing
1. run the following script to get joints for each image for testing (joints are used for image cropping only.).
```
cd utils
python process_openpose.py -f {path_of_images} -d {path_of_images}
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

## Visualization
For visualizaton, you have two options. One is to render results in the original image space to demonstrate pixel-aligned reconstructions with the following code. The rendered animation (.mp4) will be stored under `{path_of_objs}`.
```
python -m utils.render_aligned -f {path_of_objs} -i {path_of_original_images} -ww {image_width} -hh {image_height} 
#example:  python -m utils.render_aligned -f ~/data/pifuvideos/IMG_2550_output/mr_fullbody_no_nml_hg_fp0_1112/recon/ -i ~/data/pifuvideos/IMG_2550 -ww 1280 -hh 1920 --png -g
```

There are several options:
- "-g": to visualize geometry rendering
- "--png": if your input images are png format. Default if jpg
- "-v": generat concat video by concatenating vertically



The other option is to render results with turn-table with the following code. The rendered animation (.mp4) will be stored under `{path_of_objs}`.
```
cd utils
python -m render_normalized -f {path_of_objs} -ww {rendering_width} -hh {rendering_height} 
# add -g for geometry rendering. default is normal visualization.
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

## Running in FAIR Cluster
See [README_devfair.MD](README_devfair.MD)