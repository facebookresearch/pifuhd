# Running PIFuHD in devfair

## Running on a video

1. Run the following script 
- Extract a video to frames via ffmpeg
- Run Openpose to get the bbox and 2D keypoint data (this is need to normalize image size)
```
python script/preprocess_videos.py
```

2. Run PIFU
- Check apps/submit_video.py and change the root path.
- Run PIFuHD via local GPUs or devfair clusters
```
cd apps; python submit_video.py
```

3. Clean up mesh
- Remove isolated floating mesh chunks
```
Eg: python -m utils.connected_comp -f {file_path_obj_exists}
# Eg: python -m utils.connected_comp -f ~/data/pifuvideos/IMG_2553_output/mr_fullbody_no_nml_hg_fp0_1112/recon/
```

- You can run this in parallel. Check the following script
```
cd utils; python submit_meshcleaning.py
```

4. Visualize output
- Check "script_render.sh"

