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

3. Visualize output
- Check "script_render.sh"

