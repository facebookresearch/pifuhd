import os
from apps.recon_mr import reconWrapper as reconWrapperMR


#file_dir = '/private/home/hjoo/codes/wildpifu/video_cand'
file_dir = '/private/home/hjoo/codes/wildpifu/fb_cand'
fielNames = os.listdir(file_dir)

# fielNames = ['AdobeStock_156727157_Video_HD_Preview.mov']
for f in fielNames:
    if '.mov' not in f and '.mp4' not in f:
        continue
    videoFileName  = os.path.join(file_dir, f)
    targetFolder =  os.path.join(file_dir, f[:-4] )
    if os.path.exists(targetFolder + "_output"):
       continue

    #mkdir
    cmd = f'mkdir {targetFolder}'
    os.system(cmd)
    
    #ffmpeg
    cmd = f'ffmpeg -i {videoFileName} {targetFolder}/image%08d.jpg'
    #os.system(cmd)
    
    #openpose
    cmd = f'cd /private/home/hjoo/codes/wildpifu/utils; python process_openpose.py -i {targetFolder} -o {targetFolder}'
    #os.system(cmd)

    fileCheck =  os.listdir(targetFolder)

    continue
    #use apps/submit_video.py for processing in parallel

    # jpgNum = [f  for f in  fileCheck if '.jpg' in f]
    # jsonNum = [f  for f in  fileCheck if '.json' in f]

    # if len(jpgNum) != len(jsonNum):
    #     print("Wrong!")
    #     print(targetFolder)
    #     continue

    # resolution = '512'

    # cmd = ['--dataroot', targetFolder, '--results_path', targetFolder+'_output',\
    #         '--loadSize', '1024', '--resolution', resolution, '--load_netMR_checkpoint_path', \
    #         '/private/home/hjoo/codes/wildpifu/checkpoints/ours_final_train_latest']

    # reconWrapperMR(cmd)



