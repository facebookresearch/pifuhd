import os
from tqdm import tqdm 



# # # seqNumPerJob = int(len(seqNames)/job_num +1) 

# # #divide folders to groups to run simultaneously
# # for j in range(job_num):
# #     if (j+1)*seqNumPerJob<len(seqNames):
# #         seqNames_job = seqNames[j*seqNumPerJob:(j+1)*seqNumPerJob]
# #     else:
# #         seqNames_job = seqNames[j*seqNumPerJob:]
# #     # print(len(seqNames_job))

# # seqNumPerJob[::seqNumPerJob]
# def run_openpose(folder_list):
#     for n in tqdm(folder_list):
#         if os.path.exists("/private/home/hjoo/data/speech2gesture/gestures/{}/{}/openpose_result".format(seqFolder,n)):

#             op_num = len(os.listdir("/private/home/hjoo/data/speech2gesture/gestures/{}/{}/openpose_result".format(seqFolder,n)))
#             raw_num = len(os.listdir("/private/home/hjoo/data/speech2gesture/gestures/{}/{}/raw_image".format(seqFolder,n)))
#             if op_num !=raw_num:
#                 cmd = "rm -rf /private/home/hjoo/data/speech2gesture/gestures/{}/{}/openpose_result".format(seqFolder,n)  
#                 print(n)
#                 os.system(cmd)
#             else:
#                 continue

#         if os.path.exists("/private/home/hjoo/data/speech2gesture/gestures/{}/{}/raw_image".format(seqFolder,n)):
#             # cmd = "cd /private/home/hjoo/codes/monoMocap/pipeline/; ./run_mono3Dcapture_devfair_single_openpose.sh test {}".format(n)
#             cmd = "cd /private/home/hjoo/codes/MonocularTotalCapture/; ./run_pipeline_openpose.sh {} {}".format(n,seqFolder)     
#             print(cmd)
#             os.system(cmd)

# # run_openpose(seqNames)

openposedir='/private/home/hjoo/codes/openpose'
rootDir = '/private/home/hjoo/codes/wildpifu/video_cand'
seqnames = os.listdir(rootDir)

# seqnames = ['shun_1.mov']
for seqname in seqnames:
    
    if "mov" not in seqname and "MOV" not in seqname:
        continue

    rawimgdir = seqname[:-4] 
    seqname_fullpath = os.path.join(rootDir, rawimgdir)

    #if ffmpeg and openpose has not been processed
    if not os.path.exists(seqname_fullpath):
        print(f"processing:{seqname}")

        os.mkdir(seqname_fullpath)

        #Extract frames
        cmd = f'cd {rootDir}; ffmpeg -i {seqname} {rawimgdir}/img_%08d.png'
        print(cmd)
        os.system(cmd)

        # #run openpose
        cmd = f'cd {openposedir}; ./build3/examples/openpose/openpose.bin --image_dir {seqname_fullpath} --write_json {seqname_fullpath} --render_pose 0 --display 0 -model_pose BODY_25 --number_people_max 1'
        print(cmd)
        os.system(cmd)