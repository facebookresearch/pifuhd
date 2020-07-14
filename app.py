import os
from lightweight_human_pose_estimation_pytorch import main
from apps import simple_test
from lib.colab_util import generate_video_from_obj, set_renderer

image_path = '../../sample_images/irene_body.jpg' # example image

image_dir = os.path.dirname(image_path)
file_name = os.path.splitext(os.path.basename(image_path))[0]

# output pathes
obj_path = 'results/pifuhd_final/recon/result_%s_256.obj' % file_name
out_img_path = 'results/pifuhd_final/recon/result_%s_256.png' % file_name
video_path = 'results/pifuhd_final/recon/result_%s_256.mp4' % file_name
video_display_path = 'results/pifuhd_final/result_%s_256_display.mp4' % file_name

main()

simple_test()

renderer = set_renderer()
generate_video_from_obj(obj_path, out_img_path, video_path, renderer)


