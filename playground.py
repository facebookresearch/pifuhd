from PIFuHD.recontructor import Reconstructor
from PIFuHD.data import EvalWPoseDataset

from PIFuHD.options import BaseOptions

cmd = ['--dataroot', './sample_images',
       '--results_path', './results',
       '--loadSize', '1024',
       '--resolution', '256',
       '--load_netMR_checkpoint_path', './checkpoints/pifuhd.pt',
       '--start_id', '-1',
       '--end_id', '-1']

options_parser = BaseOptions()
opts = options_parser.parse(cmd)

dataset = EvalWPoseDataset(opts)

reconstructor = Reconstructor(opts)
reconstructor.evaluate(dataset)
