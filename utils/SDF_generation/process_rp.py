import os

subs = [f for f in os.listdir('../../pifu_data/RENDER') if 'rp_' in f]

subs = sorted(subs)
root = '/home/shunsukesaito/data/renderpeople'
for i, sub in enumerate(subs):
    cmd = './main.out %s %s' % (root, sub)
    os.system(cmd)