from zmq.eventloop import ioloop 
ioloop.install() # Needs to happen before any tornado imports!
import numpy as np
import os
import visdom

class Visualizer():
    def __init__(self, opt):
        self.display_id = opt.display_id
        self.name = opt.name
        self.opt = opt

        try:
            self.vis = visdom.Visdom(server='http://localhost', port=opt.display_port, use_incoming_socket=False)
        except:
            self.vis = visdom.Visdom(server='http://localhost', port=opt.display_port)

        assert self.display_id > 0, 'please set display_id > 0'

    def plot_current_losses(self, epoch, counter_ratio, losses):
        ### args
        # losses: dictionary type (ex: losses['mse'] = 1.0)
        # epoch: int type (ex: epoch = 2)
        # counter_ratio: float type (ex: counter_ratio = 0.12)

        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}

        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])

        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.display_id)

    def plot_current_test_losses(self, epoch, counter_ratio, losses):
        ### args
        # losses: dictionary type (ex: losses['mse'] = 1.0)
        # epoch: int type (ex: epoch = 2)
        # counter_ratio: float type (ex: counter_ratio = 0.12)

        if not hasattr(self, 'plot_test_data'):
            self.plot_test_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}

        self.plot_test_data['X'].append(epoch + counter_ratio)
        self.plot_test_data['Y'].append([losses[k] for k in self.plot_test_data['legend']])

        self.vis.line(
            X=np.stack([np.array(self.plot_test_data['X'])] * len(self.plot_test_data['legend']), 1),
            Y=np.array(self.plot_test_data['Y']),
            opts={
                'title': self.name + ' test loss over time',
                'legend': self.plot_test_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.display_id + 1)

    def display_current_results(self, epoch, visuals):
        ### args
        # epoch: (int)
        # visuals: (dict)

        for idx, key in enumerate(visuals.keys()):
            img = visuals[key][0]

            self.vis.image(
                img.cpu().numpy() * 0.5 + 0.5,
                opts=dict(title=key),
                win=self.display_id + 2 + idx
            )

