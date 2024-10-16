import numpy as np
import os
import sys
import ntpath
import time
from . import util, html
from subprocess import Popen, PIPE
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from utils.numpy_tools import make_dataset
from mpl_toolkits import mplot3d

import matplotlib as mpl
mpl.use("agg")


if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


def plot_tsne(dir, mode, show=True):
    '''
    plot t-sne presentation
    
    :param dir:  directory of samples
    :param show:  if to show the plots
    
    :return:
    '''

    file_set = make_dataset(dir)
    x = []
    y = []
    for file_name in file_set:
        if file_name.find('lat_eeg') > 0:
            x.append(np.load(file_name).ravel())
            y.append('b')
        elif file_name.find('lat_seeg') > 0:
            x.append(np.load(file_name).ravel())
            y.append('r')

    tsne = TSNE(n_components=3, init='pca', random_state=100, learning_rate=100., n_iter=2000)
    X_tsne = tsne.fit_transform(x)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # normalize

    if mode == '3d':
        X1, Y1, Z1 = [], [], []
        X2, Y2, Z2 = [], [], []
        for i in range(len(X_norm)):
            sample = X_norm[i]
            if y[i] == 'b':
                X1.append(sample[0])
                Y1.append(sample[1])
                Z1.append(sample[2])
            else:
                X2.append(sample[0])
                Y2.append(sample[1])
                Z2.append(sample[2])
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(X1, Y1, Z1, c='r')
        ax.scatter3D(X2, Y2, Z2, c='b')
        #fig.gca().view_init(azim=0, elev=90)
    elif mode == '2d':
        plt.figure(figsize=(8, 8))
        for i in range(X_norm.shape[0]):
            plt.text(X_norm[i, 0], X_norm[i, 1], '*', color=y[i], fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([])
        plt.yticks([])
    if show:
        plt.show()


class Statistics:

    def __init__(self, opt):
        self.opt = opt
        self.dir_path = os.path.join('..', 'experiments', "statistics", opt.name)
        if not os.path.exists(self.dir_path):
            os.makedirs(self.dir_path, exist_ok=True)


    def save_plotted_losses(self, epoch, epoch_counter, counter_ratio, losses):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """

        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())} # dict
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        labels = [k for k in self.plot_data['legend']]
        X = np.array(self.plot_data['X'])
        Y = np.stack(np.array(self.plot_data['Y']), 1)

        for i in range(len(Y)):
            plt.plot(X, Y[i], label=labels[i])
        plt.legend()
        plt.title("GAN Losses")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.savefig(os.path.join(self.dir_path, "GAN Losses"))
        plt.clf()

        save_dir = os.path.join(self.dir_path, 'losses')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for i in range(len(Y)):
            title = labels[i] + " Losses"
            plt.plot(X, Y[i], label=labels[i])
            plt.legend()
            plt.title(title)
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.savefig(os.path.join(save_dir, title))
            plt.clf()

        np.save(os.path.join(save_dir, "loss_dict"), losses)

    
    def save_score_plots(self, freq, score_dict):
        '''score dict is a dictionary stored in the format of (name, score) pairs'''
        score_name = score_dict['names']
        scores = np.array(score_dict['scores']).transpose()
        X = np.array([i * freq for i in range(scores.shape[1])])

        save_dir = os.path.join(self.dir_path, 'scores')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for i in range(len(scores)):
            title = score_name[i] + ' Scores'
            plt.plot(X, scores[i], label=score_name[i])
            plt.legend()
            plt.title(title)
            plt.xlabel('epoch')
            plt.ylabel('score')
            plt.savefig(os.path.join(save_dir, title))
            plt.clf()

        np.save(os.path.join(save_dir, "score_dict"), score_dict)


def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """

    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = util.tensor2im(im_data) # tensor to numpy
        image_name = '%s_%s.png' % (name, label) # save as png
        save_path = os.path.join(image_dir, image_name)
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """

        self.opt = opt  # cache the option
        self.display_id = opt.display_id
        # self.use_html = opt.isTrain and not opt.no_html
        self.use_html = False
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.port = opt.display_port
        self.saved = False
        if self.display_id > 0:  # connect to a visdom server given <display_port> and <display_server>
            import visdom
            self.ncols = opt.display_ncols
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env)
            if not self.vis.check_connection():
                self.create_visdom_connections()
        if self.use_html:  # create an HTML object at <checkpoints_dir>/web/; images will be saved under <checkpoints_dir>/web/images/
            self.web_dir = os.path.join('..', 'experiments', opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            if not os.path.exists(self.web_dir):
                util.mkdirs(self.web_dir)
            if not os.path.exists(self.img_dir):
                util.mkdirs(self.img_dir)
        # create a logging file to store training losses
        self.log_name = os.path.join('..', 'experiments', opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    
    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    
    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    
    def display_current_results(self, visuals, epoch, save_result):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """

        if self.display_id > 0:  # show images in the browser using visdom
            ncols = self.ncols
            if ncols > 0:        # show all the images in one visdom panel
                ncols = min(ncols, len(visuals))
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                        table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)  # create a table css
                # create a table of images.
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                for label, image in visuals.items():
                    image_numpy = util.tensor2im(image)
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                try:
                    self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                    padding=2, opts=dict(title=title + ' images'))
                    label_html = '<table>%s</table>' % label_html
                    self.vis.text(table_css + label_html, win=self.display_id + 2,
                                  opts=dict(title=title + ' labels'))
                except VisdomExceptionBase:
                    self.create_visdom_connections()

            else:     # show each image in a separate visdom panel;
                idx = 1
                try:
                    for label, image in visuals.items():
                        image_numpy = util.tensor2im(image)
                        self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                       win=self.display_id + idx)
                        idx += 1
                except VisdomExceptionBase:
                    self.create_visdom_connections()

        if self.use_html and (save_result or not self.saved):  # save images to an HTML file if they haven't been saved.
            self.saved = True
            # save images to the disk
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)

            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []

                for label, image_numpy in visuals.items():
                    image_numpy = util.tensor2im(image)
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    
    def plot_current_losses(self, epoch, counter_ratio, losses):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """

        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1), # X是n行len(self.plot_data['legend'])列的二维矩阵
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
        except VisdomExceptionBase:
            self.create_visdom_connections()

    
    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data, rank):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        
        message = '(rank: %d, epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (rank, epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
