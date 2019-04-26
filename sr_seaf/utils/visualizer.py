import numpy as np
import os
import ntpath
import time
from . import util
from . import html
from scipy.misc import imresize

import datetime

class Visualizer():
    def __init__(self, opt):
        self.display_id = opt.display_id
        self.use_html = not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.main = opt.main if hasattr(opt,"main") else opt.name
        self.opt = opt
        self.saved = False
        if self.display_id > 0:
            import visdom
            self.vis = visdom.Visdom(port=opt.display_port,env=self.main)

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        self.log_name = os.path.join(opt.checkpoints, opt.name, 'loss_log.txt')
        self.log_name_val = os.path.join(opt.checkpoints, opt.name, 'loss_log_val.txt')
        self.reid_log_name = os.path.join(opt.checkpoints, opt.name, 'reid_log.txt')
        
        util.mkdirs([os.path.dirname(self.log_name) , os.path.dirname(self.reid_log_name)])

        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        self.saved = False

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, save_result,offset=0,title=None):
        if self.display_id > 0:  # show images in the browser
            ncols = self.opt.display_single_pane_ncols
            if ncols > 0:
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
                        table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)
                title = self.name if title is None  else title
                label_html = ''
                label_html_row = ''
                nrows = int(np.ceil(len(visuals.items()) / ncols))
                images = []
                idx = 0
                for label, image_numpy in visuals.items():
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1]))*255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                # pane col = image row
                self.vis.images(images, nrow=ncols, win=self.display_id+offset + 1,
                                padding=2, opts=dict(title=title + ' images'))
                label_html = '<table>%s</table>' % label_html
                self.vis.text(table_css + label_html, win=self.display_id+offset + 2,
                              opts=dict(title=title + ' labels'))
            else:
                idx = 1
                for label, image_numpy in visuals.items():
                    self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                   win=self.display_id+offset + idx)
                    idx += 1

        if self.use_html and (save_result or not self.saved):  # save images to a html file
            self.saved = True
            for label, image_numpy in visuals.items():
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)
            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    # errors: dictionary of error labels and values
    def plot_current_errors(self, epoch, counter_ratio, opt, errors,loss_name=None,display_id_offset=0):
        keys = "_".join( sorted(list(errors.keys())) )
        if not hasattr(self, 'plot_data_list'):
            self.plot_data_list = {}
        if keys not in self.plot_data_list:
            self.plot_data_list .update( { keys :  {'X': [], 'Y': [], 'legend': list(errors.keys())}  } )

        plot_data= self.plot_data_list[ keys ]
        self.plot_data_list[ keys ] ['X'].append(epoch + counter_ratio)
        self.plot_data_list[ keys ] ['Y'].append([errors[k] for k in plot_data['legend']])
        plot_data = self.plot_data_list[ keys ]
        self.vis.line(
            X=np.stack([np.array(plot_data['X'])] * len(plot_data['legend']), 1),
            Y=np.array(plot_data['Y']),
            opts={
                'title':  self.name + ' loss over time' if loss_name is None else loss_name,
                'legend': plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.display_id*10+display_id_offset)

    def plot_current_lrs(self, epoch, counter_ratio, opt, errors,loss_name=None,display_id_offset=0):
        self.plot_current_errors(epoch, counter_ratio, opt, errors,loss_name,display_id_offset)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t,log_name=None):
        cct = datetime.datetime.now()
        cct = str(cct)
        message = '%s: (epoch: %d, iters: %d, time: %.3f) ' % (cct, epoch, i, t)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        if log_name is None :
            log_name = self.log_name
        else:
            log_name = os.path.join(os.path.dirname(self.log_name) , log_name) 
        with open(log_name, "a") as log_file:
            log_file.write('%s\n' % message)



    def print_reid_results(self, message):
        with open(self.reid_log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    # save image to the disk
    def save_images(self, webpage, visuals, image_path, aspect_ratio=1.0):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, im in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            h, w, _ = im.shape
            if aspect_ratio > 1.0:
                im = imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')
            if aspect_ratio < 1.0:
                im = imresize(im, (int(h / aspect_ratio), w), interp='bicubic')
            util.save_image(im, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)



