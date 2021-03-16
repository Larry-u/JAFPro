from __future__ import absolute_import, division, print_function
from visdom import Visdom
import numpy as np


class BaseVisualizer(object):
    """
        Base class of visualizer.
    """

    def __init__(self, env, ip, port):
        print('ip = {}, port = {}'.format(ip, port))
        self.env = env

        if ip and port:
            self.vis = Visdom(server=ip,
                              endpoint='events',
                              port=port,
                              env=self.env)
            # self.vis.close(env=self.env)


class VisdomVisualizer(BaseVisualizer):

    def __init__(self, env, time_step=1, num_points=18,
                 ip=None, port=None):
        super(VisdomVisualizer, self).__init__(env, ip=ip, port=port)

        self.time_step = time_step
        self.num_points = num_points

    def vis_keypoints(self, preds, gts):
        """
        :type preds:  torch.tensor, (self.time_step, num_points, 2)
        :param preds: the time series of predicted keypoints.

        :type gts:  torch.tensor, (self.time_step, num_points, 2)
        :param gts: the time series of ground truth keypoints.
        """

        lsp_key_points_name = ['Right ankle', 'Right knee', 'Right hip', 'Left hip', 'Left knee', 'Left ankle', 'Right wrist', 'Right elbow', 'Right shoulder',
                               'Left shoulder', 'Left elbow', 'Left wrist', 'Neck', 'Head top']
        lsp_plus_key_points_name = lsp_key_points_name + ['Left ear', 'Left eye', 'Nose', 'Right ear', 'Right eye']

        preds = preds.clone()
        preds[:, :, 1] = - preds[:, :, 1]
        gts = gts.clone()
        gts[:, :, 1] = - gts[:, :, 1]

        for i in range(self.time_step):
            win = 'pred_keypoints_' + str(i)
            self.draw_skeleton(preds[i], win, plus=True)

        for i in range(self.time_step):
            win = 'gt_keypoints_' + str(i)
            self.draw_skeleton(gts[i], win, plus=False)

    def draw_skeleton(self, key_points, win_name, plus=False):
        """
        :param key_points: coco format [14 or 19, 2]
        :return:
        """
        lsp_key_points_name = ['Right ankle', 'Right knee', 'Right hip', 'Left hip', 'Left knee', 'Left ankle', 'Right wrist', 'Right elbow', 'Right shoulder',
                               'Left shoulder', 'Left elbow', 'Left wrist', 'Neck', 'Head top']

        lsp_plus_key_points_name = lsp_key_points_name + ['Nose', 'Left eye', 'Right eye', 'Left ear', 'Right ear']

        # start from 1
        lsp_kintree_table = [(14, 13), (13, 10), (10, 11), (11, 12), (13, 9), (9, 8), (8, 7), (13, 4), (13, 3), (4, 5), (5, 6), (3, 2), (2, 1)]
        lsp_plus_kintree_table = lsp_kintree_table + [(18, 16), (16, 15), (15, 17), (17, 19)]

        # minus 1 to start from 0
        lsp_kintree_table = [(k0 - 1, k1 - 1) for k0, k1 in lsp_kintree_table]
        lsp_plus_kintree_table = [(k0 - 1, k1 - 1) for k0, k1 in lsp_plus_kintree_table]

        if plus:
            key_points_name = lsp_plus_key_points_name
            kintree_table = lsp_plus_kintree_table
        else:
            key_points_name = lsp_key_points_name
            kintree_table = lsp_kintree_table

        X = np.array([[key_points[k0][0], key_points[k1][0]] for k0, k1 in kintree_table]).T
        Y = np.array([[key_points[k0][1], key_points[k1][1]] for k0, k1 in kintree_table]).T

        self.vis.line(Y, X, win=win_name, opts=dict(xtickmin=-1, xtickmax=1, xtickstep=0.2,
                                                    ytickmin=-1, ytickmax=1, ytickstep=0.2,
                                                    markers=True, title=win_name))

    def vis_named_img(self, name, imgs, denormalize=True, transpose=False):
        """
        :param name: str, window name
        :param imgs: np.ndarray or torch.tensor, (self.time_step, 1, self.image_size, self.image_size)
        :param denormalize: True, [-1, 1] -> [0, 1]
        :param transpose: False
        :return:
        """
        if isinstance(imgs, np.ndarray):
            if imgs.ndim == 3:
                imgs = imgs[:, np.newaxis, :, :]

            if transpose:
                imgs = np.transpose(imgs, (0, 3, 1, 2))

        else:
            if imgs.ndimension() == 3:
                imgs = imgs[:, None, :, :]

            if transpose:
                imgs = imgs.permute(0, 3, 1, 2)

        if denormalize:
            imgs = (imgs + 1) / 2.0

        self.vis.images(
            tensor=imgs,
            win=name,
            opts={'title': name}
        )

    def vis_preds_gts(self, preds=None, gts=None):
        """
        :type preds: np.ndarray, (self.time_step, 1, self.image_size, self.image_size) or
                        (self.time_step, self.image_size, self.image_size)
        :param preds: the time series of predicted silhouettes,
                      if preds.ndim == 3, (self.time_step, self.image_size, self.image_size)
                      then, it will reshape to (self.time_step, 1, self.image_size, self.image_size)

        :type gts: np.ndarray, (self.time_step, 1, self.image_size, self.image_size) or
                        (self.time_step, self.image_size, self.image_size)
        :param gts: the time series of ground truth silhouettes,
                      if preds.ndim == 3, (self.time_step, self.image_size, self.image_size)
                      then, it will reshape to (self.time_step, 1, self.image_size, self.image_size)
        """
        if preds is not None:
            if type(preds) == np.ndarray:
                if preds.ndim == 3:
                    preds = preds[:, np.newaxis, :, :]
            else:
                if preds.ndimension() == 3:
                    preds = preds[:, None, :, :]

            preds = (preds + 1.0) / 2.0
            self.vis.images(
                tensor=preds,
                win='predicted images',
                opts={'title': 'predicted images'}
            )

        if gts is not None:
            if type(gts) == np.ndarray:
                if gts.ndim == 3:
                    gts = gts[:, np.newaxis, :, :]
            else:
                if gts.ndimension() == 3:
                    gts = gts[:, None, :, :]

            gts = (gts + 1.0) / 2.0
            self.vis.images(
                tensor=gts,
                win='ground truth images',
                opts={'title': 'ground truth images'}
            )
