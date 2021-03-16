from src.liquid_networks import HumanModelRecovery
from src.nmr import SMPLRenderer
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import utils.cv_utils as cv_utils
import pickle

class float_estimate(nn.Module):
    def __init__(self,smpl_pkl='../smpl_model.pkl',hmr_model_path='../hmr_tf2pt.pth'):
        super(float_estimate, self).__init__()
        self.render=SMPLRenderer(image_size=256,tex_size=3,has_front=True, fill_back=False)
        self.hmr = HumanModelRecovery(smpl_pkl_path=smpl_pkl)
        saved_data = torch.load(hmr_model_path)
        self.hmr.load_state_dict(saved_data)
    
    def forward(self,src_img,src_smpl,tgt_smpl):
        src_cam,src_pose,src_vertices,src_shape=src_smpl
        tgt_cam,tgt_pose,tgt_vertices,tgt_shape=tgt_smpl
        flow=self.cal_flow(src_cam,src_pose,src_vertices,src_shape,tgt_cam,tgt_pose,tgt_vertices,tgt_shape)
        tsf_image=self.warp_image(src_img,flow)
        return tsf_image
    
    def cal_flow(self,src_cam,src_pose,src_vertices,src_shape,tgt_cam,tgt_pose,tgt_vertices,tgt_shape): #src_pose should be dictionary, keys are pose, shape, cams
        src_f2verts, src_fim, src_wim = self.render.render_fim_wim(src_cam, src_vertices)
        src_f2verts=src_f2verts[:, :, :, 0:2]
        src_f2verts[:, :, :, 1] *= -1
        tgt_smpl=torch.cat([tgt_cam,tgt_pose,tgt_shape],dim=1)
        tsf_f2verts, tsf_fim, tsf_wim = self.render.render_fim_wim(tgt_cam, tgt_vertices)
        flow = self.render.cal_bc_transform(src_f2verts, tsf_fim, tsf_wim)
        return flow
    
    def warp_image(self,src_image,flow):
        tsf_img = F.grid_sample(src_image, flow,padding_mode='border')
        return tsf_img
        
    def swap_smpl(self,src_cam, src_shape, tgt_smpl, first_cam,cam_strategy='smooth'):
        tgt_cam = tgt_smpl[:, 0:3].contiguous()
        pose = tgt_smpl[:, 3:75].contiguous()

        # TODO, need more tricky ways
        if cam_strategy == 'smooth':

            cam = src_cam.clone()
            delta_xy = tgt_cam[:, 1:] - first_cam[:, 1:]
            cam[:, 1:] += delta_xy

        elif cam_strategy == 'source':
            cam = src_cam
        else:
            cam = tgt_cam

        tsf_smpl = torch.cat([cam, pose, src_shape], dim=1)

        return tsf_smpl