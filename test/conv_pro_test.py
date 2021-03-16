import argparse

import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.utils import data
import numpy as np

import time
import os, sys

sys.path.append("..")
import datetime
import cv2

from src.networks import UNet_inpainter, Accumulate_LSTM_no_loss
from src.data import Fusion_dataset_smpl_test
from options import get_general_options
from src.utils import Logger, TransferTexture
from src.crn_model import CRN, CRN_small, CRN_smaller
from src.flow_net import Propagation3DFlowNet
from src.cal_flow import float_estimate


def xavier_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight)
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def flip(x, dim):
    dim = x.dim() + dim if dim < 0 else dim
    return x[tuple(slice(None, None) if i != dim
                   else torch.arange(x.size(i) - 1, -1, -1).long()
                   for i in range(x.dim()))]


def texture_warp_pytorch(tex_parts, IUV, device):
    IUV = torch.from_numpy(IUV).to(device).cuda()
    U = IUV[:, :, 1]
    V = IUV[:, :, 2]
    #
    #     R_im = torch.zeros(U.size())
    #     G_im = torch.zeros(U.size())
    #     B_im = torch.zeros(U.size())
    generated_image = torch.zeros(IUV.size(), device=device).unsqueeze(0).permute(0, 3, 1, 2).cuda()
    ###
    for PartInd in range(1, 25):  ## Set to xrange(1,23) to ignore the face part.
        #         tex = TextureIm[PartInd-1,:,:,:].squeeze() # get texture for each part.
        tex = tex_parts[PartInd - 1]  # get texture for each part.
        #####
        #         R = tex[:,:,0]
        #         G = tex[:,:,1]
        #         B = tex[:,:,2]
        ###############
        #         x,y = torch.where(IUV[:,:,0]==PartInd, )
        #         u_current_points = U[x,y]   #  Pixels that belong to this specific part.
        #         v_current_points = V[x,y]
        u_current_points = torch.where(IUV[:, :, 0] == PartInd, U.float().cuda(),
                                       torch.zeros(U.size()).cuda())  # Pixels that belong to this specific part.
        v_current_points = torch.where(IUV[:, :, 0] == PartInd, V.float().cuda(), torch.zeros(V.size()).cuda())

        x = ((255 - v_current_points) / 255. - 0.5) * 2  # normalize to -1, 1
        y = (u_current_points / 255. - 0.5) * 2
        grid = torch.cat([x.unsqueeze(2), y.unsqueeze(2)], dim=2).unsqueeze(0).to(device).cuda()  # 1, H, W, 2
        tex_image = tex.unsqueeze(0).float().to(device).cuda()  # 1, 3, H, W

        sampled_patch = torch.nn.functional.grid_sample(tex_image, grid, mode='bilinear').cuda()
        generated_image = torch.where(IUV[:, :, 0] == PartInd, sampled_patch.cuda(), generated_image.cuda())

    return generated_image.squeeze()


def train(args):
    model_name = args.exp_name
    num_frames = args.num_frame
    start = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set this to prevent matplotlib import error
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'

    opt = get_general_options()
    opt['batch_size'] = 1

    ckpt_dir = os.path.join(opt['model_save_dir'], model_name)
    opt['network_dir'] = ckpt_dir
    opt["num_target"] = 1
    if num_frames > 3:
        opt["maximum_ref_frames"] = num_frames
    else:
        opt["maximum_ref_frames"] = num_frames
    result_dir = os.path.join(opt["test_save_dir"], model_name)

    # accelerate forwarding
    cudnn.benchmark = True

    num_workers = 2
    # train_data = PatchTransferDataset(opt, mode='train')
    #
    # test_data = PatchTransferDataset(opt, mode='test')
    # test_data_loader = data.DataLoader(dataset=test_data, batch_size=1, num_workers=1,
    #                                    pin_memory=True).__iter__()

    # Model
    Accu_model = Accumulate_LSTM_no_loss()
    Accu_model_dir = os.path.join(opt['model_save_dir'], model_name)
    Accu_model_weight_dir = os.path.join(Accu_model_dir, "Accu_iter_42000.pth")
    Accu_model.load_state_dict(torch.load(Accu_model_weight_dir))
    Accu_model = nn.DataParallel(Accu_model).to(device)

    inpaint_model = UNet_inpainter()  # input to this is 7*256*256, both for input and mask
    inpaint_model_dir = os.path.join(opt['model_save_dir'], model_name)
    inpaint_model_weight_dir = os.path.join(inpaint_model_dir, "inpaint_iter_42000.pth")
    inpaint_model.load_state_dict(torch.load(inpaint_model_weight_dir))
    inpaint_model = nn.DataParallel(inpaint_model).to(device)
    print("load smaller model")
    bg_model = CRN_smaller(3)
    bg_model_dir = os.path.join(opt['model_save_dir'], model_name)
    bg_model_weight_dir = os.path.join(bg_model_dir, "bg_iter_42000.pth")
    bg_model.load_state_dict(torch.load(bg_model_weight_dir))
    bg_model = nn.DataParallel(bg_model).to(device)

    refine_model = CRN_smaller(3, fg=True)
    refine_model_dir = os.path.join(opt['model_save_dir'], model_name)
    refine_model_weight_dir = os.path.join(refine_model_dir, "refine_iter_42000.pth")
    refine_model.load_state_dict(torch.load(refine_model_weight_dir))
    refine_model = nn.DataParallel(refine_model).to(device)

    propagater = Propagation3DFlowNet(9, 32, 2, 3, use_deconv=False)
    propagater_dir = os.path.join(opt['model_save_dir'], model_name)
    propagater_weight_dir = os.path.join(propagater_dir, "pro_iter_42000.pth")
    propagater.load_state_dict(torch.load(propagater_weight_dir))
    propagater = nn.DataParallel(propagater).to(device)

    flow_calculator = float_estimate()
    flow_calculator = nn.DataParallel(flow_calculator).to(device)

    print("preparation cost %f seconds" % (time.time() - start))

    start_t = time.time()
    Accu_model.eval()
    inpaint_model.eval()
    bg_model.eval()
    refine_model.eval()

    n_epoch = 200

    # import pdb; pdb.set_trace()
    data_start = time.time()
    count = 0
    train_data = Fusion_dataset_smpl_test(opt, mode='test')
    train_data_loader = data.DataLoader(dataset=train_data, batch_size=opt['batch_size'], shuffle=False,
                                        num_workers=num_workers, pin_memory=True)
    with torch.no_grad():
        for batch_id, (src_data, tgt_data, data_255, smpl_data, vid_name, img_name_list, chosen_frame) in enumerate(
                train_data_loader):
            count = count + 1
            data_t = time.time() - data_start
            vid_name = vid_name[0]

            # need src_texture_im, src_IUV, tgt_IUV, real, tgt_IUV255

            src_img, src_IUV, src_texture_im, src_mask_im, src_common_area, src_mask_in_image = src_data
            src_common_area = src_common_area.float().to(device)
            # src_common_area=src_common_area.unsqueeze(1).repeat(1,3,1,1)
            src_mask_in_image = src_mask_in_image.permute(0, 1, 4, 2, 3).float().to(device)
            src_img = src_img.permute(0, 1, 4, 2, 3).float().to(device)
            src_IUV = src_IUV.permute(0, 1, 4, 2, 3).float().to(device)
            src_texture_im = src_texture_im.permute(0, 1, 4, 2, 3).float().to(device)
            src_mask_im = src_mask_im.float().to(device)

            tgt_img, tgt_IUV = tgt_data
            tgt_img = tgt_img.permute(0, 1, 4, 2, 3).float().to(device)
            tgt_IUV = tgt_IUV.permute(0, 1, 4, 2, 3).float().to(device)
            # tgt_mask_in_image=tgt_mask_in_image.permute(0,1,4,2,3).float().to(device)
            # tgt_texture_im=tgt_texture_im.permute(0,1,4,2,3).float().to(device)
            # tgt_mask_im=tgt_mask_im.float().to(device)
            bg_mask = (1 - src_mask_in_image[:, 0].squeeze(1))
            bg_incomplete = bg_mask * src_img[:, 0].squeeze(1) + (1 - bg_mask) * torch.randn(bg_mask.shape).cuda()

            src_IUV255 = data_255[0][:, 0].squeeze()
            tgt_IUV255 = data_255[1].squeeze(1)
            src_IUV255 = src_IUV255.numpy()
            tgt_IUV255 = tgt_IUV255.numpy()

            smpl_seq, smpl_real_mask, smpl_vertices = smpl_data
            smpl_real_mask = smpl_real_mask.permute(0, 1, 4, 2, 3).float().to(device)
            # prev_real_img=prev_real_img.permute(0,3,1,2).float().to(device)
            smpl_vertices = smpl_vertices.float().to(device)
            smpl_seq = torch.tensor(smpl_seq).float().cuda()

            # print("texture_map's shape is:",src_texture_im.shape)
            # print("tgt_IUV255's shape is",tgt_IUV255.shape)
            if num_frames == 1:
                random_index = np.array([0])
            if num_frames == 2:
                random_index = np.array([0, 1])
            if num_frames == 3:
                random_index = np.array([0, 1, 2])
            if num_frames == 4:
                random_index = np.array([0, 1, 2, 3])
            if num_frames == 5:
                random_index = np.array([0, 1, 2, 3, 4])
            src_texture_im_input = []
            for i in range(4):
                for j in range(6):
                    concat_image = []
                    for z in range(random_index.shape[0]):
                        concat_image.append(
                            src_texture_im[:, random_index[z], :, i * 200:(i + 1) * 200, j * 200:(j + 1) * 200].squeeze(
                                1))
                    src_texture_im_input.append(concat_image)

            Accu_output_texture = Accu_model(src_texture_im_input)

            random_number_list = []
            for i in range(random_index.shape[0]):
                random_number_list.append(random_index[i])
            if num_frames < opt["maximum_ref_frames"]:
                for i in range(opt["maximum_ref_frames"]):
                    if i not in random_number_list:
                        src_mask_im[:, i] = src_mask_im[:, i] * 0

            src_common_area = (src_common_area * 0).byte()
            for i in range(opt["maximum_ref_frames"]):
                src_common_area = src_common_area | src_mask_im[:, i].byte()

            src_common_area = src_common_area.float()
            src_common_area = src_common_area.unsqueeze(1).repeat(1, 3, 1, 1)

            for i in range(4):
                for j in range(6):
                    common_area = src_common_area[:, :, i * 200:(i + 1) * 200, j * 200:(j + 1) * 200]
                    Accu_output_texture[i * 6 + j] = Accu_output_texture[i * 6 + j] * common_area

            inpaint_texture = inpaint_model(Accu_output_texture)

            save_video_dir = os.path.join(result_dir, vid_name)
            if os.path.exists(save_video_dir) == False:
                os.makedirs(save_video_dir)

            bg_output = bg_model(bg_incomplete, 256)

            first_frame = tgt_img[:, 0]
            # print(first_frame.shape)
            # vis_image=((first_frame.squeeze(0).permute(1,2,0).detach().cpu().numpy()/2+0.5)*255).astype(np.uint8)
            # img_save_dir=os.path.join(save_video_dir,img_name_list[0][0])
            # cv2.imwrite(img_save_dir,vis_image)
            # prev_image=src_img[:,chosen_frame.shape[1]-1]
            for i in range(tgt_IUV255.shape[1]):
                distance = np.abs(i - chosen_frame)
                src_pro = np.argmin(distance)
                prev_image = src_img[:, src_pro]
                inpaint_warp = torch.full((1, 3, 256, 256), 0, device=device)
                inpaint_texture_list = list(map(lambda inp: inp[0], inpaint_texture))

                inpaint_warp[0] = texture_warp_pytorch(inpaint_texture_list, tgt_IUV255[0, i], device)

                refine_output, fg_mask = refine_model(inpaint_warp[:], 256)

                fusion_output = refine_output * fg_mask.repeat(1, 3, 1, 1) + bg_output * (
                        1 - fg_mask.repeat(1, 3, 1, 1))

                pro_index = np.clip(chosen_frame[0, src_pro], 0, 30)
                prev_smpl = [smpl_seq[:, pro_index, 0:3], smpl_seq[:, pro_index, 3:75], smpl_vertices[:, pro_index],
                             smpl_seq[:, pro_index, 75:85]]
                tgt_smpl = [smpl_seq[:, i, 0:3], smpl_seq[:, i, 3:75], smpl_vertices[:, i], smpl_seq[:, i, 75:85]]

                tsf_image = flow_calculator(prev_image, prev_smpl, tgt_smpl)
                flow_pro_input = {'fake_tgt': fusion_output, 'tsf_image': tsf_image, 'use_mask': True,
                                  'tgt_smpl_mask': smpl_real_mask[:, i], 'use_IUV': True, 'tgt_IUV': tgt_IUV[:, i]}
                pro_output = propagater(flow_pro_input)
                final_output = pro_output['pred_target']
                mask = pro_output["weight"]
                # prev_image=final_output

                coarse_image = np.clip(
                    (fusion_output[0].squeeze(0).permute(1, 2, 0).detach().cpu().numpy() / 2 + 0.5) * 255, 0,
                    255).astype(np.uint8)
                vis_image = np.clip(
                    (final_output[0].squeeze(0).permute(1, 2, 0).detach().cpu().numpy() / 2 + 0.5) * 255, 0,
                    255).astype(np.uint8)
                mask_image = np.clip(mask[0].squeeze(0).cpu().numpy() * 255, 0, 255).astype(np.uint8)
                vis_tsf_image = np.clip(
                    (tsf_image[0].squeeze(0).permute(1, 2, 0).detach().cpu().numpy() / 2 + 0.5) * 255, 0, 255).astype(
                    np.uint8)
                # print(mask_image.shape)
                # bg_image=np.clip((bg_incomplete[0].squeeze(0).permute(1,2,0).detach().cpu().numpy()/2+0.5)*255,0,255).astype(np.uint8)
                coarse_image_dir = os.path.join(save_video_dir, "coarse_" + img_name_list[i][0])
                img_save_dir = os.path.join(save_video_dir, img_name_list[i][0])
                mask_save_dir = os.path.join(save_video_dir, "mask_" + img_name_list[i][0])
                tsf_save_dir = os.path.join(save_video_dir, "tsf_" + img_name_list[i][0])
                # bg_save_dir=os.path.join(save_video_dir,"bg_%d.jpg"%(i))
                cv2.imwrite(mask_save_dir, mask_image)
                cv2.imwrite(img_save_dir, vis_image)
                cv2.imwrite(coarse_image_dir, coarse_image)
                cv2.imwrite(tsf_save_dir, vis_tsf_image)
                # cv2.imwrite(bg_save_dir,bg_image)
                print("writing to ", img_save_dir)

            # print(inpaint_output.shape)
            data_start = time.time()

    print("Testing Done.")


if __name__ == '__main__':
    import colored_traceback

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', '-e', type=str, required=True, help='experiment name')
    parser.add_argument('--num_frame', '-n', type=int, required=True, help='number of input reference frame')
    parser.add_argument('--gpu', type=str, required=True, help='specify gpu devices')
    parser.add_argument('--debug', action='store_true', help='specify debug mode')
    parser.add_argument('--init', type=str, help='weight init method')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    colored_traceback.add_hook()
    train(args)
