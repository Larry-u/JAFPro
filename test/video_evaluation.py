import torch
import torch.nn as nn
from torch.backends import cudnn

import os, sys

sys.path.append('..')

from options import get_general_options
from src.flownet2_pytorch.networks.FlowNetSD import FlowNetSD

from src.crn_model import VGGLoss_CRN
from src.utils import get_vid_list, get_gt_img_list, get_pred_img_list
import argparse
import colored_traceback
from tqdm import tqdm
from skimage.measure import compare_ssim
from skvideo.measure import msssim, psnr
import cv2,numpy as np
import time


def vgg_preprocess(x):
    x = 255.0 * (x + 1.0) / 2.0

    # VGG_MEAN = [103.939, 116.779, 123.68] [B, G, R]
    x[:, 2, :, :] -= 103.939
    x[:, 1, :, :] -= 116.779
    x[:, 0, :, :] -= 123.68

    return x


def flownet_preprocess(img_pair):
    # (-1, 1) to (0, 1)
    return img_pair / 2.0 + 0.5


if __name__ == '__main__':
    start_t = time.time()
    colored_traceback.add_hook()

    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type=str, required=True, help='pred dir')
    parser.add_argument('--gt', type=str, required=True, help='gt dir')
    parser.add_argument('--gpu', type=str, required=True, help='specify gpu devices')
    parser.add_argument('--type', type=str, required=True, help='specify gpu devices')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt = get_general_options()

    pred_name = os.path.normpath(args.pred).split('/')[-1]
    opt['project_dir'] = '/home/Larryu/Projects/JAFPro_minimum'
    log_name = os.path.join(opt['project_dir'], 'log_results_video', '%s.errors.txt' % pred_name)

    # accelerate forwarding
    cudnn.benchmark = True

    # prepare criterions
    l1_criterion = nn.L1Loss().to(device)
    perceptual_criterion = VGGLoss_CRN(weights=[1 / 2.6, 1 / 4.8, 1 / 3.7, 1 / 5.6, 10 / 1.5]).to(device)
    flow_criterion = FlowNetSD(args=[], batchNorm=False).to(device)
    flow_criterion.load_state_dict(torch.load(opt['flownet_path'])['state_dict'])

    pred_dir = args.pred
    gt_dir = args.gt
    data_type=args.type
    
    gt_vid_list = sorted(get_vid_list(gt_dir))
    pred_vid_list=[]
    if data_type=="openpose":
        folder_list=os.listdir(pred_dir)
        for folder in folder_list:
            if folder[-1]=='o':
                continue
            else:
                pred_vid_list.append(os.path.join(pred_dir,folder))
        pred_vid_list.sort()
    if data_type == "densepose":
        pred_vid_list = sorted(get_vid_list(pred_dir))
    if data_type == "every":
        pred_vid_list = sorted(get_vid_list(pred_dir))
    #print("printing the list")
    #print(pred_vid_list)
    print(len(pred_vid_list),len(gt_vid_list))
    assert len(pred_vid_list) == len(gt_vid_list), "number of videos in gt dir and pred dir must equal"
    n_vids = len(gt_vid_list)

    total_vid_ssim = 0.
    total_vid_l1_error = 0.
    total_vid_vgg_error = 0.
    total_vid_flow_error = 0.
    total_vid_psnr= 0.
    total_vid_msssim= 0.

    print("Preparation takes %f seconds..." % (time.time() - start_t))

    n_vids=0
    total_frame=0
    with torch.no_grad():
        for gt_vid_path, pred_vid_path in zip(gt_vid_list, pred_vid_list):
            assert gt_vid_path.split('/')[-1] == pred_vid_path.split('/')[-1], "video name must be identical"

            vid_name = gt_vid_path.split('/')[-1]

            gt_file_list=os.listdir(gt_vid_path)
            pred_file_list=os.listdir(pred_vid_path)
            
            gt_img_list=[]
            pred_img_list=[]
            for file in gt_file_list:
                if file.find("text")<0 and file.find("mask")<0 and file.find("IUV")<0 and file.find("bbox")<0:
                    gt_img_list.append(file)
            for file in pred_file_list:
                if data_type=="openpose":
                    if file.find("png")>0 and file.find("src")<0:
                        pred_img_list.append(file)
                if data_type=="densepose":
                    if file.find("text")<0 and file.find("mask")<0 and file.find("IUV")<0 and file.find("bbox")<0 and file.find("coarse")<0 and file.find("tsf")<0:
                        pred_img_list.append(file)
                if data_type=="every":
                    if file.find("synthesized")>0:
                        pred_img_list.append(file)
            gt_img_list.sort(key=lambda x:int(x[6:-4]))
            if data_type=="openpose":
                pred_img_list.sort(key=lambda x:int(x[11:-4]))
            if data_type=="densepose":
                pred_img_list.sort(key=lambda x:int(x[6:-4]))
            if data_type=="every":
                pred_img_list.sort(key=lambda x:int(x[4:8]))
            #print(gt_img_list,pred_img_list)
            assert len(gt_img_list) == len(pred_img_list), "num of frames must equal"

            #gt_img_list=gt_img_list[10:30]
            #pred_img_list=pred_img_list[10:30]

            prev_pred_torch = None
            prev_gt_torch = None

            vid_ssim = 0.
            vid_l1_error = 0.
            vid_vgg_error = 0.
            vid_flow_error = 0.
            vid_msssim=0.
            vid_psnr=0.

            num_frames = len(gt_img_list)
            #print(gt_img_list,pred_img_list)
            pred_gray_video = []
            gt_gray_video = []
            total_frame=total_frame+num_frames
            for idx in tqdm(range(num_frames)):
                '''
                if data_type=="densepose":
                    assert pred_img_list[idx] == gt_img_list[idx]
                '''
                if data_type=="every":
                    pred_img = cv2.resize(cv2.imread(os.path.join(pred_vid_path,pred_img_list[idx])),(256,256),interpolation=cv2.INTER_NEAREST)
                else:
                    pred_img = cv2.imread(os.path.join(pred_vid_path,pred_img_list[idx]))
                #cv2.imwrite('/home/haolin/test.jpg',pred_img)
                #cv2.waitKey(0)
                gt_img = cv2.imread(os.path.join(gt_vid_path,gt_img_list[idx]))


                pred_img_gray = cv2.cvtColor(pred_img, cv2.COLOR_BGR2GRAY)
                gt_img_gray = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
                
                pred_gray_video.append(pred_img_gray)
                gt_gray_video.append(gt_img_gray)

                # convert BGR to RGB and normalize to (-1, 1)
                pred_img = (cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB) / 255. - 0.5) * 2
                gt_img = (cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB) / 255. - 0.5) * 2

                pred_torch = torch.from_numpy(pred_img).float().unsqueeze(0).permute(0, 3, 1, 2).to(device)
                gt_torch = torch.from_numpy(gt_img).float().unsqueeze(0).permute(0, 3, 1, 2).to(device)

                # calculate ssim score
                (score, diff) = compare_ssim(pred_img_gray, gt_img_gray, full=True)
                vid_ssim += score

                # calcuate l1 error
                vid_l1_error += l1_criterion(pred_torch, gt_torch).item()

                # calculate vgg error
                # *_torch is in RGB order
                vid_vgg_error += perceptual_criterion(vgg_preprocess(pred_torch), vgg_preprocess(gt_torch)).item()

                # calculate flow error
                if idx == 0:
                    prev_pred_torch = pred_torch
                    prev_gt_torch = gt_torch
                else:
                    pred_flow = flow_criterion(flownet_preprocess(torch.cat([prev_pred_torch, pred_torch], dim=1)))[0]
                    gt_flow = flow_criterion(flownet_preprocess(torch.cat([prev_gt_torch, gt_torch], dim=1)))[0]

                    vid_flow_error += l1_criterion(pred_flow, gt_flow).item()

                    prev_pred_torch = pred_torch
                    prev_gt_torch = gt_torch
            pred_gray_video = np.array(pred_gray_video)
            gt_gray_video = np.array(gt_gray_video)
            vid_msssim = msssim(gt_gray_video, pred_gray_video).sum()
            #if vid_msssim>0.92*30:
                #continue
            n_vids=n_vids+1
            vid_psnr = psnr(gt_gray_video, pred_gray_video).sum()
            msg = "For vid {}, Mean SSIM: {:.4f}, Mean MS-SSIM{:.4f},PSNR:{:.4f}, Mean L1: {:.4f}, Mean VGG: {:.4f}, Mean Flow: {:.4f}".format(
                vid_name,
                vid_ssim / num_frames,
                vid_msssim /num_frames,
                vid_psnr / num_frames,
                vid_l1_error / num_frames,
                vid_vgg_error / num_frames,
                vid_flow_error / num_frames
            )
            print(msg)

            with open(log_name, "a") as log_file:
                log_file.write('%s\n' % msg)  # save the message


            total_vid_ssim += vid_ssim
            total_vid_l1_error += vid_l1_error
            total_vid_vgg_error += vid_vgg_error
            total_vid_flow_error += vid_flow_error
            total_vid_msssim += vid_msssim
            total_vid_psnr += vid_psnr

        print("Evaluation done.")
        msg = "For whole dataset, Mean SSIM: {:.4f},MS-SSIM: {:.4f}, PSNR: {:.4f}, Mean L1: {:.4f}, Mean VGG: {:.4f}, Mean Flow: {:.4f}".format(
            total_vid_ssim / total_frame,
            total_vid_msssim / total_frame,
            total_vid_psnr / total_frame,
            total_vid_l1_error / total_frame,
            total_vid_vgg_error / total_frame,
            total_vid_flow_error / total_frame
        )
        print(msg)
        with open(log_name, "a") as log_file:
            log_file.write('%s\n' % msg)  # save the message
