import argparse

import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.optim import Adam, lr_scheduler, SGD
from torch.utils import data

import time
import os, sys

sys.path.append("..")
import datetime
import cv2

from src.networks import Accumulate_LSTM_no_loss, UNet_inpainter
from src.data import Fusion_dataset_textonly
from options import get_general_options
from src.utils import Logger

import numpy as np


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


def train(args):
    model_name = args.exp_name

    start = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
    # cv2.setNumThreads(0)

    # set this to prevent matplotlib import error
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'

    opt = get_general_options()
    opt['model_save_interval'] = 5000 if not args.debug else 3
    opt['vis_interval'] = 200 if not args.debug else 3
    opt['n_training_iter'] = 100000

    ckpt_dir = os.path.join(opt['model_save_dir'], model_name)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    opt['network_dir'] = ckpt_dir
    opt["num_target"] = 2
    opt["maximum_ref_frames"] = 4
    opt['batch_size'] = 2
    opt["face_GAN"] = False
    opt["output_mask"] = True

    tb_logger = Logger(ckpt_dir)

    # accelerate forwarding
    cudnn.benchmark = True

    #
    # test_data = PatchTransferDataset(opt, mode='test')
    # test_data_loader = data.DataLoader(dataset=test_data, batch_size=1, num_workers=1,
    #                                    pin_memory=True).__iter__()

    # Model
    Accu_model = Accumulate_LSTM_no_loss()  # varible with maximum 3 reference texture
    Accu_model_dir = os.path.join(opt['model_save_dir'], 'texture_accumulation_210105')
    Accu_model_weight_dir = os.path.join(Accu_model_dir, "iter_5000.pth")
    Accu_model.load_state_dict(torch.load(Accu_model_weight_dir))
    Accu_model = nn.DataParallel(Accu_model).to(device)
    # input is [incomplete_img, IUV,src_0,IUV_src_0,bg_inpaint_mask,fg_inpaint_mask]
    # D_model=nn.DataParallel(ImageDiscriminator(ndf=32,input_channel=6)).to(device) # discriminator, input is [real/fake???iuv]

    inpaint_model = UNet_inpainter()
    # inpaint_model_dir=os.path.join(opt['model_save_dir'],'text_accu_inpaint_0923')
    # inpaint_model_weight_dir=os.path.join(inpaint_model_dir,"inpaint_iter_42000.pth")
    # inpaint_model.load_state_dict(torch.load(inpaint_model_weight_dir))
    inpaint_model = nn.DataParallel(inpaint_model).to(device)

    l1_criterion = nn.L1Loss().to(device)

    if args.init == 'xavier':
        Accu_model.apply(xavier_init)

    # optimizer
    optimizer_accu = Adam(Accu_model.parameters(), lr=1e-4)
    optimizer_inpaint = Adam(inpaint_model.parameters(), lr=1e-4)
    # optimizer_D=Adam(D_model.parameters(),lr=1e-5)

    # scheduler
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100000, 150000], gamma=0.3)
    # scheduler_D = lr_scheduler.MultiStepLR(optimizer_D,milestones=[100000,150000],gamma=0.3)
    # gan_criterion = nn.BCELoss()

    print("preparation cost %f seconds" % (time.time() - start))

    start_t = time.time()
    Accu_model.train()
    inpaint_model.train()
    # D_model.train()

    # import pdb; pdb.set_trace()
    data_start = time.time()
    n_epoch = 200
    count = 0

    num_workers = 4
    # train_data = PatchTransferDataset(opt, mode='train')
    train_data = Fusion_dataset_textonly(opt, mode='train')
    train_data_loader = data.DataLoader(dataset=train_data, batch_size=opt['batch_size'], shuffle=True,
                                        num_workers=num_workers, pin_memory=True)
    IUV = cv2.imread("/home/haolin/front_IUV.png")
    for e in range(n_epoch):
        for batch_id, (src_data, tgt_data) in enumerate(train_data_loader):
            count = count + 1
            data_t = time.time() - data_start

            optimizer_accu.zero_grad()
            optimizer_inpaint.zero_grad()
            # optimizer_D.zero_grad()

            src_texture_im, src_mask_im = src_data
            src_texture_im = src_texture_im.permute(0, 1, 4, 2, 3).float().to(device)
            src_mask_im = src_mask_im.byte().to(device)

            tgt_texture_im, tgt_mask_im = tgt_data
            tgt_texture_im = tgt_texture_im.permute(0, 1, 4, 2, 3).float().to(device)
            tgt_mask_im = tgt_mask_im.float().to(device)

            # print(src_texture_im.shape)
            # print(src_mask_im.shape)
            # print(tgt_texture_im.shape)
            # print(tgt_mask_im.shape)

            # import pdb;
            # pdb.set_trace()

            # train
            # label = torch.cat([tgt_IUV, transfered_img,src_img[:,0],src_IUV[:,0],bg_inpaint_mask.unsqueeze(1),fg_inpaint_mask.unsqueeze(1)], dim=1)
            # src_texture_im=torch.cat([src_texture_im[:,0].squeeze(1),src_texture_im[:,1].squeeze(1)],dim=1)#-1,6,800,1200
            src_texture_im_input = []
            random_number = np.random.random()
            if random_number < 0.25:  # only one frame is used
                random_index = np.random.choice(4, 1, replace=False)
            elif random_number < 0.5:
                random_index = np.random.choice(4, 2, replace=False)
            elif random_number < 0.75:
                random_index = np.random.choice(4, 3, replace=False)
            else:
                random_index = np.random.choice(4, 4, replace=False)

            for i in range(4):
                for j in range(6):
                    concat_image = []
                    for z in range(random_index.shape[0]):
                        concat_image.append(
                            src_texture_im[:, random_index[z], :, i * 200:(i + 1) * 200, j * 200:(j + 1) * 200].squeeze(
                                1))
                    src_texture_im_input.append(concat_image)
            random_number_list = []
            for i in range(random_index.shape[0]):
                random_number_list.append(random_index[i])
            if random_number < 0.75:
                for i in range(opt["maximum_ref_frames"]):
                    if i not in random_number_list:
                        src_mask_im[:, i] = src_mask_im[:, i] * 0
            src_mask_im = src_mask_im.unsqueeze(2).repeat(1, 1, 3, 1, 1)
            tgt_mask_im = tgt_mask_im.unsqueeze(2).repeat(1, 1, 3, 1, 1)
            # print(torch.max(src_mask_im),torch.min(src_mask_im))
            # print(src_texture_im.shape,src_mask_im.shape,tgt_mask_im.shape)
            Accu_output_texture = Accu_model(src_texture_im_input)

            if random_number < 0.75:
                for i in range(4):
                    if i not in random_number_list:
                        src_mask_im[:, i] = src_mask_im[:, i] * 0
            # print("finish accumulation")
            src_common_area = torch.Tensor(torch.zeros(src_mask_im[:, 0].squeeze(1).shape)).byte().to(device)
            # print(src_common_area.shape)
            for i in range(4):
                src_common_area = src_common_area | src_mask_im[:, i].byte()

            src_common_area = src_common_area.float()

            for i in range(4):
                for j in range(6):
                    common_area = src_common_area[:, :, i * 200:(i + 1) * 200, j * 200:(j + 1) * 200]
                    # print(common_area.shape,Accu_output_texture[i*6+j].shape)
                    Accu_output_texture[i * 6 + j] = Accu_output_texture[i * 6 + j] * common_area

            inpaint_texture = inpaint_model(Accu_output_texture)
            total_loss = 0
            for z in range(opt["num_target"]):
                for i in range(4):
                    for j in range(6):
                        pred = inpaint_texture[i * 6 + j] * tgt_mask_im[:, z, :, i * 200:(i + 1) * 200,
                                                            j * 200:(j + 1) * 200]
                        target = tgt_texture_im[:, z, :, i * 200:(i + 1) * 200, j * 200:(j + 1) * 200] * tgt_mask_im[:,
                                                                                                         z, :, i * 200:(
                                                                                                                               i + 1) * 200,
                                                                                                         j * 200:(
                                                                                                                         j + 1) * 200]
                        # print(pred.shape,target.shape)
                        total_loss = total_loss + l1_criterion(pred, target)

            total_loss.backward()
            optimizer_accu.step()
            optimizer_inpaint.step()
            # tb_logger.scalar_summary('perceptual loss', loss.detach().cpu().numpy().sum(), batch_id)
            tb_logger.scalar_summary('loss', total_loss, count)

            output_list = []
            # show train results
            if count > 0 and count % opt['vis_interval'] == 0:
                texture_image = torch.zeros(())
                texture_image = texture_image.new_empty((tgt_texture_im.shape[0], 3, 800, 1200)).cuda()
                for i in range(4):
                    for j in range(6):
                        # print(i*6+j)
                        # print(texture_list[i*6+j].shape)
                        texture_image[:, :, i * 200:(i + 1) * 200, j * 200:(j + 1) * 200] = inpaint_texture[
                            i * 6 + j]
                output_texture = np.clip(
                    ((texture_image.permute(0, 2, 3, 1).detach().cpu().numpy() / 2.0 + 0.5) * 255), 0, 255).astype(
                    np.uint8)
                # transfered_img1 = TransferTexture(output_texture[0], IUV) / 255.0
                # transfered_img2 = TransferTexture(output_texture[1], IUV) / 255.0
                # transfered_img3 = TransferTexture(output_texture[2], IUV) / 255.0
                # transfered_img4 = TransferTexture(output_texture[3], IUV) / 255.0
                # transfered_img = np.concatenate((transfered_img1[np.newaxis], transfered_img2[np.newaxis]), axis=0)
                output_texture = output_texture / 255.0
                real = tgt_texture_im[:, 0].permute(0, 2, 3, 1).detach().cpu().numpy() / 2.0 + 0.5

                # vis_image = transfered_img[:, :, :, ::-1]
                vis_image2 = output_texture[:, :, :, ::-1]
                vis_image3 = real[:, :, :, ::-1]

                tb_logger.image_summary("train texture images", vis_image2, count)
                # tb_logger.image_summary("train transfered images", vis_image, count)
                tb_logger.image_summary("real texture images", vis_image3, count)

            # conduct test
            # if batch_id > 0 and batch_id % opt['test_interval'] == 0:
            #     inpaint_model.eval()
            #     try:
            #         with torch.no_grad():
            #             test_x, test_y = next(test_data_loader)
            #
            #             test_x_src_clone = test_x[0].clone().float()
            #             test_y_src_clone = test_y.clone().float()
            #
            #             test_x = map(lambda inp: inp.permute(0, 3, 1, 2).float().to(device), test_x)
            #             test_y = test_y.permute(0, 3, 1, 2).float().to(device)
            #
            #             test_x_src, test_x_pose_src, _, test_x_pose_tgt, _ = test_x
            #
            #             test_pred_y = inpaint_model(test_x_src, test_x_pose_src, test_x_pose_tgt)
            #             test_loss = vgg_criterion(vgg_preprocess(test_pred_y), vgg_preprocess(test_y))
            #
            #             test_pred_y = test_pred_y.detach().permute(0, 2, 3, 1).cpu()
            #
            #             concat_image = torch.cat([test_x_src_clone, test_pred_y, test_y_src_clone], dim=2).numpy()[:, :, :,
            #                            ::-1]
            #             concat_image = concat_image / 2.0 + 0.5
            #
            #             tb_logger.scalar_summary("test loss", test_loss, batch_id)
            #             tb_logger.image_summary("eval images", concat_image, batch_id)
            #     except Exception as ex:
            #         print(ex)
            #         print("Test failed!")
            #
            #     model.train()

            # save model
            if count > 0 and (count + 1) % opt['model_save_interval'] == 0:
                Accu_model.eval()
                try:
                    ckpt_model_filename = "iter_{}.pth".format(count + 1)
                    accu_ckpt_model_path = os.path.join(ckpt_dir, "accu_" + ckpt_model_filename)
                    torch.save(Accu_model.module.state_dict(), accu_ckpt_model_path)
                    print("the aacu model is saved at", accu_ckpt_model_path)
                    inpaint_ckpt_model_path = os.path.join(ckpt_dir, "inpaint_" + ckpt_model_filename)
                    torch.save(inpaint_model.module.state_dict(), inpaint_ckpt_model_path)
                    print("the inpaint model is saved at", inpaint_ckpt_model_path)
                except:
                    print("Save model failed!")

                Accu_model.train()
                # inpaint_model.eval()

                # inpaint_model.save()
                # print("Saved model at iteration #%d" % batch_id)

            # print log
            mesg = "{:0>8},{}:{} ,[{}/{}], {}: {:.3f}".format(
                str(datetime.timedelta(seconds=round(time.time() - start_t))),
                "epoch",
                e,
                batch_id + 1,
                len(train_data_loader),
                'loss',
                total_loss,
            )
            # mesg += '  {}: {:.2f}'.format('data', data_t)
            # for log in logs:
            #     mesg += "{}: {:.3f}  ".format(log[0], log[1])

            print(mesg)
            data_start = time.time()

    print("Training Done.")


if __name__ == '__main__':
    import colored_traceback

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', '-n', type=str, required=True, help='experiment name')
    parser.add_argument('--gpu', type=str, required=True, help='specify gpu devices')
    parser.add_argument('--debug', action='store_true', help='specify debug mode')
    parser.add_argument('--init', type=str, help='weight init method')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    colored_traceback.add_hook()
    train(args)
