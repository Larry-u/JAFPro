import argparse

import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.optim import Adam, lr_scheduler
from torch.utils import data
import torch.nn.functional as F
import numpy as np

import time
import os, sys

sys.path.append("..")
import datetime
import cv2

from src.networks import ImageDiscriminator, VGG_l1_loss, UNet_inpainter, FaceDiscriminator, Accumulate_LSTM_no_loss
from src.data import Fusion_dataset
from options import get_general_options
from src.utils import Logger
from src.crn_model import CRN, CRN_small, CRN_smaller


# from flownet2_pytorch.networks.FlowNetSD import FlowNetSD


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

    start = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
    # cv2.setNumThreads(0)

    # set this to prevent matplotlib import error
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'

    opt = get_general_options()
    opt['model_save_interval'] = 3000 if not args.debug else 3
    opt['vis_interval'] = 200 if not args.debug else 3
    opt['n_training_iter'] = 500000
    opt['batch_size'] = 4

    ckpt_dir = os.path.join(opt['model_save_dir'], model_name)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    opt['network_dir'] = ckpt_dir
    opt["num_target"] = 1
    opt["maximum_ref_frames"] = 4
    opt["self_recon"] = False

    tb_logger = Logger(ckpt_dir)

    # accelerate forwarding
    cudnn.benchmark = True

    num_workers = 8
    # train_data = PatchTransferDataset(opt, mode='train')
    #
    # test_data = PatchTransferDataset(opt, mode='test')
    # test_data_loader = data.DataLoader(dataset=test_data, batch_size=1, num_workers=1,
    #                                    pin_memory=True).__iter__()

    # Model
    Accu_model = Accumulate_LSTM_no_loss()
    Accu_model_dir = os.path.join(opt['model_save_dir'], 'texture_inpaint_210106')
    Accu_model_weight_dir = os.path.join(Accu_model_dir, "accu_iter_20000.pth")
    Accu_model.load_state_dict(torch.load(Accu_model_weight_dir))
    Accu_model = nn.DataParallel(Accu_model).to(device)

    inpaint_model = UNet_inpainter()
    inpaint_model_dir = os.path.join(opt['model_save_dir'], 'texture_inpaint_210106')
    inpaint_model_weight_dir = os.path.join(inpaint_model_dir, "inpaint_iter_20000.pth")
    inpaint_model.load_state_dict(torch.load(inpaint_model_weight_dir))
    inpaint_model = nn.DataParallel(inpaint_model).to(device)

    bg_model = CRN_smaller(3)
    # bg_model_dir=os.path.join(opt['model_save_dir'],'AF_convLSTM_FGAN_0204')
    # bg_model_weight_dir=os.path.join(bg_model_dir,"bg_iter_18000.pth")
    # bg_model.load_state_dict(torch.load(bg_model_weight_dir))
    bg_model = nn.DataParallel(bg_model).to(device)

    refine_model = CRN_smaller(3, fg=True)
    # refine_model_dir=os.path.join(opt['model_save_dir'],'AF_convLSTM_FGAN_0204')
    # refine_model_weight_dir=os.path.join(refine_model_dir,"refine_iter_18000.pth")
    # refine_model.load_state_dict(torch.load(refine_model_weight_dir))
    refine_model = nn.DataParallel(refine_model).to(device)

    discriminator = ImageDiscriminator(ndf=32, input_channel=6)
    # D_model_dir=os.path.join(opt['model_save_dir'],'AF_convLSTM_FGAN_0204')
    # D_model_weight_dir=os.path.join(bg_model_dir,"D_iter_18000.pth")
    # discriminator.load_state_dict(torch.load(D_model_weight_dir))
    discriminator = nn.DataParallel(discriminator).to(device)

    F_Discriminator = FaceDiscriminator(ndf=32, input_channel=6)
    # FD_model_dir=os.path.join(opt['model_save_dir'],'AF_convLSTM_FGAN_0204')
    # FD_model_weight_dir=os.path.join(FD_model_dir,"FD_iter_18000.pth")
    # F_Discriminator.load_state_dict(torch.load(FD_model_weight_dir))
    F_Discriminator = nn.DataParallel(F_Discriminator).to(device)

    # if args.init == 'xavier':
    #    Accu_model.apply(xavier_init)

    # optimizer
    optimizer_accu = Adam(Accu_model.parameters(), lr=1e-4)
    optimizer_inpaint = Adam(inpaint_model.parameters(), lr=1e-4)
    optimizer_bg = Adam(bg_model.parameters(), lr=1e-4)
    optimizer_refine = Adam(refine_model.parameters(), lr=1e-4)
    optimizer_D = Adam(discriminator.parameters(), lr=3e-6)
    optimizer_Face = Adam(F_Discriminator.parameters(), lr=3e-6)
    # RMSprop(params, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    # optimizer_D=RMSprop(discriminator.parameters(),lr=5e-6, alpha=0.99, eps=1e-08)

    gan_criterion = nn.BCELoss().to(device)
    loss_criterion = VGG_l1_loss().to(device)

    print("preparation cost %f seconds" % (time.time() - start))

    start_t = time.time()
    Accu_model.train()
    inpaint_model.train()
    bg_model.train()
    refine_model.train()
    discriminator.train()
    F_Discriminator.train()

    n_epoch = 100

    # import pdb; pdb.set_trace()
    data_start = time.time()
    count = 0
    train_data = Fusion_dataset(opt, mode='train')
    train_data_loader = data.DataLoader(dataset=train_data, batch_size=opt['batch_size'], shuffle=True,
                                        num_workers=num_workers, pin_memory=True)
    for e in range(n_epoch):
        for batch_id, (src_data, tgt_data, data_255) in enumerate(train_data_loader):
            count = count + 1
            data_t = time.time() - data_start

            optimizer_accu.zero_grad()
            optimizer_inpaint.zero_grad()
            optimizer_D.zero_grad()
            optimizer_Face.zero_grad()
            optimizer_bg.zero_grad()
            optimizer_refine.zero_grad()

            # need src_texture_im, src_IUV, tgt_IUV, real, tgt_IUV255

            src_img, src_IUV, src_texture_im, src_mask_im, image_inpaing_mask, src_mask_in_image, src_common_area = src_data
            src_img = src_img.permute(0, 1, 4, 2, 3).float().to(device)
            src_IUV = src_IUV.permute(0, 1, 4, 2, 3).float().to(device)
            src_texture_im = src_texture_im.permute(0, 1, 4, 2, 3).float().to(device)
            # image_inpaing_mask=image_inpaing_mask.permute(0,1,4,2,3).float().to(device) # -1,2,256,256,3
            src_mask_in_image = src_mask_in_image.permute(0, 1, 4, 2, 3).float().to(device)
            src_mask_im = src_mask_im.float().to(device)
            src_common_area = src_common_area.float().to(device)

            tgt_img, tgt_IUV, tgt_texture_im, tgt_mask_im, tgt_mask_in_image, face_mask, face_bbox = tgt_data
            tgt_img = tgt_img.permute(0, 1, 4, 2, 3).float().to(device)
            tgt_IUV = tgt_IUV.permute(0, 1, 4, 2, 3).float().to(device)
            tgt_mask_in_image = tgt_mask_in_image.permute(0, 1, 4, 2, 3).float().to(device)
            # tgt_texture_im=tgt_texture_im.permute(0,1,4,2,3).float().to(device)
            # tgt_mask_im=tgt_mask_im.float().to(device)
            face_mask = face_mask.float().to(device)
            face_mask = face_mask.unsqueeze(2).repeat(1, 1, 3, 1, 1)
            face_bbox = face_bbox.to(device)
            bg_mask = (1 - src_mask_in_image[:, 0].squeeze(1))
            bg_incomplete = bg_mask * src_img[:, 0].squeeze(1) + (1 - bg_mask) * torch.randn(bg_mask.shape).cuda()

            src_IUV255 = data_255[0][:, 0].squeeze()
            tgt_IUV255 = data_255[1].squeeze(1)
            src_IUV255 = src_IUV255.numpy()
            tgt_IUV255 = tgt_IUV255.numpy()

            # print("texture_map's shape is:",src_texture_im.shape)
            # print("tgt_IUV255's shape is",tgt_IUV255.shape)
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

            Accu_output_texture = Accu_model(src_texture_im_input)

            random_number_list = []
            for i in range(random_index.shape[0]):
                random_number_list.append(random_index[i])
            if random_number < 0.75:
                for i in range(4):
                    if i not in random_number_list:
                        src_mask_im[:, i] = src_mask_im[:, i] * 0

            src_common_area = (src_common_area * 0).byte()
            for i in range(4):
                src_common_area = src_common_area | src_mask_im[:, i].byte()

            src_common_area = src_common_area.float()
            src_common_area = src_common_area.unsqueeze(1).repeat(1, 3, 1, 1)

            for i in range(4):
                for j in range(6):
                    common_area = src_common_area[:, :, i * 200:(i + 1) * 200, j * 200:(j + 1) * 200]
                    Accu_output_texture[i * 6 + j] = Accu_output_texture[i * 6 + j] * common_area

            inpaint_texture = inpaint_model(Accu_output_texture)
            # print(inpaint_texture_list[0].shape)

            '''
            Accu_warp=torch.full((src_img[:,0].squeeze(1).shape),0,device=device)
            for i in range(src_img.shape[0]):
                Accu_output_list=list(map(lambda inp: inp[i], Accu_output_texture))
                Accu_warp[i]=texture_warp_pytorch(Accu_output_list,tgt_IUV255[i],device)
            '''
            inpaint_warp = torch.full((src_img[:, 0].squeeze(1).shape), 0, device=device)
            for i in range(src_img.shape[0]):
                inpaint_texture_list = list(map(lambda inp: inp[i], inpaint_texture))
                inpaint_warp[i] = texture_warp_pytorch(inpaint_texture_list, tgt_IUV255[i], device)

            # print(Accu_warp.shape,mask.shape)
            # print(torch.max(Accu_warp),torch.min(Accu_warp))
            # print(torch.max(mask),torch.min(mask))
            # sprint(Accu_warp.shape,mask[:,0].shape,tgt_IUV[:,0].squeeze(1).shape)
            refine_output, fg_mask = refine_model(inpaint_warp, 256)
            # with torch.no_grad():
            bg_output = bg_model(bg_incomplete, 256)
            final_output = refine_output * fg_mask.repeat(1, 3, 1, 1) + bg_output * (1 - fg_mask.repeat(1, 3, 1, 1))
            # print(inpaint_output.shape)
            target = tgt_img.squeeze(1)
            loss = loss_criterion(final_output, target)

            face_pred = []
            face_real = []
            face_IUV = []
            all_valid = True
            for i in range(tgt_img.shape[0]):
                if face_bbox[i, 0, 0] == face_bbox[i, 0, 1]:
                    all_valid = False
                    continue
                face_org_pred = final_output[i, :, face_bbox[i, 0, 2]:face_bbox[i, 0, 3],
                                face_bbox[i, 0, 0]:face_bbox[i, 0, 1]]
                face_org_real = tgt_img[i, 0, :, face_bbox[i, 0, 2]:face_bbox[i, 0, 3],
                                face_bbox[i, 0, 0]:face_bbox[i, 0, 1]]
                face_org_IUV = tgt_IUV[i, 0, :, face_bbox[i, 0, 2]:face_bbox[i, 0, 3],
                               face_bbox[i, 0, 0]:face_bbox[i, 0, 1]]
                face_pred.append(F.upsample(face_org_pred.unsqueeze(0), size=(64, 64), mode='bilinear'))
                face_real.append(F.upsample(face_org_real.unsqueeze(0), size=(64, 64), mode='bilinear'))
                face_IUV.append(F.upsample(face_org_IUV.unsqueeze(0), size=(64, 64), mode='nearest'))
            face_pred = torch.cat(face_pred, dim=0)
            face_real = torch.cat(face_real, dim=0)
            face_IUV = torch.cat(face_IUV, dim=0)
            # print(face_pred.shape)

            # print("predict's shape is",predict_output.shape,"inpaint_warp's shape is",inpaint_warp.shape,"IUV's shape is:",tgt_IUV.shape)
            ###train the face discriminator
            real_label = torch.full((face_IUV.shape[0],), 1, device=device)
            fake_label = torch.full((face_IUV.shape[0],), 0, device=device)
            F_errD = 0
            F_errG = 0
            for i in range(3):
                real_input = torch.cat([face_real, face_IUV], dim=1)
                pred_real = F_Discriminator(real_input)
                F_errD_real = gan_criterion(pred_real, real_label)  # gan_criterion(pred_real,real_label)
                F_errD_real.backward()

                fake_input = torch.cat([face_pred.detach(), face_IUV], dim=1)
                pred_fake = F_Discriminator(fake_input)
                F_errD_fake = gan_criterion(pred_fake, fake_label)
                F_errD_fake.backward()

                F_errD = F_errD_real + F_errD_fake
                optimizer_Face.step()
            ###train the discriminator
            errD = 0
            errG = 0
            real_label = torch.full((src_img.shape[0],), 1, device=device)
            fake_label = torch.full((src_img.shape[0],), 0, device=device)
            for i in range(3):
                # for p in discriminator.parameters(): p.data.clamp_(-0.01, 0.01)

                real_input = torch.cat([tgt_img.squeeze(1), src_img[:, 0].squeeze(1)], dim=1)
                pred_real = discriminator(real_input)
                errD_real = gan_criterion(pred_real, real_label)  # gan_criterion(pred_real,real_label)
                errD_real.backward()

                fake_input = torch.cat([final_output.detach(), src_img[:, 0].squeeze(1)], dim=1)
                pred_fake = discriminator(fake_input)
                errD_fake = gan_criterion(pred_fake, fake_label)
                errD_fake.backward()

                errD = errD_real + errD_fake
                optimizer_D.step()

            ####train the generator

            generator_pred = discriminator(torch.cat([final_output, src_img[:, 0].squeeze(1)], dim=1))
            generator_face = F_Discriminator(torch.cat([face_pred, face_IUV], dim=1))
            errG = gan_criterion(generator_pred, real_label)

            real_label = torch.full((face_IUV.shape[0],), 1, device=device)
            fake_label = torch.full((face_IUV.shape[0],), 0, device=device)
            F_errG = gan_criterion(generator_face, real_label)
            # collect loss

            total_loss = loss.sum() + 2 * errG + 2 * F_errG  # errG cannot be too large
            total_loss.backward()
            optimizer_accu.step()
            optimizer_inpaint.step()
            optimizer_bg.step()
            optimizer_refine.step()

            # outputs, gen_loss, dis_loss, logs = Accu_model.process(transfered_img, tgt_IUV, fg_mask, real)

            # Accu_model.backward(gen_loss, dis_loss)

            # logging
            # for log in logs:

            # import pdb; pdb.set_trace()
            loss = total_loss.detach().cpu().numpy().sum()
            # tb_logger.scalar_summary('perceptual loss', loss.detach().cpu().numpy().sum(), batch_id)
            tb_logger.scalar_summary('total loss', loss, count)
            tb_logger.scalar_summary('D loss', errD.item(), count)
            tb_logger.scalar_summary('G loss', errG.item(), count)
            tb_logger.scalar_summary('Face D loss', F_errD.item(), count)
            tb_logger.scalar_summary('Face G loss', F_errG.item(), count)

            # show train results
            if count % opt['vis_interval'] == 0 and all_valid == True:
                print("visualizing images in tensorboard...")
                real = tgt_img.squeeze(1).detach().cpu() / 2.0 + 0.5
                pred = final_output.detach().cpu() / 2.0 + 0.5
                source_img = src_img[:, 0].squeeze(1).detach().cpu() / 2.0 + 0.5
                source_IUV = src_IUV[:, 0].squeeze(1).detach().cpu() / 2.0 + 0.5
                target_IUV = tgt_IUV.squeeze(1).detach().cpu() / 2.0 + 0.5
                bg = bg_output.detach().cpu() / 2.0 + 0.5
                inpaint_warp = refine_output.detach().cpu() / 2.0 + 0.5
                refine_output = refine_output.detach().cpu() / 2.0 + 0.5
                face_real = F.upsample(face_real, size=(256, 256), mode='bilinear').detach().cpu() / 2.0 + 0.5
                face_pred = F.upsample(face_pred, size=(256, 256), mode='bilinear').detach().cpu() / 2.0 + 0.5

                row1 = []
                row1.append(source_IUV)
                row1.append(target_IUV)
                row1.append(refine_output)
                row1.append(bg)
                row1.append(face_real)
                row1 = torch.cat(row1, dim=3)
                row2 = []
                row2.append(inpaint_warp)
                row2.append(source_img)
                row2.append(real)
                row2.append(pred)
                row2.append(face_pred)
                row2 = torch.cat(row2, dim=3)

                vis_image = torch.cat([row1, row2], dim=2).permute(0, 2, 3, 1).numpy()
                vis_image = vis_image[:, :, :, ::-1]
                vis_image = np.clip(vis_image, 0, 1)

                # concat_image = torch.cat([src_img, src_IUV, tgt_IUV, outputs, real], dim=2).numpy()
                # concat_image = concat_image / 2.0 + 0.5
                tb_logger.image_summary("train images", vis_image, count)
                texture_image = torch.zeros(())
                texture_image = texture_image.new_empty((src_img.shape[0], 3, 800, 1200)).cuda()
                for i in range(4):
                    for j in range(6):
                        # print(i*6+j)
                        # print(texture_list[i*6+j].shape)
                        texture_image[:, :, i * 200:(i + 1) * 200, j * 200:(j + 1) * 200] = inpaint_texture[i * 6 + j][
                                                                                            :, :, :, :]
                output_texture = np.clip(((texture_image.permute(0, 2, 3, 1).detach().cpu().numpy() / 2.0 + 0.5) * 255),
                                         0, 255).astype(np.uint8)
                tb_logger.image_summary("texture images", output_texture, count)
                # tb_logger.image_summary("train images", concat_image, batch_id)

            # conduct test
            # if batch_id > 0 and batch_id % opt['test_interval'] == 0:
            #     Accu_model.eval()
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
            #             test_pred_y = Accu_model(test_x_src, test_x_pose_src, test_x_pose_tgt)
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
            if count > 0 and count % opt['model_save_interval'] == 0:
                # Accu_model.eval()
                try:
                    ckpt_model_filename = "iter_{}.pth".format(count)
                    # ckpt_model_filename = "crn_latest.pth"
                    ckpt_Accu_model_path = os.path.join(ckpt_dir, "Accu_" + ckpt_model_filename)
                    ckpt_inpaint_model_path = os.path.join(ckpt_dir, "inpaint_" + ckpt_model_filename)
                    ckpt_bg_model_path = os.path.join(ckpt_dir, "bg_" + ckpt_model_filename)
                    ckpt_refine_model_path = os.path.join(ckpt_dir, "refine_" + ckpt_model_filename)
                    ckpt_D_model_path = os.path.join(ckpt_dir, "D_" + ckpt_model_filename)
                    ckpt_FD_model_path = os.path.join(ckpt_dir, "FD_" + ckpt_model_filename)
                    torch.save(Accu_model.module.state_dict(), ckpt_Accu_model_path)
                    torch.save(inpaint_model.module.state_dict(), ckpt_inpaint_model_path)
                    torch.save(discriminator.module.state_dict(), ckpt_D_model_path)
                    torch.save(F_Discriminator.module.state_dict(), ckpt_FD_model_path)
                    torch.save(bg_model.module.state_dict(), ckpt_bg_model_path)
                    torch.save(refine_model.module.state_dict(), ckpt_refine_model_path)

                except:
                    print("Save model failed!")

                Accu_model.train()
                inpaint_model.train()
                # bg_model.train()
                refine_model.train()
                discriminator.train()
                F_Discriminator.train()
                # Accu_model.eval()

                # Accu_model.save()
                # print("Saved model at iteration #%d" % batch_id)

            # print log
            mesg = "{:0>8},{}:{} ,[{}/{}], {}: {:.3f},{}: {:.3f},{}: {:.3f},{}: {:.3f},{}: {:.3f}".format(
                str(datetime.timedelta(seconds=round(time.time() - start_t))),
                "epoch",
                e,
                batch_id + 1,
                len(train_data_loader),
                'total loss',
                loss,
                'D loss',
                errD,
                'G loss',
                errG,
                'Face D loss',
                F_errD,
                'Face G loss',
                F_errG,
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
