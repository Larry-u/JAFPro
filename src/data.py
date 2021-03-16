from random import randint, uniform
import torch.utils.data as data
import os, sys

sys.path.append("..")
from src.utils import get_vid_list, get_img_iuv_text_mask, TransferTexture, get_mask_list
import numpy as np
import cv2
from src.computer_angle import compute_angle
import pickle

from options import get_general_options


class Fusion_dataset(data.Dataset):
    def __init__(self, params, mode='train'):
        super(Fusion_dataset, self).__init__()

        data_root = params['data_root']
        self.data_dir = os.path.join(data_root, mode)

        self.vid_list = get_vid_list(self.data_dir)

        self.batch_size = params['batch_size']
        self.n_iters = params['n_training_iter']
        self.n_sample = params['n_sample']  # get N+1 control points
        self.max_ref_frames = params["maximum_ref_frames"]

        num_vids = len(self.vid_list)
        self.num_frames = params['num_frames']
        self.frame_interval = params['frame_interval']
        use_fix_intv = params['use_fix_interval']
        self.num_inputs = params["maximum_ref_frames"]
        self.num_target = params["num_target"]
        self.fix_frame = params["fix_frame"]
        self.face_GAN = params["face_GAN"]
        self.output_mask = params["output_mask"]
        self.self_recon = params["self_recon"]

        self.dataset_len = len(self.vid_list)

    def __getitem__(self, index):
        vid_idx = index

        random_number = np.random.random()

        vid_path = self.vid_list[vid_idx]
        img_list, iuv_list, text_list, mask_list = get_img_iuv_text_mask(vid_path)

        frames = np.random.choice(len(img_list), self.num_inputs + self.num_target, replace=False)

        if not self.fix_frame:
            if random_number < 0.33333:
                frames[1 + self.num_target] = frames[self.num_target]
                frames[2 + self.num_target] = frames[self.num_target]
            elif random_number < 0.66666:
                frames[1 + self.num_target] = frames[self.num_target]
        if self.self_recon:
            random_number_recon = np.random.random()
            if random_number_recon < 0.3:
                random_index = np.random.choice(self.num_inputs, 1)
                frames[random_index] = frames[self.num_target]
                print("self_reconstruction")

        src_texture_im = np.zeros((self.num_inputs, 800, 1200, 3), np.uint8)
        for i in range(self.num_inputs):
            src_texture_im[i] = cv2.imread(text_list[frames[i + self.num_target]])

        tgt_texture_im = np.zeros((self.num_target, 800, 1200, 3), np.uint8)
        for i in range(self.num_target):
            tgt_texture_im[i] = cv2.imread(text_list[frames[i]])

        src_mask_im = np.zeros((self.num_inputs, 800, 1200), np.uint8)
        for i in range(self.num_inputs):
            src_mask_im[i] = cv2.imread(mask_list[frames[i + self.num_target]])[:, :, 0]
        tgt_mask_im = np.zeros((self.num_target, 800, 1200), np.uint8)
        for i in range(self.num_target):
            tgt_mask_im[i] = cv2.imread(mask_list[frames[i]])[:, :, 0]

        src_img = np.zeros((self.num_inputs, 256, 256, 3), np.uint8)
        for i in range(self.num_inputs):
            src_img[i] = cv2.imread(img_list[frames[i + self.num_target]])
        tgt_img = np.zeros((self.num_target, 256, 256, 3), np.uint8)
        for i in range(self.num_target):
            tgt_img[i] = cv2.imread(img_list[frames[i]])

        src_IUV = np.zeros((self.num_inputs, 256, 256, 3), np.uint8)
        for i in range(self.num_inputs):
            src_IUV[i] = cv2.imread(iuv_list[frames[i + self.num_target]])
        tgt_IUV = np.zeros((self.num_target, 256, 256, 3), np.uint8)
        for i in range(self.num_target):
            tgt_IUV[i] = cv2.imread(iuv_list[frames[i]])

        if self.output_mask == True:

            src_common_area = np.zeros((800, 1200), np.uint8)
            for i in range(self.num_inputs):
                src_common_area = np.logical_or(src_common_area, src_mask_im[i] / 255)

            src_area = np.zeros((self.num_target, 256, 256, 3), np.uint8)
            for i in range(self.num_target):
                src_area[i] = TransferTexture(TextureIm=np.repeat(src_common_area[:, :, np.newaxis], 3, axis=2),
                                              IUV=tgt_IUV[i])

            image_inpaint_area = np.zeros((self.num_target, 256, 256, 3), np.uint8)
            tgt_mask_in_image = np.zeros((self.num_target, 256, 256, 3), np.uint8)
            for i in range(self.num_target):
                tgt_mask_in_image[i] = TransferTexture(TextureIm=np.ones((800, 1200, 3), np.uint8), IUV=tgt_IUV[i])
                image_inpaint_area[i] = np.logical_xor(tgt_mask_in_image[i], src_area[i])

            src_mask_in_image = np.zeros((self.num_inputs, 256, 256, 3), np.uint8)
            for i in range(self.num_inputs):
                src_mask_in_image[i] = TransferTexture(TextureIm=np.ones((800, 1200, 3), np.uint8), IUV=src_IUV[i])

        # compute the face Bbox
        # 23 and 24 correspond to the face
        if self.face_GAN == True:
            face_bbox = np.zeros((self.num_target, 4), np.uint8)
            for i in range(self.num_target):
                try:
                    Y1, X1 = np.where(tgt_IUV[i, :, :, 0] == 23)
                    Y2, X2 = np.where(tgt_IUV[i, :, :, 0] == 24)
                    X_con = np.concatenate([X1, X2])
                    Y_con = np.concatenate([Y1, Y2])
                    leftmost = max(np.min(X_con) - 2, 0)
                    rightmost = min(np.max(X_con) + 3, 256)
                    upmost = max(np.min(Y_con) - 2, 0)
                    bottomost = min(np.max(Y_con) + 3, 256)
                    face_bbox[i] = np.array([leftmost, rightmost, upmost, bottomost])
                    # print(face_bbox[i])
                except:
                    face_bbox = np.zeros((self.num_target, 4), np.uint8)
            face_mask = np.zeros((self.num_target, 256, 256), np.uint8)
            for i in range(self.num_target):
                face_mask[i] = np.where(tgt_IUV[i, :, :, 0] == 23, 1, 0)
                face_mask[i] = face_mask[i] + np.where(tgt_IUV[i, :, :, 0] == 24, 1, 0)
        # cv2.imwrite('/home/haolin/test_data/face_mask.jpg',face_mask)

        # canvas=tgt_IUV[0]
        # canvas=cv2.rectangle(canvas, (face_bbox[0,0], face_bbox[0,2]), (face_bbox[0,1], face_bbox[0,3]), (0, 255, 0), 2)
        # cv2.imwrite('/home/haolin/test_data/canvas.jpg',canvas)
        # cv2.imwrite('/home/haolin/test_data/IUV.jpg',tgt_IUV[0,:,:])

        # print(face_mask.shape)

        # normalize to (-1, 1)

        src_IUV255 = src_IUV
        tgt_IUV255 = tgt_IUV

        src_texture_im = (src_texture_im / 255.0 - 0.5) * 2
        tgt_texture_im = (tgt_texture_im / 255.0 - 0.5) * 2
        src_IUV = (src_IUV / 255.0 - 0.5) * 2
        tgt_IUV = (tgt_IUV / 255.0 - 0.5) * 2
        src_img = (src_img / 255.0 - 0.5) * 2
        tgt_img = (tgt_img / 255.0 - 0.5) * 2
        src_mask_im = (src_mask_im / 255.0)
        tgt_mask_im = (tgt_mask_im / 255.0)

        src_data = [src_img, src_IUV, src_texture_im, src_mask_im]
        tgt_data = [tgt_img, tgt_IUV, tgt_texture_im, tgt_mask_im]
        data_255 = [src_IUV255, tgt_IUV255]
        if self.output_mask == True:
            src_data.append(image_inpaint_area)
            src_data.append(src_mask_in_image)
            src_data.append(src_common_area)
            tgt_data.append(tgt_mask_in_image)
        if self.face_GAN == True:
            # prepare return values
            tgt_data.append(face_mask)
            tgt_data.append(face_bbox)

        '''
        for i in range(len(src_data)):
            cv2.imwrite("/data1/haolin/test/"+"src_%i.jpg"%(i),src_data[i])
        for i in range(len(tgt_data)):
            cv2.imwrite("/data1/haolin/test/"+"tgt_%i.jpg"%(i),tgt_data[i])
        '''

        # return {'IUV': tgt_IUV, 'transfered_img': transfered_img, 'y': y, 'tgt_fg_mask': tgt_fg_mask}
        return src_data, tgt_data, data_255

    def __len__(self):
        return self.dataset_len


class Fusion_dataset_textonly(data.Dataset):
    def __init__(self, params, mode='train'):
        super(Fusion_dataset_textonly, self).__init__()

        data_root = params['data_root']
        self.data_dir = os.path.join(data_root, mode)

        self.vid_list = get_vid_list(self.data_dir)

        self.batch_size = params['batch_size']
        self.n_iters = params['n_training_iter']
        self.n_sample = params['n_sample']  # get N+1 control points
        self.max_ref_frames = params["maximum_ref_frames"]

        num_vids = len(self.vid_list)
        self.num_frames = params['num_frames']
        self.frame_interval = params['frame_interval']
        use_fix_intv = params['use_fix_interval']
        self.num_inputs = params["maximum_ref_frames"]
        self.num_target = params["num_target"]
        self.fix_frame = params["fix_frame"]
        self.face_GAN = params["face_GAN"]
        self.output_mask = params["output_mask"]

        self.dataset_len = len(self.vid_list)

    def __getitem__(self, index):
        vid_idx = index

        frames = np.random.choice(self.num_frames, self.num_inputs + self.num_target, replace=False)

        random_number = np.random.random()

        if not self.fix_frame:
            if random_number < 0.33333:
                frames[1 + self.num_target] = frames[self.num_target]
                frames[2 + self.num_target] = frames[self.num_target]
            elif random_number < 0.66666:
                frames[1 + self.num_target] = frames[self.num_target]

        vid_path = self.vid_list[vid_idx]
        img_list, iuv_list, text_list, mask_list = get_img_iuv_text_mask(vid_path)

        src_texture_im = np.zeros((self.num_inputs, 800, 1200, 3), np.uint8)
        for i in range(self.num_inputs):
            src_texture_im[i] = cv2.imread(text_list[frames[i + self.num_target]])

        tgt_texture_im = np.zeros((self.num_target, 800, 1200, 3), np.uint8)
        for i in range(self.num_target):
            tgt_texture_im[i] = cv2.imread(text_list[frames[i]])

        src_mask_im = np.zeros((self.num_inputs, 800, 1200), np.uint8)
        for i in range(self.num_inputs):
            src_mask_im[i] = cv2.imread(mask_list[frames[i + self.num_target]])[:, :, 0]
        tgt_mask_im = np.zeros((self.num_target, 800, 1200), np.uint8)
        for i in range(self.num_target):
            tgt_mask_im[i] = cv2.imread(mask_list[frames[i]])[:, :, 0]

        src_texture_im = (src_texture_im / 255.0 - 0.5) * 2
        tgt_texture_im = (tgt_texture_im / 255.0 - 0.5) * 2
        src_mask_im = (src_mask_im / 255.0)
        tgt_mask_im = (tgt_mask_im / 255.0)

        src_data = [src_texture_im, src_mask_im]
        tgt_data = [tgt_texture_im, tgt_mask_im]

        return src_data, tgt_data

    def __len__(self):
        return self.dataset_len


class Fusion_dataset_smpl(data.Dataset):
    def __init__(self, params, mode='train'):
        super(Fusion_dataset_smpl, self).__init__()

        data_root = params['data_root']
        smpl_root = params['smpl_root']
        mask_root = params['mask_root']
        self.data_dir = os.path.join(data_root, mode)
        self.smpl_dir = os.path.join(smpl_root, mode)
        self.mask_dir = os.path.join(mask_root, mode)

        self.vid_list = get_vid_list(self.data_dir)

        self.batch_size = params['batch_size']
        self.n_iters = params['n_training_iter']
        self.n_sample = params['n_sample']  # get N+1 control points
        self.max_ref_frames = params["maximum_ref_frames"]

        num_vids = len(self.vid_list)
        self.num_frames = params['num_frames']
        self.frame_interval = params['frame_interval']
        use_fix_intv = params['use_fix_interval']
        self.num_inputs = params["maximum_ref_frames"]
        self.num_target = params["num_target"]
        self.fix_frame = params["fix_frame"]
        self.face_GAN = params["face_GAN"]
        self.output_mask = params["output_mask"]
        self.self_recon = params["self_recon"]

        self.dataset_len = len(self.vid_list)

    def __getitem__(self, index):
        vid_idx = index

        random_number = np.random.random()

        vid_path = self.vid_list[vid_idx]
        vid_name = vid_path.split("/")[-1]
        smpl_path = os.path.join(self.smpl_dir, vid_name)
        smpl_path = os.path.join(smpl_path, "pose_shape.pkl")
        mask_path = os.path.join(self.mask_dir, vid_name)
        real_mask_list = get_mask_list(mask_path)
        # print(real_mask_list)
        img_list, iuv_list, text_list, mask_list = get_img_iuv_text_mask(vid_path)

        frames = np.random.choice(len(img_list) - 1, self.num_inputs + self.num_target, replace=False) + 1
        target_index = frames[0]
        prev_real_index = target_index - 1

        src_texture_im = np.zeros((self.num_inputs, 800, 1200, 3), np.uint8)
        for i in range(self.num_inputs):
            src_texture_im[i] = cv2.imread(text_list[frames[i + self.num_target]])

        src_mask_im = np.zeros((self.num_inputs, 800, 1200), np.uint8)
        for i in range(self.num_inputs):
            src_mask_im[i] = cv2.imread(mask_list[frames[i + self.num_target]])[:, :, 0]

        src_img = np.zeros((self.num_inputs, 256, 256, 3), np.uint8)
        for i in range(self.num_inputs):
            src_img[i] = cv2.imread(img_list[frames[i + self.num_target]])
        tgt_img = np.zeros((self.num_target, 256, 256, 3), np.uint8)
        for i in range(self.num_target):
            tgt_img[i] = cv2.imread(img_list[frames[i]])

        src_IUV = np.zeros((self.num_inputs, 256, 256, 3), np.uint8)
        for i in range(self.num_inputs):
            src_IUV[i] = cv2.imread(iuv_list[frames[i + self.num_target]])
        tgt_IUV = np.zeros((self.num_target, 256, 256, 3), np.uint8)
        for i in range(self.num_target):
            tgt_IUV[i] = cv2.imread(iuv_list[frames[i]])

        if self.output_mask == True:

            src_common_area = np.zeros((800, 1200), np.uint8)
            for i in range(self.num_inputs):
                src_common_area = np.logical_or(src_common_area, src_mask_im[i] / 255)

            src_area = np.zeros((self.num_target, 256, 256, 3), np.uint8)
            for i in range(self.num_target):
                src_area[i] = TransferTexture(TextureIm=np.repeat(src_common_area[:, :, np.newaxis], 3, axis=2),
                                              IUV=tgt_IUV[i])

            image_inpaint_area = np.zeros((self.num_target, 256, 256, 3), np.uint8)
            tgt_mask_in_image = np.zeros((self.num_target, 256, 256, 3), np.uint8)
            for i in range(self.num_target):
                tgt_mask_in_image[i] = TransferTexture(TextureIm=np.ones((800, 1200, 3), np.uint8), IUV=tgt_IUV[i])
                image_inpaint_area[i] = np.logical_xor(tgt_mask_in_image[i], src_area[i])

            src_mask_in_image = np.zeros((self.num_inputs, 256, 256, 3), np.uint8)
            for i in range(self.num_inputs):
                src_mask_in_image[i] = TransferTexture(TextureIm=np.ones((800, 1200, 3), np.uint8), IUV=src_IUV[i])

        # compute the face Bbox
        # 23 and 24 correspond to the face
        if self.face_GAN == True:
            face_bbox = np.zeros((self.num_target, 4), np.uint8)
            for i in range(self.num_target):
                try:
                    Y1, X1 = np.where(tgt_IUV[i, :, :, 0] == 23)
                    Y2, X2 = np.where(tgt_IUV[i, :, :, 0] == 24)
                    X_con = np.concatenate([X1, X2])
                    Y_con = np.concatenate([Y1, Y2])
                    leftmost = max(np.min(X_con) - 2, 0)
                    rightmost = min(np.max(X_con) + 3, 256)
                    upmost = max(np.min(Y_con) - 2, 0)
                    bottomost = min(np.max(Y_con) + 3, 256)
                    face_bbox[i] = np.array([leftmost, rightmost, upmost, bottomost])
                    # print(face_bbox[i])
                except:
                    face_bbox = np.zeros((self.num_target, 4), np.uint8)
            face_mask = np.zeros((self.num_target, 256, 256), np.uint8)
            for i in range(self.num_target):
                face_mask[i] = np.where(tgt_IUV[i, :, :, 0] == 23, 1, 0)
                face_mask[i] = face_mask[i] + np.where(tgt_IUV[i, :, :, 0] == 24, 1, 0)
        # cv2.imwrite('/home/haolin/test_data/face_mask.jpg',face_mask)

        # canvas=tgt_IUV[0]
        # canvas=cv2.rectangle(canvas, (face_bbox[0,0], face_bbox[0,2]), (face_bbox[0,1], face_bbox[0,3]), (0, 255, 0), 2)
        # cv2.imwrite('/home/haolin/test_data/canvas.jpg',canvas)
        # cv2.imwrite('/home/haolin/test_data/IUV.jpg',tgt_IUV[0,:,:])

        # print(face_mask.shape)

        # normalize to (-1, 1)
        smpl_file = open(smpl_path, 'rb')
        smpl_content = pickle.load(smpl_file)

        smpl_pair_index = np.array([prev_real_index, target_index])
        smpl_pair = np.concatenate([smpl_content['cams'][smpl_pair_index], smpl_content['pose'][smpl_pair_index],
                                    smpl_content['shape'][smpl_pair_index]], axis=1)

        prev_real_img = cv2.imread(img_list[prev_real_index])
        smpl_real_mask = cv2.imread(real_mask_list[target_index])

        prev_real_img = (prev_real_img / 255.0 - 0.5) * 2
        smpl_real_mask = smpl_real_mask / 255.0

        smpl_data = [smpl_pair, prev_real_img, smpl_real_mask, smpl_content['vertices'][smpl_pair_index]]

        src_IUV255 = src_IUV
        tgt_IUV255 = tgt_IUV

        src_texture_im = (src_texture_im / 255.0 - 0.5) * 2
        src_IUV = (src_IUV / 255.0 - 0.5) * 2
        tgt_IUV = (tgt_IUV / 255.0 - 0.5) * 2
        src_img = (src_img / 255.0 - 0.5) * 2
        tgt_img = (tgt_img / 255.0 - 0.5) * 2
        src_mask_im = (src_mask_im / 255.0)

        src_data = [src_img, src_IUV, src_texture_im, src_mask_im]
        tgt_data = [tgt_img, tgt_IUV]
        data_255 = [src_IUV255, tgt_IUV255]
        if self.output_mask == True:
            src_data.append(image_inpaint_area)
            src_data.append(src_mask_in_image)
            src_data.append(src_common_area)
        if self.face_GAN == True:
            # prepare return values
            tgt_data.append(face_mask)
            tgt_data.append(face_bbox)

        '''
        for i in range(len(src_data)):
            cv2.imwrite("/data1/haolin/test/"+"src_%i.jpg"%(i),src_data[i])
        for i in range(len(tgt_data)):
            cv2.imwrite("/data1/haolin/test/"+"tgt_%i.jpg"%(i),tgt_data[i])
        '''

        # return {'IUV': tgt_IUV, 'transfered_img': transfered_img, 'y': y, 'tgt_fg_mask': tgt_fg_mask}
        return src_data, tgt_data, data_255, smpl_data

    def __len__(self):
        return self.dataset_len


class Fusion_dataset_smpl_test(data.Dataset):
    def __init__(self, params, mode='test'):
        super(Fusion_dataset_smpl_test, self).__init__()

        data_root = params['data_root']
        smpl_root = params['smpl_root']
        mask_root = params['mask_root']
        self.data_dir = os.path.join(data_root, mode)
        self.smpl_dir = os.path.join(smpl_root, mode)
        self.mask_dir = os.path.join(mask_root, mode)

        self.vid_list = get_vid_list(self.data_dir)
        self.vid_list.sort()

        self.batch_size = params['batch_size']
        self.n_iters = params['n_training_iter']
        self.n_sample = params['n_sample']  # get N+1 control points
        self.max_ref_frames = params["maximum_ref_frames"]

        num_vids = len(self.vid_list)
        self.num_frames = params['num_frames']
        self.frame_interval = params['frame_interval']
        use_fix_intv = params['use_fix_interval']
        self.num_inputs = params["maximum_ref_frames"]
        self.num_target = params["num_target"]
        self.fix_frame = params["fix_frame"]
        self.face_GAN = params["face_GAN"]
        self.output_mask = params["output_mask"]
        self.self_recon = params["self_recon"]

        self.log_file_dir = os.path.join(os.path.dirname(params['test_save_dir']), "log_result",
                                         "chosen_frame_train.txt")

        self.dataset_len = len(self.vid_list)
        print("the dataset len is ", self.dataset_len)
        # self.dataset_len = 1

    def __getitem__(self, index):
        vid_idx = index
        '''
        target_video_name="Popping_video_17_7"
        for i,video in enumerate(self.vid_list):
            print(video)
            if video.find(target_video_name)>=0:
                vid_idx=i
                break
        print(vid_idx)
        '''

        random_number = np.random.random()

        vid_path = self.vid_list[vid_idx]
        vid_name = vid_path.split("/")[-1]
        smpl_path = os.path.join(self.smpl_dir, vid_name)
        smpl_path = os.path.join(smpl_path, "pose_shape.pkl")
        mask_path = os.path.join(self.mask_dir, vid_name)
        real_mask_list = get_mask_list(mask_path)
        # print(real_mask_list)
        img_list, iuv_list, text_list, mask_list = get_img_iuv_text_mask(vid_path)
        img_name_list = []
        for image_path in img_list:
            img_name = image_path.split('/')[-1]
            img_name_list.append(img_name)

        self.num_frames = len(img_list)
        all_IUV = np.zeros((self.num_frames, 256, 256, 3), np.uint8)
        angle = np.zeros((self.num_frames))
        for i in range(self.num_frames):
            all_IUV[i] = cv2.imread(iuv_list[i])
        for i in range(self.num_frames):
            angle[i] = compute_angle(all_IUV[i])
            # print(angle[i])
        max_angle = np.max(angle)
        min_angle = np.min(angle)
        median_angle = np.median(angle)
        max_index = np.argmax(angle)
        min_index = np.argmin(angle)
        if self.num_inputs == 4:
            index_33 = np.argsort(angle)[self.num_frames // 3]
            index_66 = np.argsort(angle)[self.num_frames * 2 // 3]
            frames = np.array([max_index, index_33, index_66, min_index], np.int)
        elif self.num_inputs == 1:
            angle = np.abs(angle)
            front_index = np.argmin(angle)
            frames = np.array([front_index], np.int)
        elif self.num_inputs < 4:
            index_median = np.argsort(angle)[self.num_frames // 2]
            frames = np.array([max_index, index_median, min_index], np.int)
        elif self.num_inputs == 5:
            index_25 = np.argsort(angle)[self.num_frames // 4]
            index_50 = np.argsort(angle)[self.num_frames * 2 // 4]
            index_75 = np.argsort(angle)[self.num_frames * 3 // 4]
            frames = np.array([max_index, index_25, index_50, index_75, min_index], np.int)
        pro_frames = frames
        frames = np.clip(frames, 0, 30)
        print(pro_frames, frames)
        with open(self.log_file_dir, "a") as log_file:
            msg = "the chosen frame index of video %s is" % (vid_name)
            for i in range(frames.shape[0]):
                msg += ",%s" % (img_name_list[frames[i]])
            msg += ".\n"
            log_file.write('%s\n' % msg)  # save the message

        src_texture_im = np.zeros((self.num_inputs, 800, 1200, 3), np.uint8)
        for i in range(self.num_inputs):
            src_texture_im[i] = cv2.imread(text_list[frames[i]])

        src_mask_im = np.zeros((self.num_inputs, 800, 1200), np.uint8)
        for i in range(self.num_inputs):
            src_mask_im[i] = cv2.imread(mask_list[frames[i]])[:, :, 0]

        src_img = np.zeros((self.num_inputs, 256, 256, 3), np.uint8)
        for i in range(self.num_inputs):
            src_img[i] = cv2.imread(img_list[frames[i]])

        tgt_img = np.zeros((self.num_frames, 256, 256, 3), np.uint8)
        for i in range(self.num_frames):
            tgt_img[i] = cv2.imread(img_list[i])

        src_IUV = np.zeros((self.num_inputs, 256, 256, 3), np.uint8)
        for i in range(self.num_inputs):
            src_IUV[i] = cv2.imread(iuv_list[frames[i]])

        tgt_IUV = np.zeros((self.num_frames, 256, 256, 3), np.uint8)
        for i in range(self.num_frames):
            tgt_IUV[i] = cv2.imread(iuv_list[i])

        if self.output_mask == True:

            src_common_area = np.zeros((800, 1200), np.uint8)
            for i in range(self.num_inputs):
                src_common_area = np.logical_or(src_common_area, src_mask_im[i] / 255)

            src_mask_in_image = np.zeros((self.num_inputs, 256, 256, 3), np.uint8)
            for i in range(self.num_inputs):
                src_mask_in_image[i] = TransferTexture(TextureIm=np.ones((800, 1200, 3), np.uint8), IUV=src_IUV[i])

        # normalize to (-1, 1)
        smpl_file = open(smpl_path, 'rb')
        smpl_content = pickle.load(smpl_file)

        smpl_seq = np.concatenate([smpl_content['cams'], smpl_content['pose'], smpl_content['shape']], axis=1)

        smpl_real_mask = np.zeros((self.num_frames, 256, 256, 3), np.uint8)
        for i in range(self.num_frames):
            smpl_real_mask[i] = cv2.imread(real_mask_list[i])

        smpl_real_mask = smpl_real_mask / 255.0

        smpl_data = [smpl_seq, smpl_real_mask, smpl_content['vertices']]

        src_IUV255 = src_IUV
        tgt_IUV255 = tgt_IUV

        src_texture_im = (src_texture_im / 255.0 - 0.5) * 2
        src_IUV = (src_IUV / 255.0 - 0.5) * 2
        tgt_IUV = (tgt_IUV / 255.0 - 0.5) * 2
        src_img = (src_img / 255.0 - 0.5) * 2
        tgt_img = (tgt_img / 255.0 - 0.5) * 2
        src_mask_im = (src_mask_im / 255.0)

        src_data = [src_img, src_IUV, src_texture_im, src_mask_im]
        tgt_data = [tgt_img, tgt_IUV]
        data_255 = [src_IUV255, tgt_IUV255]
        if self.output_mask == True:
            src_data.append(src_common_area)
            src_data.append(src_mask_in_image)

        return src_data, tgt_data, data_255, smpl_data, vid_name, img_name_list, pro_frames

    def __len__(self):
        return self.dataset_len


class Fusion_dataset_smpl_interval(data.Dataset):
    def __init__(self, params, mode='train'):
        super(Fusion_dataset_smpl_interval, self).__init__()

        data_root = params['data_root']
        smpl_root = params['smpl_root']
        mask_root = params['mask_root']
        self.data_dir = os.path.join(data_root, mode)
        self.smpl_dir = os.path.join(smpl_root, mode)
        self.mask_dir = os.path.join(mask_root, mode)

        self.vid_list = get_vid_list(self.data_dir)

        self.batch_size = params['batch_size']
        self.n_iters = params['n_training_iter']
        self.n_sample = params['n_sample']  # get N+1 control points
        self.max_ref_frames = params["maximum_ref_frames"]

        num_vids = len(self.vid_list)
        self.num_frames = params['num_frames']
        self.frame_interval = params['frame_interval']
        use_fix_intv = params['use_fix_interval']
        self.num_inputs = params["maximum_ref_frames"]
        self.num_target = params["num_target"]
        self.fix_frame = params["fix_frame"]
        self.face_GAN = params["face_GAN"]
        self.output_mask = params["output_mask"]
        self.self_recon = params["self_recon"]

        self.dataset_len = len(self.vid_list)
        self.log_file_dir = os.path.join(params['project_dir'], "log_result", "chosen_frame.txt")

    def __getitem__(self, index):
        vid_idx = index

        random_number = np.random.random()

        vid_path = self.vid_list[vid_idx]
        vid_name = vid_path.split("/")[-1]
        smpl_path = os.path.join(self.smpl_dir, vid_name)
        smpl_path = os.path.join(smpl_path, "pose_shape.pkl")
        mask_path = os.path.join(self.mask_dir, vid_name)
        real_mask_list = get_mask_list(mask_path)
        # print(real_mask_list)
        img_list, iuv_list, text_list, mask_list = get_img_iuv_text_mask(vid_path)

        frames = np.random.choice(len(img_list), self.num_inputs + self.num_target, replace=False)

        src_texture_im = np.zeros((self.num_inputs, 800, 1200, 3), np.uint8)
        for i in range(self.num_inputs):
            src_texture_im[i] = cv2.imread(text_list[frames[i + self.num_target]])

        src_mask_im = np.zeros((self.num_inputs, 800, 1200), np.uint8)
        for i in range(self.num_inputs):
            src_mask_im[i] = cv2.imread(mask_list[frames[i + self.num_target]])[:, :, 0]

        src_img = np.zeros((self.num_inputs, 256, 256, 3), np.uint8)
        for i in range(self.num_inputs):
            src_img[i] = cv2.imread(img_list[frames[i + self.num_target]])
        tgt_img = np.zeros((self.num_target, 256, 256, 3), np.uint8)
        for i in range(self.num_target):
            tgt_img[i] = cv2.imread(img_list[frames[i]])

        src_IUV = np.zeros((self.num_inputs, 256, 256, 3), np.uint8)
        for i in range(self.num_inputs):
            src_IUV[i] = cv2.imread(iuv_list[frames[i + self.num_target]])
        tgt_IUV = np.zeros((self.num_target, 256, 256, 3), np.uint8)
        for i in range(self.num_target):
            tgt_IUV[i] = cv2.imread(iuv_list[frames[i]])

        if self.output_mask == True:

            src_common_area = np.zeros((800, 1200), np.uint8)
            for i in range(self.num_inputs):
                src_common_area = np.logical_or(src_common_area, src_mask_im[i] / 255)

            src_area = np.zeros((self.num_target, 256, 256, 3), np.uint8)
            for i in range(self.num_target):
                src_area[i] = TransferTexture(TextureIm=np.repeat(src_common_area[:, :, np.newaxis], 3, axis=2),
                                              IUV=tgt_IUV[i])

            image_inpaint_area = np.zeros((self.num_target, 256, 256, 3), np.uint8)
            tgt_mask_in_image = np.zeros((self.num_target, 256, 256, 3), np.uint8)
            for i in range(self.num_target):
                tgt_mask_in_image[i] = TransferTexture(TextureIm=np.ones((800, 1200, 3), np.uint8), IUV=tgt_IUV[i])
                image_inpaint_area[i] = np.logical_xor(tgt_mask_in_image[i], src_area[i])

            src_mask_in_image = np.zeros((self.num_inputs, 256, 256, 3), np.uint8)
            for i in range(self.num_inputs):
                src_mask_in_image[i] = TransferTexture(TextureIm=np.ones((800, 1200, 3), np.uint8), IUV=src_IUV[i])

        # compute the face Bbox
        # 23 and 24 correspond to the face
        if self.face_GAN == True:
            face_bbox = np.zeros((self.num_target, 4), np.uint8)
            for i in range(self.num_target):
                try:
                    Y1, X1 = np.where(tgt_IUV[i, :, :, 0] == 23)
                    Y2, X2 = np.where(tgt_IUV[i, :, :, 0] == 24)
                    X_con = np.concatenate([X1, X2])
                    Y_con = np.concatenate([Y1, Y2])
                    leftmost = max(np.min(X_con) - 2, 0)
                    rightmost = min(np.max(X_con) + 3, 256)
                    upmost = max(np.min(Y_con) - 2, 0)
                    bottomost = min(np.max(Y_con) + 3, 256)
                    face_bbox[i] = np.array([leftmost, rightmost, upmost, bottomost])
                    # print(face_bbox[i])
                except:
                    face_bbox = np.zeros((self.num_target, 4), np.uint8)
            face_mask = np.zeros((self.num_target, 256, 256), np.uint8)
            for i in range(self.num_target):
                face_mask[i] = np.where(tgt_IUV[i, :, :, 0] == 23, 1, 0)
                face_mask[i] = face_mask[i] + np.where(tgt_IUV[i, :, :, 0] == 24, 1, 0)
        # cv2.imwrite('/home/haolin/test_data/face_mask.jpg',face_mask)

        # canvas=tgt_IUV[0]
        # canvas=cv2.rectangle(canvas, (face_bbox[0,0], face_bbox[0,2]), (face_bbox[0,1], face_bbox[0,3]), (0, 255, 0), 2)
        # cv2.imwrite('/home/haolin/test_data/canvas.jpg',canvas)
        # cv2.imwrite('/home/haolin/test_data/IUV.jpg',tgt_IUV[0,:,:])

        # print(face_mask.shape)

        # normalize to (-1, 1)
        smpl_file = open(smpl_path, 'rb')
        smpl_content = pickle.load(smpl_file)

        smpl_seq_index = frames[:]
        smpl_seq = np.concatenate([smpl_content['cams'][smpl_seq_index], smpl_content['pose'][smpl_seq_index],
                                   smpl_content['shape'][smpl_seq_index]], axis=1)

        smpl_real_mask = cv2.imread(real_mask_list[frames[0]])
        smpl_real_mask = smpl_real_mask / 255.0

        smpl_data = [smpl_seq, smpl_real_mask, smpl_content['vertices'][smpl_seq_index]]

        src_IUV255 = src_IUV
        tgt_IUV255 = tgt_IUV

        src_texture_im = (src_texture_im / 255.0 - 0.5) * 2
        src_IUV = (src_IUV / 255.0 - 0.5) * 2
        tgt_IUV = (tgt_IUV / 255.0 - 0.5) * 2
        src_img = (src_img / 255.0 - 0.5) * 2
        tgt_img = (tgt_img / 255.0 - 0.5) * 2
        src_mask_im = (src_mask_im / 255.0)

        src_data = [src_img, src_IUV, src_texture_im, src_mask_im]
        tgt_data = [tgt_img, tgt_IUV]
        data_255 = [src_IUV255, tgt_IUV255]
        if self.output_mask == True:
            src_data.append(image_inpaint_area)
            src_data.append(src_mask_in_image)
            src_data.append(src_common_area)
        if self.face_GAN == True:
            # prepare return values
            tgt_data.append(face_mask)
            tgt_data.append(face_bbox)

        '''
        for i in range(len(src_data)):
            cv2.imwrite("/data1/haolin/test/"+"src_%i.jpg"%(i),src_data[i])
        for i in range(len(tgt_data)):
            cv2.imwrite("/data1/haolin/test/"+"tgt_%i.jpg"%(i),tgt_data[i])
        '''

        # return {'IUV': tgt_IUV, 'transfered_img': transfered_img, 'y': y, 'tgt_fg_mask': tgt_fg_mask}
        return src_data, tgt_data, data_255, smpl_data

    def __len__(self):
        return self.dataset_len


if __name__ == "__main__":
    opt = get_general_options()
    print("param is ready")
    train_data = Fusion_dataset_smpl(opt)
    print("data_loader is ready")
    src_data, tgt_data, data_255, smpl_data, target_vid_name, img_name_list, frames = train_data.__getitem__(1)
    for i in range(len(smpl_data)):
        print(smpl_data[i].shape)
