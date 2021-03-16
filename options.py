import os


def get_general_options():
    opt = {}

    opt['n_sample'] = 6  # get N+1 points
    opt['num_frames'] = 30
    opt['frame_interval'] = 5
    opt['use_fix_interval'] = False

    user = os.getlogin()
    opt['uname'] = user
    opt['resume_train'] = False
    opt['n_training_iter'] = 200001
    opt['test_interval'] = 1000
    opt['validate_interval'] = 10000
    opt['vis_interval'] = 10000
    opt['gan_test_interval'] = 10
    opt['model_save_interval'] = 200
    opt["test_num_inputs"] = 1
    opt["num_outputs"] = 2
    opt["num_target"] = 3
    opt["fix_frame"] = True
    opt["self_recon"] = False
    opt['data_aug'] = False

    opt['project_dir'] = '/data3/haolin/JAFPro'
    opt['model_save_dir'] = opt['project_dir'] + '/checkpoints'
    opt["test_save_dir"] = '/home/Larryu/Projects/JAFPro_minimum' + '/test_results'
    opt['flownet_path'] = '/data3/haolin/JAFPro/flownet2_pytorch/FlowNet2-SD_checkpoint.pth.tar'

    # data related configs
    opt['smpl_root'] = '/data3/haolin/dataset/dance_dataset_smpl'
    opt['mask_root'] = '/data3/haolin/dataset/dance_dataset_mask'
    opt['data_root'] = '/data3/haolin/dataset/dance_dataset_split_0622'

    opt['isTrain'] = True
    opt['num_ref_frames'] = 3
    opt['batch_size'] = 4
    opt["maximum_ref_frames"] = 3
    opt["face_GAN"] = True
    opt["output_mask"] = True
    return opt
