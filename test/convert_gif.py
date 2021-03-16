import os,sys
sys.path.append("..")
from src.utils import gif,get_gt_img_list
import cv2
import numpy as np
import colored_traceback
import argparse



if __name__ =="__main__":

    colored_traceback.add_hook()
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', type=str, required=True, help='pred dir')

    args = parser.parse_args()

    pred_dir=args.pred_dir
    project_name=pred_dir.split("/")[-1]
    save_dir=os.path.join("../gif_result",project_name)
    
    video_list=os.listdir(pred_dir)
    angle_log_file="../video_angle.txt"
    
    large_angle_list=[]
    f=open(angle_log_file,'r')
    line=f.readline()
    while line:
        video_name=line.split(" ")[1]
        #print(video_name)
        large_angle_list.append(video_name)
        line=f.readline()
    #print(large_angle_list)
    for video in video_list:
        #if video not in large_angle_list:
        #    continue
        video_path=os.path.join(pred_dir,video)
        save_video_path=os.path.join(save_dir,video+"_video")
        if os.path.exists(save_video_path)==False:
            os.makedirs(save_video_path)
        img_list=[]
        file_list=os.listdir(video_path)
        for file in file_list:
            if file.find("mask")<0 and file.find("coarse")<0 and file.find("tsf") and file.find("IUV")<0 and file.find("text")<0:
                img_list.append(os.path.join(video_path,file))
        img_list.sort(key=lambda x:int(x.split("/")[-1][6:-4]))
        concat_images=np.zeros((30,256,256,3),np.uint8)
        for frames in range(0,len(img_list)):
            concat_images[frames]=cv2.imread(img_list[frames])[:,:,::-1]
        gif(save_video_path+"/video.gif",concat_images)
        
        
