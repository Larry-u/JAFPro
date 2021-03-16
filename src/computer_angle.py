import numpy as np
import cv2

def compute_angle(IUV):
    front_index=[2,9,10,13,14] # 2 is frontal body, 9 is upper frontal right leg, 10 is left right leg,
    back_index=[1,7,8,11,12]
    front_area=0
    back_area=0
    for partID in front_index:
        part=np.where(IUV[:,:,0]==partID,np.ones(IUV[:,:,0].shape),np.zeros(IUV[:,:,0].shape))
        if partID==2:
            y,x_frontal=np.where(part==1)
            frontal_avg_x=np.average(x_frontal)
        area=np.sum(part)
        front_area+=area 
    for partID in back_index:
        part = np.where(IUV[:, :, 0] == partID, np.ones(IUV[:, :, 0].shape), np.zeros(IUV[:, :, 0].shape))
        if partID==1:
            y,x_back = np.where(part == 1)
            if x_back.shape[0]>0:
                back_avg_x=np.average(x_back)
            else:
                back_avg_x=frontal_avg_x
            if x_frontal.shape[0]==0:
                frontal_avg_x=back_avg_x
        area=np.sum(part)
        back_area+=area
    if frontal_avg_x<back_avg_x:
        front_back_ratio = (front_area + 10e-5) / (back_area + 10e-5)
        angle = np.arctan(front_back_ratio)
        angle = angle / np.pi * 180-90
    else:
        front_back_ratio = - (front_area+10e-5) / (back_area + 10e-5)
        angle = np.arctan(front_back_ratio)
        angle=angle/np.pi*180+90

    if angle<-65:
        return 65
    return angle

if __name__=="__main__":
    IUV=cv2.imread("pipeline_tgt_2_IUV.png")
    compute_angle(IUV)
