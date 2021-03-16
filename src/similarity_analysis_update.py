from skimage import measure
from PIL import Image
import numpy as np
import os

# Paras #
gtPkgName = '/data3/haolin/dance_dataset_split/test'
#predPkgName = r'F:\LMD-test\warper-vid-test' #1
predPkgName = "/data1/haolin/patch-transfer-letmedance/checkpoints/fusion_test_3fr_0302" #2
#predPkgName = r'F:\LMD-test\warp-updater-2fr-ConvLSTM-3inputs-vid-test' #3
#predPkgName = r'F:\LMD-test\updater-one' #4
frameNumber = 30
#Get GT/pred package
gtPkgFileList = os.listdir(gtPkgName)
predPkgFileList = os.listdir(predPkgName)
gtPkgList=[]
predPkgList=[]

for file in gtPkgFileList:
    gtPkgList.append(file)

for file in predPkgFileList:
    predPkgList.append(file)
    
print(gtPkgList)
print(predPkgList)

#Calculate PSNR, SSIM and L1-norm
overAllPSNR = 0
overAllSSIM = 0
overAllL1Norm = 0
overAllPSNRList = []
overAllSSIMList = []
overAllL1NormList = []
print(len(gtPkgList))
for i in range(0,int(len(gtPkgList))):
    #Check clipId
    #continue
    #Start iteration for one clip
    gtClipList=[]
    predClipList=[]
    video_name=predPkgList[i]
    gtClipFileList=os.listdir(os.path.join(gtPkgName,video_name))
    predClipFileList=os.listdir(os.path.join(predPkgName,video_name))
    for file in gtClipFileList:
      if file.find("jpg")>0:
        gtClipList.append(file)
    for file in predClipFileList:
      if (file.find("jpg")>0): #and (file.find("pred")>=0):
        predClipList.append(file)
    gtClipList.sort(key=lambda x:int(x[6:-4]))
    predClipList.sort(key=lambda x:int(x[6:-4]))
    psnrAllPic = 0
    ssimAllPic = 0
    l1normAllPic = 0
    for j in range(0,frameNumber):
        #Calculation
        gtPicDir = os.path.join(gtPkgName,video_name,gtClipList[j])
        #predPicDir = os.path.join(predPkgName, predPkgList[i], 'pred_frame_' + str(j) + '.png') #1,4
        predPicDir = os.path.join(predPkgName,video_name,predClipList[j]) #2,3
        gtImg = np.array(Image.open(gtPicDir), 'f')
        predImg = np.array(Image.open(predPicDir), 'f')
        dmin = np.min(gtImg)
        dmax = np.max(gtImg)
        psnrPic = measure.compare_psnr(gtImg,predImg,dmax-dmin)
        ssimPic = measure.compare_ssim(gtImg,predImg,win_size=11,data_range=dmax-dmin,multichannel=True)
        l1normPic = np.linalg.norm((((gtImg-predImg).reshape((3*256*256,1,1)))/255).squeeze(), ord=1)/256/256
        psnrAllPic += psnrPic
        ssimAllPic += ssimPic
        l1normAllPic += l1normPic
    overAllPSNRList.append(psnrAllPic/frameNumber)
    overAllSSIMList.append(ssimAllPic/frameNumber)
    overAllL1NormList.append(l1normAllPic/frameNumber)
    overAllPSNR += psnrAllPic
    overAllSSIM += ssimAllPic
    overAllL1Norm += l1normAllPic
    print('The Mean PSNR for clip ' + video_name + ' is ' + str(psnrAllPic/frameNumber) + '.')
    print('The Mean SSIM for clip ' + video_name + ' is ' + str(ssimAllPic/frameNumber) + '.')
    print('The Mean L1-norm for clip ' + video_name + ' is ' + str(l1normAllPic / frameNumber) + '.')

allPSNRArray = np.array(overAllPSNRList)
allSSIMArray = np.array(overAllSSIMList)
allL1NormArray = np.array(overAllL1NormList)
print('The Overall Mean PSNR is ' + str(overAllPSNR/frameNumber/int(len(gtPkgList))) + '.')
print('The Overall std PSNR is ' + str(allPSNRArray.std()) + '.')
print('The Overall Mean SSIM is ' + str(overAllSSIM/frameNumber/int(len(predPkgList))) + '.')
print('The Overall std SSIM is ' + str(allSSIMArray.std()) + '.')
print('The Overall Mean L1-norm is ' + str(overAllL1Norm/frameNumber/int(len(predPkgList))) + '.')
print('The Overall std L1-norm is ' + str(allL1NormArray.std()) + '.')
