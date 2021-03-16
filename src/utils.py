import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from moviepy.editor import ImageSequenceClip


def get_vid_list(data_dir):
    vid_list = []
    for vid in os.listdir(data_dir):
        vid_path = os.path.join(data_dir, vid)
        if os.path.isdir(vid_path):
            vid_list.append(vid_path)

    return vid_list


def get_img_and_iuv_list(vid_path):
    img_list = sorted(glob.glob("%s/*.jpg" % vid_path))
    iuv_list = sorted(glob.glob("%s/*.png" % vid_path))

    return img_list, iuv_list
def get_mask_list(vid_path):
    file_list=os.listdir(vid_path)
    mask_list=[]
    for file in file_list:
        if file.find("png"):
            mask_list.append(file)
    #print(mask_list)
    mask_list.sort(key=lambda x:int(x[6:-9]))
    for i,mask_file in enumerate(mask_list):
        mask_list[i]=os.path.join(vid_path,mask_file)
    return mask_list
    
def get_img_iuv_text_mask(vid_path):
    file_list=os.listdir(vid_path)
    img_list=list()
    iuv_list=list()
    text_list=list()
    mask_list=list()
    for file in file_list:
        if file.find("IUV")<0 and file.find("mask")<0 and file.find("text")<0 and file.find("bbox")<0 and file.find("pkl")<0:
            img_list.append(os.path.join(vid_path,file))
        if file.find("IUV")>0:
            iuv_list.append(os.path.join(vid_path,file))
        if file.find("mask")>0:
            mask_list.append(os.path.join(vid_path,file))
        if file.find("text")>0:
            text_list.append(os.path.join(vid_path,file))
    #print(img_list)
    img_list.sort(key=lambda x:int(x.split("/")[-1][6:-4]))
    iuv_list.sort(key=lambda x:int(x.split("/")[-1][6:-8]))
    mask_list.sort(key=lambda x:int(x.split("/")[-1][6:-9]))
    text_list.sort(key=lambda x:int(x.split("/")[-1][6:-9]))
    return img_list,iuv_list,text_list,mask_list
    
def get_img_iuv_text_mask_fashion(vid_path):
    file_list=os.listdir(vid_path)
    img_list=list()
    iuv_list=list()
    text_list=list()
    mask_list=list()
    for file in file_list:
        if file.find("jpg")>0:
            img_list.append(file)
        if file.find("IUV")>0:
            iuv_list.append(file)
        if file.find("mask")>0:
            mask_list.append(file)
        if file.find("text")>0:
            text_list.append(file)
    img_list.sort(key=lambda x:x[0:4])
    iuv_list.sort(key=lambda x:x[0:4])
    mask_list.sort(key=lambda x:x[0:4])
    text_list.sort(key=lambda x:x[0:4])
    for i in range(len(img_list)):
        img_list[i]=os.path.join(vid_path,img_list[i])
        iuv_list[i]=os.path.join(vid_path,iuv_list[i])
        mask_list[i]=os.path.join(vid_path,mask_list[i])
        text_list[i]=os.path.join(vid_path,text_list[i])
    return img_list,iuv_list,text_list,mask_list
    
#print(get_img_iuv_text_mask_fashion("/data3/haolin/deepFashion_256/1"))


def get_vid_list_pred(data_dir):
    vid_list = []
    for vid in os.listdir(data_dir):
        vid_path = os.path.join(data_dir, vid)
        vid_list.append(vid_path)

    return sorted(vid_list)


def get_vid_list_gt(data_dir):
    vid_list = []
    for dc in os.listdir(data_dir):
        dc_path = os.path.join(data_dir, dc)
        if os.path.isdir(dc_path):
            for vid in os.listdir(dc_path):
                vid_path = os.path.join(dc_path, vid)
                vid_list.append(vid_path)

    return sorted(vid_list)


def get_pred_img_list(vid_path):
    # return sorted(glob.glob("%s/pred_frame_withbg_*.png" % vid_path))
    # return sorted(glob.glob("%s/pred_frame_*.png" % vid_path))
    return sorted(glob.glob("%s/pred_frame_*.jpg" % vid_path))


def get_gt_img_list(vid_path):
    return sorted(glob.glob("%s/*.jpg" % vid_path))


def vgg_preprocess(x):
    x = 255.0 * (x + 1.0) / 2.0

    x[:, 0, :, :] -= 103.939
    x[:, 1, :, :] -= 116.779
    x[:, 2, :, :] -= 123.68

    return x


def get_bg(src_img, src_Imap, tgt_Imap):
    bg_mask = np.zeros(src_img.shape, dtype=np.uint8)
    bg_mask[src_Imap == 0] = 1
    src_bg = src_img * bg_mask

    bg_mask = np.zeros(src_img.shape, dtype=np.uint8)
    bg_mask[tgt_Imap == 0] = 1
    tgt_bg = src_bg * bg_mask

    return np.uint8(tgt_bg)


def get_patch(img, Imap, partId):
    region = np.where(Imap == partId)
    npix = region[0].shape[0]
    # print("patch has {} pixels".format(region[0].shape[0]))
    patch_mask = np.zeros(img.shape, dtype=np.uint8)
    patch_mask[region] = 1
    patch = img * patch_mask
    rgb_mask = np.transpose(np.tile(patch_mask, (3, 1, 1)), [1, 2, 0])

    return patch, rgb_mask, npix


def get_ctl_points(img, n_sample, mode):
    # try:
    if mode == 'cv2':
        _, cnt, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnt = np.array(cnt).squeeze()
        # print("patch has %d points" % (cnt.shape[0]))
    elif mode == 'plt':
        cnt = plt.contour(img, 1)
        cnt = cnt.allsegs[0][0]
        # print("patch has %d points" % (cnt.shape[0]))
    else:
        raise NotImplementedError

    if cnt.shape[0] < n_sample:
        return None
    else:
        inter = int(cnt.shape[0] / n_sample)

        return cnt[::inter]

    # except:
    #     pass


def patch_transfer(src_img, src_Imap, tgt_Imap, n_sample):
    success = 0
    fail = 0

    tgt_bg = get_bg(src_img, src_Imap, tgt_Imap)
    tgt_img_stack = []
    for partId in range(1, 25):
        try:
            # print("=" * 88)
            # construct patches
            src_patch, src_mask, src_npix = get_patch(src_img, src_Imap, partId)
            src_Ipatch, _, _ = get_patch(src_Imap, src_Imap, partId)
            tgt_Ipatch, tgt_mask, tgt_npix = get_patch(tgt_Imap, tgt_Imap, partId)

            if src_npix == 0 or tgt_npix == 0:
                # print("no pixel, continued")
                fail += 1
                continue

            # get control points
            # print("source {}".format(partId))
            src_ctl_points = get_ctl_points(src_Ipatch, n_sample, 'plt')
            # print("target {}".format(partId))
            tgt_ctl_points = get_ctl_points(tgt_Ipatch, n_sample, 'plt')

            if src_ctl_points is None or tgt_ctl_points is None:
                # print("no ctl points, continued")
                fail += 1
                continue
            # print("src has %d points, tgt has %d points" % (src_ctl_points.shape[0], tgt_ctl_points.shape[0]))

            assert len(src_ctl_points) == len(tgt_ctl_points), "number of control points must be equal"

            matches = []
            for i in range(len(src_ctl_points)):
                matches.append(cv2.DMatch(i, i, 0))

            tps = cv2.createThinPlateSplineShapeTransformer()
            tps.estimateTransformation(tgt_ctl_points.reshape(1, -1, 2), src_ctl_points.reshape(1, -1, 2), matches)

            # tps warp
            tgt_patch = tps.warpImage(src_patch) * tgt_mask
            # tgt_patch = tps.warpImage(src_patch)
            tgt_img_stack.append(np.uint8(tgt_patch))
            success += 1
        except Exception as e:
            # print(e)
            # print("continued")
            fail += 1
            continue
    print("success %d, fail %d" % (success, fail))
    return tgt_img_stack


def get_texture(im, IUV, tex_size=32, final_size=200):
    solution_float = float(tex_size) - 1

    U = IUV[:, :, 1]
    V = IUV[:, :, 2]
    parts = list()
    for PartInd in range(1, 25):  ## Set to xrange(1,23) to ignore the face part.
        actual_part = np.zeros((tex_size, tex_size, 3))
        x, y = np.where(IUV[:, :, 0] == PartInd)
        if len(x) == 0:
            parts.append(cv2.resize(actual_part, (final_size, final_size), cv2.INTER_LINEAR))
            continue

        u_current_points = U[x, y]  # Pixels that belong to this specific part.
        v_current_points = V[x, y]
        ##
        tex_map_coords = ((255 - v_current_points) * solution_float / 255.).astype(int), (
                u_current_points * solution_float / 255.).astype(int)
        for c in range(3):
            actual_part[tex_map_coords[0], tex_map_coords[1], c] = im[x, y, c]

        parts.append(cv2.resize(actual_part, (final_size, final_size), cv2.INTER_LINEAR)[:, :, ::-1] / 255.)

    return parts


def texture_warp(tex_parts, IUV):
    U = IUV[:, :, 1]
    V = IUV[:, :, 2]
    #
    R_im = np.zeros(U.shape)
    G_im = np.zeros(U.shape)
    B_im = np.zeros(U.shape)
    ###
    for PartInd in range(1, 25):  ## Set to range(1,23) to ignore the face part.
        tex = tex_parts[PartInd - 1]  # get texture for each part.
        #####
        R = tex[:, :, 0]
        G = tex[:, :, 1]
        B = tex[:, :, 2]
        ###############
        x, y = np.where(IUV[:, :, 0] == PartInd)
        u_current_points = U[x, y]  # Pixels that belong to this specific part.
        v_current_points = V[x, y]
        ##

        r_current_points = R[
            ((255 - v_current_points) * 199. / 255.).astype(int), (u_current_points * 199. / 255.).astype(int)]
        g_current_points = G[
            ((255 - v_current_points) * 199. / 255.).astype(int), (u_current_points * 199. / 255.).astype(int)]
        b_current_points = B[
            ((255 - v_current_points) * 199. / 255.).astype(int), (u_current_points * 199. / 255.).astype(int)]
        ##  Get the RGB values from the texture images.
        R_im[IUV[:, :, 0] == PartInd] = r_current_points
        G_im[IUV[:, :, 0] == PartInd] = g_current_points
        B_im[IUV[:, :, 0] == PartInd] = b_current_points
    generated_image = np.concatenate((B_im[:, :, np.newaxis], G_im[:, :, np.newaxis], R_im[:, :, np.newaxis]), axis=2)

    return generated_image


def uv_transfer(src_img, src_IUV, tgt_IUV):
    tex_parts = get_texture(src_img, src_IUV, tex_size=24)
    transfered_im = texture_warp(tex_parts, tgt_IUV)

    return transfered_im


# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x


class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            # scipy.misc.toimage(img).save(s, format="png")
            plt.imsave(s, img, format='png')

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

def TransferTexture(TextureIm,IUV,im=None): 
    #input format
    #TextureIm (1200,800,3) 0~255, 
    # IUV (256,256,3) 0~255, 
    # im (256,256,3) 0~255, if im is None, this function would consider im as background, and fuse the result with the background
    output_img=np.zeros((256,256,3),np.uint8)
    U = np.rint(IUV[:,:,1]/255.*199.).astype(np.uint8)
    V = np.rint(IUV[:,:,2]/255.*199.).astype(np.uint8)
    for partId in range(1,25):
        i_cor=(partId-1)//6
        j_cor=partId-i_cor*6-1
        tex=TextureIm[i_cor*200:i_cor*200+200,j_cor*200:j_cor*200+200,:]
        x,y = np.where(IUV[:,:,0]==partId)
        u_current_points = U[x,y]   #  Pixels that belong to this specific part.
        v_current_points = V[x,y]
        tex_map_coords = (u_current_points,199-v_current_points)
        '''
        for i in range(len(x)):
            output_img[x,y,:]=tex[tex_map_coords[0],tex_map_coords[1],:]
        '''
        output_img[x,y,:]=tex[tex_map_coords[0],tex_map_coords[1],:]
    if im is not None:
        BG_MASK = output_img==0
        output_img[BG_MASK] = im[BG_MASK]  ## Set the BG as the old image.
    #output format (256,256,3)0~255
    return output_img

def Texture_fusion(Texture1,Texture2,Mask1,Mask2):
    #input format
    #Texture1, Texture2 (1200,800,3) 0~255, 
    #Mask1, Mask2 (256,256,1) 0~255, 
    output_text=np.zeros(Texture1.shape)
    output_mask=np.zeros(Mask1.shape)

    Mask1=(Mask1/255).astype(np.uint8)
    Mask2=(Mask2/255).astype(np.uint8)

    intersection=np.zeros(Mask1.shape)
    np.logical_and(Mask1,Mask2,out=intersection)
    radius=7
    kernel = np.ones((radius,radius), np.uint8)
    dilated_intersection=cv2.dilate(intersection,kernel).astype(np.uint8)
    non_overlap=np.subtract(Mask2,dilated_intersection,dtype=np.uint8)

    non_overlap_mask=np.repeat(non_overlap[:,:,np.newaxis],3,2)
    complement_texture=np.multiply(non_overlap_mask,Texture2,dtype=np.uint8)

    output_text=complement_texture+Texture1

    output_mask=Mask1+np.multiply(non_overlap,Mask2,dtype=np.uint8)

    inpaint_area=np.subtract(1,output_mask,dtype=np.uint8)

    '''
    cv2.imshow("mask2",cv2.resize(Mask2*255,(200*3,200*2)))
    cv2.imshow("mask1",cv2.resize(Mask1*255,(200*3,200*2)))
    cv2.imshow("intersection",cv2.resize(output_text,(200*3,200*2)))
    cv2.imshow("output_mask",cv2.resize(inpaint_area*255,(200*3,200*2)))
    '''
    #output format
    #output_text (1200,800,3) 0~255,
    #inpaint_area (1200,800,1) 0~255,
    #output_mask (1200,800,1) 0~255
    return output_text,(output_mask*255).astype(np.uint8),(inpaint_area*255).astype(np.uint8)

def gif(filename, array, fps=10, scale=1.0):
    """Creates a gif given a stack of images using moviepy
    Notes
    -----
    works with current Github version of moviepy (not the pip version)
    https://github.com/Zulko/moviepy/commit/d4c9c37bc88261d8ed8b5d9b7c317d13b2cdf62e
    Usage
    -----
    >>> X = randn(100, 64, 64)
    >>> gif('test.gif', X)
    Parameters
    ----------
    filename : string
        The filename of the gif to write to
    array : array_like
        A numpy array that contains a sequence of images
    fps : int
        frames per second (default: 10)
    scale : float
        how much to rescale each image by (default: 1.0)
    """

    # ensure that the file has the .gif extension
    fname, _ = os.path.splitext(filename)
    filename = fname + '.gif'

    # copy into the color dimension if the images are black and white
    if array.ndim == 3:
        array = array[..., np.newaxis] * np.ones(3)

    # make the moviepy clip
    clip = ImageSequenceClip(list(array), fps=fps).resize(scale)
    clip.write_gif(filename, fps=fps)
    return clip

