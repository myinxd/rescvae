# copyright (C) 2017 zxma_sjtu@qq.com

"""
The utilities for image preprocessing
"""

import os
import pickle
import numpy as np
import tensorflow as tf
import scipy.io as sio
from astropy.io import fits
from scipy.misc import imread
from skimage import transform
from astropy.stats import sigma_clip


def get_sigma_clip(img,sigma=3,iters=100):
    """
    Do sigma clipping on the raw images to improve constrast of
    target regions.

    Reference
    =========
    [1] sigma clip
        http://docs.astropy.org/en/stable/api/astropy.stats.sigma_clip.html
    """
    img_clip = sigma_clip(img, sigma=sigma, iters=iters)
    img_mask = img_clip.mask.astype(float)
    img_new = img * img_mask

    return img_new


def get_augmentation(img, crop_box=(150,150), rez_box=(50,50),
                     num_aug = 1, clipflag=False,clipparam=None):
    """
    Do image augmentation

    References
    ==========
    [1] http://blog.sina.com.cn/s/blog_5562b04401015bys.html
    [2] http://blog.csdn.net/guduruyu/article/details/70842142

    steps
    =====
    flip -> rotate -> crop -> resize

    inputs
    ======
    img: np.ndarray or str
        image or the image path
    crop_box: tuple
        Size of the crop box
    rez_box: tuple
        Size of the resized box

    output
    ======
    img_aug: augmented image
    """
    from PIL import Image
    from astropy.io import fits
    # load image
    if isinstance(img, str):
        if img.split(".")[-1] == "fits":
            h = fits.open(img)
            h = h[0].data
            h = np.nan_to_num(h)
            h = (h-h.min())/(h.max()-h.min())
            img_raw = Image.fromarray(h)
        else:
            img_raw = Image.open(img)
            img_raw = img_raw.convert('L')
    else:
        img_raw = Image.fromarray(img)

    # sigma clipping
    if clipflag == True:
        img_raw = get_sigma_clip(np.array(img_raw),
                                 sigma=clipparam[0],
                                 iters=clipparam[1])
        img_raw = Image.fromarray(img_raw)
    # rbg2grey
    img_r = np.zeros((num_aug, rez_box[0], rez_box[1]))
    for i in range(num_aug):
        # flip
        idx = np.random.permutation(2)[0]
        if idx == 0:
            img_aug = img_raw.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            img_aug = img_raw.transpose(Image.FLIP_TOP_BOTTOM)

        # rotate
        angle = np.random.uniform() * 360
        img_aug = img_aug.rotate(angle, expand=True)

        # crop
        rows = img_aug.width
        cols = img_aug.height
        row_cnt = int(np.round(rows/2))
        col_cnt = int(np.round(cols/2))
        row_crop_half = int(np.round(crop_box[0]/2))
        col_crop_half = int(np.round(crop_box[1]/2))
        crop_tuple = (row_cnt-row_crop_half,
                      col_cnt-col_crop_half,
                      row_cnt+row_crop_half,
                      col_cnt+col_crop_half)
        img_aug = img_aug.crop(box=crop_tuple)

        # resize
        img_aug = img_aug.resize(rez_box)

        # Image to matrix
        img_r[i,:,:] = np.array(img_aug)

    return img_r


def get_augmentation_single(img, crop_box=(150,150), rez_box=(50,50),
                     num_aug = 1, clipflag=False,clipparam=None):
    """
    Do image augmentation, only clipping

    References
    ==========
    [1] http://blog.sina.com.cn/s/blog_5562b04401015bys.html
    [2] http://blog.csdn.net/guduruyu/article/details/70842142

    steps
    =====
    flip -> rotate -> crop -> resize

    inputs
    ======
    img: np.ndarray or str
        image or the image path
    crop_box: tuple
        Size of the crop box
    rez_box: tuple
        Size of the resized box

    output
    ======
    img_aug: augmented image
    """
    from PIL import Image
    from astropy.io import fits
    # load image
    if isinstance(img, str):
        if img.split(".")[-1] == "fits":
            h = fits.open(img)
            h = h[0].data
            h = np.nan_to_num(h)
            h = (h-h.min())/(h.max()-h.min())
            img_raw = Image.fromarray(h)
        else:
            img_raw = Image.open(img)
            img_raw = img_raw.convert('L')
    else:
        img_raw = Image.fromarray(img)
        # img_raw = img_raw.convert('L')

    # sigma clipping
    if clipflag == True:
        img_raw = get_sigma_clip(np.array(img_raw),
                                 sigma=clipparam[0],
                                 iters=clipparam[1])
        img_raw = Image.fromarray(img_raw)
    # rbg2grey
    img_r = np.zeros((num_aug, rez_box[0], rez_box[1]))
    for i in range(num_aug):
        # crop
        rows = img_raw.width
        cols = img_raw.height
        row_cnt = int(np.round(rows/2))
        col_cnt = int(np.round(cols/2))
        row_crop_half = int(np.round(crop_box[0]/2))
        col_crop_half = int(np.round(crop_box[1]/2))
        crop_tuple = (row_cnt-row_crop_half,
                      col_cnt-col_crop_half,
                      row_cnt+row_crop_half,
                      col_cnt+col_crop_half)
        img_aug = img_raw.crop(box=crop_tuple)

        # resize
        img_aug = img_aug.resize(rez_box)

        # Image to matrix
        img_r[i,:,:] = np.array(img_aug)

    return img_r


def gen_sample_augmentation(folder, ftype='jpg', savepath=None,
                            crop_box=(200, 200), res_box=(50, 50),
                            num_aug=1, clipflag=False, clipparam=None):
    """
    Read the sample images and reshape to required structure

    input
    =====
    folder: str
        Name of the folder, i.e., the path
    ftype: str
        Type of the images, default as 'jpg'
    savepath: str
        Path to save the reshaped sample mat
        default as None
    crop_box: tuple
        Boxsize of the cropping of the center region
    res_box: tuple
        Scale of the resized image
    num_aug: integer
        Number of augmentated images of each sample
    clipflag: booling
        The flag of sigma clipping, default as False
    clipparam: list
        Parameters of the sigma clipping, [sigma, iters]

    output
    ======
    sample_mat: np.ndarray
        The sample matrix
    """
    # Init
    if os.path.exists(folder):
        sample_list = os.listdir(folder)
    else:
        return

    sample_mat = np.zeros((len(sample_list)*num_aug,
                           res_box[0]*res_box[1]))

    def read_image(fpath,ftype):
        if ftype == 'fits':
            h = fits.open(fpath)
            img = h[0].data
        else:
            img = imread(name=fpath, flatten=True)
        return img

    # load images
    sess = tf.InteractiveSession()
    idx = 0
    for fname in sample_list:
        fpath = os.path.join(folder,fname)
        if fpath.split('.')[-1] == ftype:
            #read image
            img = read_image(fpath=fpath, ftype=ftype)
            # augmentation
            for i in range(num_aug):
                img_rsz = get_augmentation(img=img,
                                           crop_box=crop_box,
                                           res_box=res_box,
                                           sess = sess)
                # push into sample_mat
                img_vec = img_rsz.reshape((res_box[0]*res_box[1],))
                sample_mat[idx+i,:] = img_vec
            idx = idx + num_aug
        else:
            continue

    # save
    if not savepath is None:
        stype = savepath.split('.')[-1]
        if stype == 'mat':
            # save as mat
            sample_dict = {'data':sample_mat,
                           'name':sample_list}
            sio.savemat(savepath,sample_dict)
        elif stype == 'pkl':
            fp = open(savepath,'wb')
            sample_dict = {'data':sample_mat,
                           'name':sample_list}
            pickle.dump(sample_dict,fp)
            fp.close()

    return sample_mat


def load_sample(samplepath):
    """Load the sample matrix

    input
    =====
    samplepath: str
        Path to save the samples
    """
    ftype = samplepath.split('.')[-1]
    if ftype == 'pkl':
        try:
            fp = open(samplepath, 'rb')
        except:
            return None
        sample_dict = pickle.load(fp)
        sample_mat = sample_dict['data']
        sample_list = sample_dict['name']
    elif ftype == 'mat':
        try:
            sample_dict = sio.loadmat(samplepath)
        except:
            return None
        sample_mat = sample_dict['data']
        sample_list = sample_dict['name']

    return sample_mat, sample_list
