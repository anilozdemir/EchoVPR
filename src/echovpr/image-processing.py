#!/usr/bin/env python
# coding: utf-8
# Anil Ozdemir
# Apply NetVLAD to set of images (run once)


import numpy as np
import cv2
import os 
import argparse
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Apply NetVLAD through set of images')

parser.add_argument('--dataset'     , type=str ,  default='Nordland', metavar='d', help='dataset name')
parser.add_argument('--mode'        , type=str ,  default='ref'     , metavar='m', help='mode, either ref or query')
parser.add_argument('--nImage'      , type=int ,  default=100       , metavar='n', help='number of images in the path to process')
parser.add_argument('--width'       , type=int ,  default=640       , metavar='w', help='width of images to resize')
parser.add_argument('--height'      , type=int ,  default=480       , metavar='H', help='height of images to resize')
parser.add_argument('--saveNumpy'   , type=bool,  default=False     , metavar='s', help='save Numpy?')
parser.add_argument('--saveTorchESN', type=bool,  default=False     , metavar='t', help='save Torch for ESN?')

args = parser.parse_args()

modulePath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
path = modulePath + '/datasets/0.rgb_images/' + args.dataset + '/' + args.mode

if not os.path.exists(modulePath):
    raise Exception("The module path does not exist...") 
    exit()

if not os.path.exists(path):
    raise Exception("The dataset does not exist...") 
    exit()

# if does not exist; create the following directories:
saveNumpy    = modulePath + '/datasets/1.numpy_images/'  + args.dataset + '/'
saveTorch    = modulePath + '/datasets/2.torch_netVLAD/' + args.dataset + '/' 
saveTorchESN = modulePath + '/datasets/3.torch_ESN/'     + args.dataset + '/' 

for directory in [saveNumpy, saveTorch, saveTorchESN]:
    if not os.path.exists(directory):
        os.makedirs(directory)
    
## Dataset Operations

def transformImages(imgDir: str, nImg: int, img_w: int, img_h: int):
    '''
    transforms the images to grayscale and downsamples using OpenCV, and normalises to [0,1].
    Args:
        imgDir (str): image directory
        nImg (int): number of images to process
        img_w (int): image width
        img_h (int): image height
    Returns:
        X (np.array): transformed set of images in imgDir.
    '''
    img_c = 3    # img_c number of channels
    X = np.zeros((nImg, img_h, img_w, img_c))
    Images = sorted(os.listdir(imgDir))[:nImg]
#     print(len(Images), len(os.listdir(imgDir)))
    for i, im in enumerate(tqdm(Images)):
        img  = cv2.imread(imgDir+'/'+im)
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # is it necessary for NetVLAD?
        img  = cv2.resize(img, (img_w, img_h),interpolation = cv2.INTER_AREA)
        X[i] = img.reshape(img_h, img_w, img_c)
    return X # do not scale here!

Imgs = transformImages(path, args.nImage, args.width, args.height)

if args.saveNumpy:
    np.save(saveNumpy + args.mode + '.npy', Imgs/255) # scale the image when saving
    print('>> dataset was saved to '+ saveNumpy + args.mode + '.npy')

# if dims of Imgs is not 4 (no batch-dim), add it
if len(Imgs.shape) == 3:
    Imgs = np.expand_dims(Imgs, axis=0)

## Restore Pre-trained NetVLAD
from netVLAD import nets
from netVLAD import net_from_mat  as nfm
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
tf.reset_default_graph()

image_batch = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 3])
net_out     = nets.vgg16NetvladPca(image_batch)
saver       = tf.train.Saver()


if len(Imgs) > 100:
    resultList = []
    for i, k in enumerate(range(0,len(Imgs),100)):
        img_batch = Imgs[k:k+100] # add at most +100, if less, that's fine
        sess = tf.Session()
        saver.restore(sess, nets.defaultCheckpoint())
    
        # Generate TF results
        result = sess.run(net_out, feed_dict={image_batch: img_batch})
        print('\t [IMAGE-PROCESSING MESSAGE: ] >> batch: {}/{} processed {} images to vector of length {}'.format(i+1, len(Imgs)//100, *result.shape))
        resultList.append(result)
    Result = np.vstack(resultList)
else:   
    sess = tf.Session()
    saver.restore(sess, nets.defaultCheckpoint())
    
    # Generate TF results
    Result = sess.run(net_out, feed_dict={image_batch: Imgs})
    print('\t [IMAGE-PROCESSING MESSAGE: ] >> processed {} images to vector of length {}'.format(*Result.shape))

torch.save(torch.Tensor(Result), saveTorch + args.mode + '.pt')
print('>> dataset was saved to '+ saveTorch + args.mode + '.pt')

### Transform for ESN: gray-scale and flatten

def transformImagesESN(imgDir: str, nImg: int, img_w: int, img_h: int):
    '''
    transforms the images to grayscale and downsamples using OpenCV, and normalises to [0,1].
    Args:
        imgDir (str): image directory
        nImg (int): number of images to process
        img_w (int): image width
        img_h (int): image height
    Returns:
        X (np.array): transformed set of images in imgDir.
    '''
    X = np.zeros((nImg, img_h*img_w))
    Images = sorted(os.listdir(imgDir))[:nImg]
    for i, im in enumerate(tqdm(Images)):
        img  = cv2.imread(imgDir+'/'+im)
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # is it necessary for NetVLAD?
        img  = cv2.resize(img, (img_w, img_h),interpolation = cv2.INTER_AREA)
        X[i] = img.flatten()
    return X/255 # scale the image for ESN

if args.saveTorchESN:
    ImgsT = transformImagesESN(path, args.nImage, args.width, args.height)
    torch.save(torch.Tensor(ImgsT), saveTorchESN + args.mode + '.pt')
    print('>> dataset (for ESN) was saved to '+ saveTorchESN + args.mode + '.pt')
