import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate

import numpy as np
import json
import csv
import h5py
import random
import os
import os.path

import cv2

def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    
    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3,0,1,2]))


def load_rgb_frames(root, vid, start, num):
    frames = []
    cap = cv2.VideoCapture(os.path.join(root, vid))

    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        w,h,c = frame.shape
        if w < 226 or h < 226:
            d = 226.-min(w,h)
            sc = 1+d/min(w,h)
            frame = cv2.resize(frame,dsize=(0,0),fx=sc,fy=sc)
        frame = (frame/255.)*2 - 1
        frames.append(frame)

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    if len(frames) < 64:
        frames += [frames[-i] for i in range(len(frames))]
    start = int((len(frames)-64)*start)
    return np.asarray(frames[start:start+64], dtype=np.float32)

def load_flow_frames(image_dir, vid, start, num):
  frames = []
  for i in range(start, start+num):
    imgx = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'x.jpg'), cv2.IMREAD_GRAYSCALE)
    imgy = cv2.imread(os.path.join(image_dir, vid, vid+'-'+str(i).zfill(6)+'y.jpg'), cv2.IMREAD_GRAYSCALE)
    
    w,h = imgx.shape
    if w < 224 or h < 224:
        d = 224.-min(w,h)
        sc = 1+d/min(w,h)
        imgx = cv2.resize(imgx,dsize=(0,0),fx=sc,fy=sc)
        imgy = cv2.resize(imgy,dsize=(0,0),fx=sc,fy=sc)
        
    imgx = (imgx/255.)*2 - 1
    imgy = (imgy/255.)*2 - 1
    img = np.asarray([imgx, imgy]).transpose([1,2,0])
    frames.append(img)
  return np.asarray(frames, dtype=np.float32)


def make_dataset(root, split):
    dataset = []
    with open(os.path.join(root, split), 'r') as f:
        data = [line.split('\t')[0] for line in f]

    for vid in data:
        if not os.path.exists(os.path.join(root, vid)):
            continue
            
        label = 0 if 'noFights' in vid else 1
        dataset.append((vid, label))
    
    return dataset


class ViolenceDetection(data_utl.Dataset):

    def __init__(self, root, split, mode, transforms=None):
        
        self.data = make_dataset(root, split)
        self.transforms = transforms
        self.mode = mode
        self.root = root

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        vid, label = self.data[index]
        start_f = random.random()

        if self.mode == 'rgb':
            imgs = load_rgb_frames(self.root, vid, start_f, 64)
        else:
            imgs = load_flow_frames(self.root, vid, start_f, 64)
        label = np.tile(label, (1, 64)).astype(np.float32)

        imgs = self.transforms(imgs)

        return video_to_tensor(imgs), torch.from_numpy(label)

    def __len__(self):
        return len(self.data)

    
class VSD2014YouTube(ViolenceDetection):

    def __init__(self, root, mode, transforms=None):
        self.root = root
        self.data = self.list_dataset()
        self.transforms = transforms
        self.mode = mode
        
    def list_dataset(self):
        dataset = []
        for file in os.listdir(os.path.join(self.root, 'violence')):
            dataset.append((os.path.join('violence', file), 1))
        for file in os.listdir(os.path.join(self.root, 'no-violence')):
            dataset.append((os.path.join('no-violence', file), 0))
        random.shuffle(dataset)
        return dataset