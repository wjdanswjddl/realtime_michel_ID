#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
import torchvision
import torchvision.transforms.functional as TF

import numpy as np
import pandas as pd
import h5py
import random
import json
from train_utils import read_config, get_patch_size, count_events
import math
import datetime
from tqdm import tqdm
from networks import Model3Class
import argparse
from io import BytesIO
import pickle
import os


def update_dataloader(dirs, X, Y, size_w, size_d):
    for d in dirs:
        fnames = [f for f in os.listdir(d) if 'hdf5' in f]

        for this_file in fnames:
            #print("loading file", os.path.join(INPUT_DIR, d, this_file))
            frames = h5py.File(os.path.join(d, this_file), "r")
    
            raw = frames.get("rawdata/adc")
            dataX = np.array(raw)
            dataX = dataX.reshape(dataX.shape[0], 1, size_w, size_d)
            
            pixelid = frames.get("pixelid/pixid")
            pixelid = np.array(pixelid)
            is_michel = (pixelid==9900011)
            collapse = np.sum(is_michel, axis=1)
            collapse = np.sum(collapse, axis=1)
            collapse[collapse > 0] = 1
            dataY = collapse

            if X is None:
                assert Y is None
                X = dataX
                Y = dataY

            else:
                X = np.concatenate((X, dataX))
                Y = np.concatenate((Y, dataY))
    
    shuffler = np.random.permutation(len(X))
    X = X[shuffler]
    Y = Y[shuffler]
    return X, Y


def main(args):
    config = read_config(args.config)
    
    # device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # config
    INPUT_DIR = config['training_on_patches']['input_dir']
    dirs_eval = [os.path.join(INPUT_DIR, d) for d in os.listdir(INPUT_DIR)]
    print("number of training directories: " + str(len(dirs_eval)))
    
    IMG_SIZE_W = config['training_on_patches']['img_size_w']
    IMG_SIZE_D = config['training_on_patches']['img_size_d']
    N_CLASSES = config['training_on_patches']['nb_classes']
    N_EPOCH = config['training_on_patches']['nb_epoch']
    
    X, Y = update_dataloader(dirs_eval, None, None, IMG_SIZE_W, IMG_SIZE_D)
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    Y = torch.as_tensor(Y, dtype=torch.float32, device=device)
        
    # model 
    MODEL = config['training_on_patches']['model']
    model = Model3Class()
    model = torch.load(MODEL, map_location=device)
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    model.eval()
    BEST=False
    with torch.no_grad():
        logits_val = model(X)
        Y = Y.type(torch.LongTensor)
        loss_val = criterion(logits_val, Y) #, reduction='mean')
        _, predicted = torch.max(logits_val, dim=1)  # shape: [batch_size]
        correct = (predicted == Y).sum().item()
        val_accuracy = correct / Y.size(0)
        print("Val loss: {:.6f}, val accuracy {}:".format(loss_val, val_accuracy))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run CNN training on input images for shower/michel/other classification')
    parser.add_argument('-c', '--config', help = 'JSON with script configuration')
    parser.add_argument('-o', '--output', help = 'Output model file name')
    parser.add_argument('-g', '--gpu',    help = 'Which GPU index', default = '0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)
