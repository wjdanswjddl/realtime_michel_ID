#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.utils.tensorboard import SummaryWriter
import torchvision
from sklearn.metrics import confusion_matrix
import torchvision.transforms.functional as TF

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import seaborn as sn
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


def save_model(model, name):
    try:
        torch.save(model, name+".pt")
        return True
    except: 
        return False


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
    
    # log
    output_name = args.output
    log_dir = "./logs/"+output_name+"/train/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir)

    # device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # config
    INPUT_DIR = config['training_on_patches']['input_dir']
    dirs_train = [os.path.join(INPUT_DIR, d) for d in os.listdir(INPUT_DIR)]
    print("number of training directories: " + str(len(dirs_train)))
    
    IMG_SIZE_W = config['training_on_patches']['img_size_w']
    IMG_SIZE_D = config['training_on_patches']['img_size_d']
    BATCH_SIZE = config['training_on_patches']['batch_size']
    N_CLASSES = config['training_on_patches']['nb_classes']
    N_EPOCH = config['training_on_patches']['nb_epoch']
    
    FRACTION_VAL = 0.1
    FRACTION_TEST = 0.1
    FRACTION_TRAIN = 1 - FRACTION_VAL - FRACTION_TEST
    X, Y = update_dataloader(dirs_train, None, None, IMG_SIZE_W, IMG_SIZE_D)
    n_train = int(len(X)*FRACTION_TRAIN)
    n_val = int(len(X)*FRACTION_VAL)
    print("n_train", n_train, "n_val", n_val)
    X_train, Y_train = X[:n_train], Y[:n_train]
    X_val, Y_val = X[n_train:n_train+n_val], Y[n_train:n_train+n_val]
    X_train = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    Y_train = torch.as_tensor(Y_train, dtype=torch.float32, device=device)
    X_val = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    Y_val = torch.as_tensor(Y_val, dtype=torch.float32, device=device)
    trainSteps = len(X_train) // BATCH_SIZE
        
    # model 
    model = Model3Class().to(device)
    optimizer = Adam(model.parameters(), lr=3e-5)
    #optimizer = SGD(model.parameters(), lr=3e-3)
    criterion = nn.CrossEntropyLoss()

    n_update_train = 0
    n_update_val = 0
    bestValLoss = 1e10
    bestValAccuracy = 0
    for epoch in range(N_EPOCH):
        totalTrainLoss = 0
        trainCorrect = 0
        valCorrect = 0
    
        model.train()
        for i in tqdm(range(trainSteps)):
            batch_idxs = np.random.randint(0, len(X_train), BATCH_SIZE) # shuffle batch every epoch
            X_batch = X_train[batch_idxs]
            target = Y_train[batch_idxs]
    
            logits = model(X_batch)          
            target = target.long().to(device)
            loss = criterion(logits, target)
    
            totalTrainLoss += loss
    #		trainCorrect += (pred.argmax(1) == y).type(torch.float).sum().item()
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            n_update_train += 1
            writer.add_scalar('michel/train', loss, n_update_train)
    
        avgTrainLoss = totalTrainLoss / trainSteps
    
        model.eval()
        BEST=False
        with torch.no_grad():
            logits_val = model(X_val)
            Y_val = Y_val.type(torch.LongTensor)
            loss_val = criterion(logits_val, Y_val) #, reduction='mean')
            _, predicted = torch.max(logits_val, dim=1)  # shape: [batch_size]
            correct = (predicted == Y_val).sum().item()
            val_accuracy = correct / Y_val.size(0)
            
            if loss_val < bestValLoss:
                bestValLoss = loss_val
                best_model = model
                BEST=True
                print("*******validation loss improved!*******")
                if save_model(best_model, './models/model_bestloss'+datetime.datetime.now().strftime("%Y%m%d")):
                    print("Model3Class saved")
                else:
                    print("[ERR] Couldn't save model")
    
            if val_accuracy > bestValAccuracy:
                bestValAccuracy = val_accuracy
                print("best accuracy", val_accuracy)
                if save_model(best_model, './models/model_bestacc'+datetime.datetime.now().strftime("%Y%m%d")):
                    print("Model3Class saved")
                else:
                    print("[ERR] Couldn't save model")
    
    
        n_update_val += 1
        writer.add_scalar('michel/validation', loss_val, n_update_val)
        writer.add_scalar('loss/validation', loss_val, n_update_val)
    
        # print the model training and validation information
        print("[INFO] EPOCH: {}/{}".format(epoch + 1, N_EPOCH))
        print("Train loss: {:.6f}: ".format(avgTrainLoss)) #, trainCorrect))
        print("Val loss: {:.6f}, val accuracy {}:".format(loss_val, val_accuracy))
        print("Best val loss: {:.6f}, best val accuracy {} :".format(bestValLoss, bestValAccuracy))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run CNN training on input images for shower/michel/other classification')
    parser.add_argument('-c', '--config', help = 'JSON with script configuration')
    parser.add_argument('-o', '--output', help = 'Output model file name')
    parser.add_argument('-g', '--gpu',    help = 'Which GPU index', default = '0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)
