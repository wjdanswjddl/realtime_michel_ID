import numpy as np
from os import listdir
from os.path import isfile, join
import os, json
from collections import defaultdict
import datetime
import copy

def read_config(cfgname):
    config = None
    with open(cfgname, 'r') as fin:
        config = json.loads(fin.read());
    if config is None:
        print ('This script requires configuration file: config.json')
        exit(1)
    return config

def get_patch_size(folder):
    dlist = [f for f in listdir(folder) if '' in f]
    flist = [f for f in listdir(folder + '/' + dlist[0]) if '_x.npy' in f]
    d = np.load(folder + '/' + dlist[0] + '/' + flist[0])
    return d.shape[1], d.shape[2]

def count_events(folder, key):
    nevents = 0
    dlist = [f for f in listdir(folder) if key in f]
    dlist.sort()
    for dirname in dlist:
        flist = [f for f in listdir(folder + '/' + dirname) if '_y.npy' in f]
        for fname in flist:
            d = np.load(folder + '/' + dirname + '/' + fname)
            nevents += d.shape[0]
    return nevents
