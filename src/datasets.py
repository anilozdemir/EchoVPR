#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np


DataSets       = ['GardensPoint'      , 'ESSEX3IN1'      , 'SPEDTest'      , 'Nordland']
Tolerance      = {'GardensPoint' : 2  , 'ESSEX3IN1' : 0  , 'SPEDTest' : 0  , 'Nordland': 10}
N_IMAGE        = {'GardensPoint' : 200, 'ESSEX3IN1' : 210, 'SPEDTest' : 600, 'Nordland': 1000}

def getHiddenRepr(dataset: str):
    '''
        Example usage: 
        getHiddenRepr(dataset = 'GardensPoint')
    '''
    hlrTrain = np.load(f'../data/hiddenRepr-{dataset}.npz')['hlrTrain']
    hlrTest  = np.load(f'../data/hiddenRepr-{dataset}.npz')['hlrTest']
    return hlrTrain, hlrTest