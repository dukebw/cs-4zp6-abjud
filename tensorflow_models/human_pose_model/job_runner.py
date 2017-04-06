#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import subprocess
import sys
import json
import random
import numpy as np
"""
List of Icelandic Volcanoes.
"""
volcanoes = ['Gunnuhver',
             'Trölladyngja',
             'Hengill',
             'Hrómundartindur',
             'Seyðishólar',
             'Laugarfjall',
             'Prestahnúkur',
             'Hveravellir',
             'Hofsjökull',
             'Snækollur',
             'Tungnafellsjökull',
             'Eyjafjallajökull',
             'Katla',
             'Tindfjallajökull',
             'Hekla',
             'Torfajökull',
             'Bárðarbunga',
             'Thórdarhyrna',
             'Vonarskard',
             'Kverkfjöll',
             'Askja',
             'Krafla',
             'Þeistareykjabunga',
             'Öræfajökull',
             'Snæhetta',
             'Snæfell',
             'Helgrindur',
             'Snæfellsjökull']


def train_from_config(learning_rate,
                      batch_size,
                      radius,
                      checkpoint_name,
                      log_dir_num,
                      argv=None):
    """Runs `train.py` either on copper or locally using a set of
    parameters taken from an input JSON config file.

    i.e. running on copper
        python train_from_config.py config.json copper

    i.e. running locally
        python train_from_config.py config.json local

    i.e. running manually
        python train.py
    """

    assert len(argv) == 3, "Usage: python train.py [copper_config.json] [copper/local]"

    # Get basic stuff from JSON file
    with open(argv[1]) as config_file:
        config = json.load(config_file)

    ##########
    # Make config from json + hyperparameter search
    ##########
    config_string = ""
    for option in config:
        config_string += ' --' + option + ' ' + str(config[option])

    config_string += ' --' + 'initial_learning_rate' + ' ' + str(learning_rate)
    config_string += ' --' + 'batch_size' + ' ' + str(batch_size)
    config_string += ' --' + 'heatmap_stddev_pixels' + ' ' + str(radius)
    config_string += ' --' + 'checkpoint_name' + ' ' + checkpoint_name
    print(config_string)
    command = 'python3 -m train' + config_string
    ##########
    # Check if log directory exists
    ##########
    log_dir = config['log_dir'] + '/' + str(log_dir_num)
    if os.path.exists(log_dir):
        print('Using logging directory ' + log_dir)
    else:
        print('Logging directory doesnt exist, creating ' + log_dir)
        os.mkdir(log_dir)

    if argv[2] == 'copper':
        sqsub = 'sqsub -q gpu -f mpi -n 8 --gpp '+'4'+' -r 3600 -o ' + checkpoint_name + '%J.out --gpp=' + \
        str(config['num_gpus']) + ' --mpp=92g --nompirun '
        #print(sqsub + command)
        subprocess.call(sqsub + command, shell=True)

    elif argv[2] == 'local':
        #print(command)
        subprocess.call(command, shell=True)


def train_volcanoes(EXP_FAMILY):
    logdir = os.path.split(os.getcwd())[1]
    logdir = os.path.join(logdir+EXP_FAMILY)
    rand_idx = random.shuffle(range(len(volcanoes)))
    lr_bins = [10**(-i) for i in range(2,4)]
    mb_size_bins = [2**i for i in range(3,6)]
    heatmap_radius_bin = [2**i for i in range(2,5)]

    for i,lr in enumerate(lr_bins):
        learning_rate = np.random.uniform(lr, lr*10)
        for j,mb_size in enumerate(mb_size_bins):
            batch_size = np.random.randint(mb_size, mb_size*2)
            for k, heatmap_radius in enumerate(heatmap_radius_bin):
                radius = np.random.randint(heatmap_radius, heatmap_radius*2)
                log_dir_num = i + j + k
                checkpoint_name = rand_idx(i+j+k)+np.random.randint(0,1000)
                train_from_config(learning_rate,
                                  batch_size,
                                  radius,
                                  checkpoint_name,
                                  log_dir_num)

if __name__ == "__main__":
    lr = 0.01
    batch_size = 32
    radius = 5
    checkpoint_name = 'G'
    log_dir_num = 0
    train_from_config(lr, batch_size, radius, checkpoint_name, log_dir_num, sys.argv)


