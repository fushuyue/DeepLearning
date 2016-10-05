#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import argparse
import time 
import random

import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList,iterators,training
from chainer.training import extensions
import chainer.functions as F
import chainer.links as L
import h5py

# my models
import models
    
def main():

    parser = argparse.ArgumentParser(description='Train CIFAR10 with Chainer')

    parser.add_argument('--batchsize', '-b', type=int, default=500        , help='Number of images in each mini-batch')

    parser.add_argument('--epoch'    , '-e', type=int, default=10         , help='Number of sweeps over the dataset to train')

    parser.add_argument('--model'    , '-u', type=str, default='CNN_N_P_D', help='Model used')

    parser.add_argument('--method'   , '-m', type=str, default='SGD'      , help='Optimization methods')

    parser.add_argument('--device'   , '-d', type=int, default='0'        , help='Device used')

    parser.add_argument('--flip'     , '-f', type=int, default='0'        , help='Data augmentation')    

    args = parser.parse_args()

    print('')
    print('PS2 BY Shuyue Fu')
    print('----------------------------')
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('Optimization method: {}'.format(args.method))

    temp = {

        'CNN_N_P'       : 'Convolution Neural Network with max pooling and batch normalization',
        'CNN_Drop'      : 'Convolution Neural Network with max pooling and dropout',
        'CNN_Pooling'   : 'Convolution Neural Network with max pooling',
        'CNN_N_P_D'     : 'Convolution Neural Network with max pooling, batch normalization and dropout',
        'CNN_avePooling': 'Convolution Neural Network with average pooling'

    }[args.model]

    print('Model used: {}'.format(temp))

    temp1 = ['No data augmentation','Data Augmentation: flip'][args.flip]
    print(temp1)
    print('----------------------------')
    print('')

    # load data
    CIFAR10_data = h5py.File('CIFAR10.hdf5', 'r')
    x_train      = np.float32(CIFAR10_data['X_train'][:])
    x_test       = np.float32(CIFAR10_data['X_test'][:] )

    y_train      = np.int32(np.array(CIFAR10_data['Y_train'][:]))
    y_test       = np.int32(np.array(CIFAR10_data['Y_test'][:] ))

    CIFAR10_data.close()
    train = tuple((x_train[i],y_train[i]) for i in range(len(x_train)))
    test  = tuple((x_test[i] ,y_test[i] ) for i in range(len(x_test )))
    n = len(train)

    if args.flip == 1:
        # flip the picture
        temp = tuple((x_train[i][:][:,::-1,:],y_train[i]) for i in range(len(x_train)))
        temp = train + temp
        train = tuple(random.sample(temp,n))

    # Set up a neural network model
    model = {

        'CNN_N_P'       : L.Classifier(models.CNN_N_P()),
        'CNN_N_P_D'     : L.Classifier(models.CNN_N_P_D()),
        'CNN_Drop'      : L.Classifier(models.CNN_Drop()),
        'CNN_Pooling'   : L.Classifier(models.CNN_Pooling()),
        'CNN_avePooling': L.Classifier(models.CNN_avePooling())

    }[args.model]

    # copy the model to gpu
    model.to_gpu(args.device)

    # four types of optimization:
    # 1.normal SGD, 2.ADAM, 3.RMSPROP, 4.MomentumSGD
    optimizer = {

        'SGD'        : chainer.optimizers.SGD(),
        'Adam'       : chainer.optimizers.Adam(),
        'MomentumSGD': chainer.optimizers.MomentumSGD(momentum = .99),
        'RMSprop'    : chainer.optimizers.RMSprop(lr=0.001, alpha=0.99, eps=1e-08),

    }[args.method]

    
    optimizer.setup(model)

    # shuffle the training set and test
    train_iter = iterators.SerialIterator(train, batch_size=args.batchsize, shuffle=True)
    test_iter  = iterators.SerialIterator(test , batch_size=args.batchsize, repeat =False, shuffle=False)

    # set update method
    updater    = training.StandardUpdater(train_iter, optimizer, device=args.device)

    # train the multi-layer perceptron by iterating over the training set 20 times.
    trainer    = training.Trainer(updater, (args.epoch, 'epoch'), out=(args.model+args.method))

    # Evaluates the current model on the test dataset at the end of every epoch.
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.device))

    # Accumulates the reported values and emits them to the log file in the output directory.
    trainer.extend(extensions.LogReport())

    # Prints the selected items in the LogReport.
    trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy']))
    
    time1         = time.time()
    trainer.run()  
    time2         = time.time()

    training_time = time2 - time1
    print("Training time: %f" % training_time)

if __name__ == '__main__':
    main()


