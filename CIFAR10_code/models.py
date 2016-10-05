import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
  
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
  
class CNN_N_P(Chain):

# convolution neural network with batch normalization and max pooling

    def __init__(self):
        super(CNN_N_P, self).__init__(

            # input 3 channel of 32*32
            conv1_1 = L.Convolution2D(3 , 64 , 3, pad=1),
            bn1_1   = L.BatchNormalization(64),
            conv1_2 = L.Convolution2D(64, 64 , 3, pad=1),
            bn1_2   = L.BatchNormalization(64),

            conv2_1 = L.Convolution2D(64, 128, 3, pad=1),
            bn2_1   = L.BatchNormalization(128),
            conv2_2 = L.Convolution2D(128, 128, 3, pad=1),
            bn2_2   = L.BatchNormalization(128),

            conv3_1 = L.Convolution2D(128, 256, 3, pad=1),
            bn3_1   = L.BatchNormalization(256),
            conv3_2 = L.Convolution2D(256, 256, 3, pad=1),
            bn3_2   = L.BatchNormalization(256),
            conv3_3 = L.Convolution2D(256, 256, 3, pad=1),
            bn3_3   = L.BatchNormalization(256),
            conv3_4 = L.Convolution2D(256, 256, 3, pad=1),
            bn3_4   = L.BatchNormalization(256),

            fc4 = L.Linear(256*4*4, 500),                                       
            fc5 = L.Linear(500    , 500),                                         
            fc6 = L.Linear(500    , 10 ),
        )

    def __call__(self, x):

        p = 0.25
        dropout_bool = False
        bn_bool = True
        h = F.relu(self.bn1_1(self.conv1_1(x), bn_bool))
        h = F.relu(self.bn1_2(self.conv1_2(h), bn_bool))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, p, dropout_bool)

        h = F.relu(self.bn2_1(self.conv2_1(h), bn_bool))
        h = F.relu(self.bn2_2(self.conv2_2(h), bn_bool))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, p, dropout_bool)

        h = F.relu(self.bn3_1(self.conv3_1(h), bn_bool))
        h = F.relu(self.bn3_2(self.conv3_2(h), bn_bool))
        h = F.relu(self.bn3_3(self.conv3_3(h), bn_bool))
        h = F.relu(self.bn3_4(self.conv3_4(h), bn_bool))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, p, dropout_bool)

        h = F.dropout(F.relu(self.fc4(h)), p, dropout_bool)
        h = F.dropout(F.relu(self.fc5(h)), p, dropout_bool)
        h = self.fc6(h)
        L_out = h
        return L_out

 
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
  
class CNN_N_P_D(Chain):

# convolution neural network with batch normalization and max pooling and dropout

    def __init__(self):
        super(CNN_N_P_D, self).__init__(

            # input 3 channel of 32*32
            conv1_1=L.Convolution2D(3, 64, 3, pad=1),
            bn1_1=L.BatchNormalization(64,False),
            conv1_2=L.Convolution2D(64, 64, 3, pad=1),
            bn1_2=L.BatchNormalization(64,False),

            conv2_1=L.Convolution2D(64, 128, 3, pad=1),
            bn2_1=L.BatchNormalization(128,False),
            conv2_2=L.Convolution2D(128, 128, 3, pad=1),
            bn2_2=L.BatchNormalization(128,False),

            conv3_1=L.Convolution2D(128, 256, 3, pad=1),
            bn3_1=L.BatchNormalization(256,False),
            conv3_2=L.Convolution2D(256, 256, 3, pad=1),
            bn3_2=L.BatchNormalization(256,False),
            conv3_3=L.Convolution2D(256, 256, 3, pad=1),
            bn3_3=L.BatchNormalization(256,False),
            conv3_4=L.Convolution2D(256, 256, 3, pad=1),
            bn3_4=L.BatchNormalization(256,False),

            fc4 = L.Linear(256*4*4, 500),                                       
            fc5 = L.Linear(500, 500),                                         
            fc6 = L.Linear(500,10),
        )

    def __call__(self, x):

        p = 0.25
        dropout_bool = True
        bn_bool = True
        h = F.relu(self.bn1_1(self.conv1_1(x), bn_bool))
        h = F.relu(self.bn1_2(self.conv1_2(h), bn_bool))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, p, dropout_bool)

        h = F.relu(self.bn2_1(self.conv2_1(h), bn_bool))
        h = F.relu(self.bn2_2(self.conv2_2(h), bn_bool))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, p, dropout_bool)

        h = F.relu(self.bn3_1(self.conv3_1(h), bn_bool))
        h = F.relu(self.bn3_2(self.conv3_2(h), bn_bool))
        h = F.relu(self.bn3_3(self.conv3_3(h), bn_bool))
        h = F.relu(self.bn3_4(self.conv3_4(h), bn_bool))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, p, dropout_bool)

        h = F.dropout(F.relu(self.fc4(h)), p, dropout_bool)
        h = F.dropout(F.relu(self.fc5(h)), p, dropout_bool)
        h = self.fc6(h)
        L_out = h
        return L_out

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

class CNN_Drop(Chain):
# convolution neural network with max pooling

    def __init__(self):
        super(CNN_Drop, self).__init__(

            # input 3 channel of 32*32
            conv1_1=L.Convolution2D(3, 64, 3, pad=1),
            conv1_2=L.Convolution2D(64, 64, 3, pad=1),
            

            conv2_1=L.Convolution2D(64, 128, 3, pad=1),
            conv2_2=L.Convolution2D(128, 128, 3, pad=1),
            

            conv3_1=L.Convolution2D(128, 256, 3, pad=1),
            conv3_2=L.Convolution2D(256, 256, 3, pad=1),
            conv3_3=L.Convolution2D(256, 256, 3, pad=1),
            conv3_4=L.Convolution2D(256, 256, 3, pad=1),

            fc4 = L.Linear(256*4*4, 500),                                       
            fc5 = L.Linear(500, 500),                                         
            fc6 = L.Linear(500,10),
        )
    def __call__(self, x):

        p = 0.25
        dropout_bool = True
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, p, dropout_bool)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, p, dropout_bool)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.relu(self.conv3_4(h))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.dropout(h, p, dropout_bool)

        h = F.dropout(F.relu(self.fc4(h)), p, dropout_bool)
        h = F.dropout(F.relu(self.fc5(h)), p, dropout_bool)

        h = self.fc6(h)
        L_out = h
        return L_out


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

class CNN_Pooling(Chain):
# convolution neural network with max pooling

    def __init__(self):
        super(CNN_Pooling, self).__init__(

            # input 3 channel of 32*32
            conv1_1=L.Convolution2D(3, 64, 3, pad=1),
            conv1_2=L.Convolution2D(64, 64, 3, pad=1),
            

            conv2_1=L.Convolution2D(64, 128, 3, pad=1),
            conv2_2=L.Convolution2D(128, 128, 3, pad=1),
            

            conv3_1=L.Convolution2D(128, 256, 3, pad=1),
            conv3_2=L.Convolution2D(256, 256, 3, pad=1),
            conv3_3=L.Convolution2D(256, 256, 3, pad=1),
            conv3_4=L.Convolution2D(256, 256, 3, pad=1),

            fc4 = L.Linear(256*4*4, 500),                                       
            fc5 = L.Linear(500, 500),                                         
            fc6 = L.Linear(500,10),
        )
    def __call__(self, x):

        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, 2, 2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, 2, 2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.relu(self.conv3_4(h))
        h = F.max_pooling_2d(h, 2, 2)

        h = F.relu(self.fc4(h))
        h = F.relu(self.fc5(h))
        h = self.fc6(h)
        L_out = h
        return L_out


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

class CNN_avePooling(Chain):
# convolution neural network with max pooling

    def __init__(self):
        super(CNN_avePooling, self).__init__(

            # input 3 channel of 32*32
            conv1_1=L.Convolution2D(3, 64, 3, pad=1 ),
            conv1_2=L.Convolution2D(64, 64, 3, pad=1),
            
            conv2_1=L.Convolution2D(64, 128, 3, pad=1 ),
            conv2_2=L.Convolution2D(128, 128, 3, pad=1),
            
            conv3_1=L.Convolution2D(128, 256, 3, pad=1),
            conv3_2=L.Convolution2D(256, 256, 3, pad=1),
            conv3_3=L.Convolution2D(256, 256, 3, pad=1),
            conv3_4=L.Convolution2D(256, 256, 3, pad=1),

            fc4 = L.Linear(256*4*4, 500),                                       
            fc5 = L.Linear(500, 500),                                         
            fc6 = L.Linear(500,10),
        )
    def __call__(self, x):

        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.average_pooling_2d(h, 2, 2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.average_pooling_2d(h, 2, 2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.relu(self.conv3_4(h))
        h = F.average_pooling_2d(h, 2, 2)

        h = F.relu(self.fc4(h))
        h = F.relu(self.fc5(h))
        h = self.fc6(h)
        L_out = h
        return L_out