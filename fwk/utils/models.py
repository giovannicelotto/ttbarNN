import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.simplefilter(action='ignore', category=dlerror)

import ROOT

import os
import tensorflow as tf
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras

# from keras import backend as K
# K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads =32, inter_op_parallelism_threads=32)))


# import uproot
# import numpy as np
# from lbn import LBN,LBNLayer
# import matplotlib.pyplot as plt
# import pandas as pd

# from sklearn.model_selection import train_test_split
# import six
# import random



# example of creating a CNN with an inception module
# from keras.models import Model
# from keras.layers import Input
# from tf.keras.layers import Conv1D
# from tf.keras.layers import MaxPooling1D
# from tf.keras.layers import Dense,Flatten,Dropout
# from tf.keras.layers.merge import concatenate
# from tf.keras.utils import plot_model
#
# from keras.layers import Input

def residual_module_new(data, K, stride, chanDim, red=False,
		reg=0.0001, bnEps=2e-5, bnMom=0.9):
    # the shortcut branch of the ResNet module should be
    # initialize as the input (identity) data
    shortcut = data

    # the first block of the ResNet module are the 1x1 CONVs
    bn1 = tf.keras.layers.BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(data)
    act1 = tf.keras.layers.Activation("relu")(bn1)
    conv1 = tf.keras.layers.Conv1D(int(K * 0.25), 1, use_bias=False,kernel_regularizer=tf.keras.regularizers.l2(reg))(act1)
    # the second block of the ResNet module are the 3x3 CONVs
    bn2 = tf.keras.layers.BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv1)
    act2 = tf.keras.layers.Activation("relu")(bn2)
    conv2 = tf.keras.layers.Conv1D(int(K * 0.25),3,padding='same',strides=stride, use_bias=False,kernel_regularizer=tf.keras.regularizers.l2(reg))(act2)
    # the third block of the ResNet module is another set of 1x1 CONVs
    bn3 = tf.keras.layers.BatchNormalization(axis=chanDim, epsilon=bnEps,momentum=bnMom)(conv2)
    act3 = tf.keras.layers.Activation("relu")(bn3)
    conv3 = tf.keras.layers.Conv1D(K,  1 ,use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(reg))(act3)
    # if we are to reduce the spatial size, apply a CONV layer to the shortcut
    if red:
       shortcut = tf.keras.layers.Conv1D(K, 1, use_bias=False,
       		kernel_regularizer=tf.keras.regularizers.l2(reg))(act1)

    # add together the shortcut and the final CONV
    x = tf.keras.layers.add([conv3, shortcut])

    # return the addition as the output of the ResNet module
    return x

def residual_module(layer_in, n_filters):
	merge_input = layer_in
	# check if the number of filters needs to be increase, assumes channels last format
	if layer_in.shape[-1] != n_filters:
		merge_input = tf.keras.layers.Conv1D(n_filters, 1, padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
	# conv1
	conv1 = tf.keras.layers.Conv1D(n_filters, 3, padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
	# conv2
	conv2 = tf.keras.layers.Conv1D(n_filters, 3, padding='same', activation='linear', kernel_initializer='he_normal')(conv1)
	# add filters, assumes filters/channels last
	layer_out = tf.keras.layers.add([conv2, merge_input])
	# activation function
	layer_out = tf.keras.layers.Activation('relu')(layer_out)
	return layer_out

def residual_module_dense(layer_in, n_filters,dense_kwargs,activation):
	merge_input = layer_in
	# check if the number of filters needs to be increase, assumes channels last format
	# if layer_in.shape[-1] != n_filters:
	# 	merge_input = tf.keras.layers.Conv1D(n_filters, 1, padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
	# conv1
	dense1 = tf.keras.layers.Dense(n_filters,**dense_kwargs)(layer_in)
	# conv2
	dense2 = tf.keras.layers.Dense(n_filters,**dense_kwargs)(dense1)
	# add filters, assumes filters/channels last
	layer_out = tf.keras.layers.add([dense2, merge_input])
	# activation function
	layer_out = tf.keras.layers.Activation(activation)(layer_out)
	return layer_out

# function for creating a naive inception block
def naive_inception_module(layer_in, f1, f2, f3,activation):
	# 1x1 conv
	conv1 = tf.keras.layers.Conv1D(f1, 1, padding='same', activation=activation)(layer_in)
	# 3x3 conv
	conv3 = tf.keras.layers.Conv1D(f2, 3, padding='same', activation=activation)(layer_in)
	# 5x5 conv
	conv5 = tf.keras.layers.Conv1D(f3, 5, padding='same', activation=activation)(layer_in)
	# 3x3 max pooling
	pool = tf.keras.layers.MaxPooling1D(3, strides=1, padding='same')(layer_in)
	# concatenate filters, assumes filters/channels last
	layer_out = tf.keras.layers.concatenate([conv1, conv3, conv5, pool], axis=-1)
	return layer_out

# def inception_module(layer_in, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
# 	# 1x1 conv
# 	conv1 = keras.layers.Conv1D(f1, 1, padding='same', activation='relu')(layer_in)
# 	# 3x3 conv
# 	conv3 = keras.layers.Conv1D(f2_in, 1, padding='same', activation='relu')(layer_in)
# 	conv3 = keras.layers.Conv1D(f2_out, 3, padding='same', activation='relu')(conv3)
# 	# 5x5 conv
# 	conv5 = keras.layers.Conv1D(f3_in, 1, padding='same', activation='relu')(layer_in)
# 	conv5 = keras.layers.Conv1D(f3_out, 5, padding='same', activation='relu')(conv5)
# 	# 3x3 max pooling
# 	pool = keras.layers.MaxPooling1D(3, strides=1, padding='same')(layer_in)
# 	pool = keras.layers.Conv1D(f4_out, 1, padding='same', activation='relu')(pool)
# 	# concatenate filters, assumes filters/channels last
# 	layer_out = keras.layers.merge.concatenate([conv1, conv3, conv5, pool], axis=-1)
# 	return layer_out
def inception_module(layer_in, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
	# 1x1 conv
	conv1 = tf.keras.layers.Conv1D(f1, 1, padding='same', activation='relu')(layer_in)
	# 3x3 conv
	conv3 = tf.keras.layers.Conv1D(f2_in, 1, padding='same', activation='relu')(layer_in)
	conv3 = tf.keras.layers.Conv1D(f2_out, 3, padding='same', activation='relu')(conv3)
	# 5x5 conv
	conv5 = tf.keras.layers.Conv1D(f3_in, 1, padding='same', activation='relu')(layer_in)
	conv5 = tf.keras.layers.Conv1D(f3_out, 5, padding='same', activation='relu')(conv5)
	# 3x3 max pooling
	pool = tf.keras.layers.MaxPooling1D(3, strides=1, padding='same')(layer_in)
	pool = tf.keras.layers.Conv1D(f4_out, 1, padding='same', activation='relu')(pool)
	# concatenate filters, assumes filters/channels last
	layer_out = tf.keras.layers.concatenate([conv1, conv3, conv5, pool], axis=-1)
	return layer_out

# def getModel(nParticles=20,regRate=1e-3,activation='selu',dropout=0.1,nDense=3,nNodes=10):
def getModel(regRate=1e-3,activation='selu',dropout=0.1,nDense=3,nNodes=10):

    # lbn_layer = LBNLayer(n_particles=nParticles, boost_mode="pairs")
    #~ lbn_layer = LBNLayer(n_particles=20, boost_mode="combinations") #20 to 10
    #~ l2_reg = tf.keras.regularizers.l2(1e-4)
    l2_reg = tf.keras.regularizers.l2(regRate)
    dense_kwargs = dict(
        activation=activation,
        kernel_initializer=tf.keras.initializers.lecun_normal(),
        kernel_regularizer=l2_reg,
    )

    # model=tf.keras.models.Sequential()
    # model.add(lbn_layer)
    # visible = keras.layers.Input(shape=(11,4))
    visible = tf.keras.layers.Input(shape=(44,))
    # visible = keras.layers.Input(shape=(36,))
    # layer1 = naive_inception_module(visible, 64, 128, 32,activation)
    # layer = inception_module(visible, 64, 96, 128, 16, 32, 32)
    # layer = inception_module(visible, 16, 32, 64, 8, 16, 16)
    # layer = keras.layers.MaxPooling1D(3, strides=1, padding='same')(layer)
    # layer = inception_module(layer, 128, 128, 192, 32, 96, 64)

    # layer= residual_module(layer,64)

    # x = keras.layers.Flatten()(layer)
    # x = keras.layers.Flatten()(layer)
    # model.add(naive_inception_module())
    x=tf.keras.layers.Dense(nNodes,**dense_kwargs)(visible)
    for i in range(nDense-1):
        x=tf.keras.layers.Dense(nNodes,**dense_kwargs)(x)
        x=tf.keras.layers.Dropout(dropout)(x)

    # model.add(tf.keras.layers.SeparableConv1D(256, 2, padding='same',depth_multiplier=2, activation=activation, input_shape=(11, 4)))
    # model.add(tf.keras.layers.SeparableConv1D(256, 2,padding='same',depth_multiplier=2, activation=activation))
    # model.add(tf.keras.layers.UpSampling1D(size=2))
    # model.add(tf.keras.layers.Conv1D(200, 2, activation=activation))
    # model.add(tf.keras.layers.MaxPooling1D(pool_size=4))
    # model.add(tf.keras.layers.Conv1D(200, 2, activation=activation))
    # model.add(tf.keras.layers.Conv1D(256, 2,padding='causal', activation=activation))
    # model.add(tf.keras.layers.SeparableConv1D(256, 2, activation=activation))
    # model.add(tf.keras.layers.UpSampling1D(size=2))
    # model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    # model.add(tf.keras.layers.SeparableConv1D(256, 2, activation=activation))
    # model.add(tf.keras.layers.Flatten())

    # for i in range(nDense):
        # model.add(tf.keras.layers.Dense(nNodes,**dense_kwargs))
        # model.add(tf.keras.layers.Dropout(dropout))

    # model.add(lbn_layer)

    # model.add(tf.keras.layers.Dense(4,activation='linear'))
    # hidden = keras.layers.Dense(4,activation='linear')(x)
    hidden = tf.keras.layers.Dense(4)(x)
    #~ model.add(tf.keras.layers.Dense(1,activation='linear'))
    # model = keras.Model(inputs=visible, outputs=hidden)
    model = tf.keras.Model(inputs=visible, outputs=hidden)
    model.summary()

    return model

def normalize(array):
    for particles in array:
        for i in range(particles.shape[1]):
            scaler = max(np.max(particles[:,i]), abs(np.min(particles[:,i])))
            if scaler!=0:
                particles[:, i] = particles[:,i]/scaler

    return array

def LSTM(inputLayer, cellNumber,  dropOut, kwargs):
    # layer = tf.keras.layers.Masking(mask_value=0.)(inputLayer)
    layer = tf.keras.layers.Masking(mask_value=-999.)(inputLayer)
    layer = tf.keras.layers.LSTM(cellNumber, **kwargs)(layer)
    layer = tf.keras.layers.Dropout(dropOut)(layer)
    # layer = tf.expand_dims(layer, -1)

    return layer

def lstmModel(regRate=1e-3,activation='selu',dropout=0.1,nDense=3,nNodes=10):
	l2_reg = tf.keras.regularizers.l2(regRate)
	dense_kwargs = dict(
	    activation=activation,
	    kernel_initializer=tf.keras.initializers.lecun_normal(),
	    kernel_regularizer=l2_reg,
	)

	cellNumber=200
	convFilter=200

	inputJets = tf.keras.layers.Input(shape=(None,4))
	inputBJets = tf.keras.layers.Input(shape=(None,4))
	inputLMET = tf.keras.layers.Input(shape=(None,4))

	xJets = LSTM(inputJets, cellNumber, dropout, dense_kwargs)
	xJets = tf.keras.layers.Conv1D(convFilter, 3, activation=activation)(xJets)

	xBJets = LSTM(inputBJets, cellNumber, dropout, dense_kwargs)
	xBJets = tf.keras.layers.Conv1D(convFilter, 3, activation=activation)(xBJets)

	xMET = LSTM(inputLMET, cellNumber, dropout, dense_kwargs)
	xMET = tf.keras.layers.Conv1D(convFilter, 3, activation=activation)(xMET)
	# xMET = tf.keras.layers.Dense(nNodes,**dense_kwargs)(inputLMET)
	# xMET = tf.keras.layers.Dense(nNodes,**dense_kwargs)(xMET)

	# z = tf.keras.layers.Add()([xCharged, xNeutral, xSV])
	z = tf.keras.layers.Add()([xMET, xJets, xBJets, xMET])
	z = tf.keras.layers.Flatten()(z)
	z = tf.keras.layers.Dropout(dropout)(z)

	x=tf.keras.layers.Dense(nNodes,**dense_kwargs)(z)
	for i in range(nDense-1):
	    x=tf.keras.layers.Dense(nNodes,**dense_kwargs)(x)
	    x=tf.keras.layers.Dropout(dropout)(x)

	outputLayer = tf.keras.layers.Dense(4)(z)

	model = tf.keras.Model(inputs=[inputLMET, inputJets, inputBJets], outputs=outputLayer, name="JetModel")
	model.summary()

	return model

def getRhoRegModel(regRate=1e-3,activation='selu',dropout=0.1,nDense=5,nNodes=200):
    l2_reg = tf.keras.regularizers.l2(regRate)
    dense_kwargs = dict(
        activation=activation,
        kernel_initializer=tf.keras.initializers.lecun_normal(),
        kernel_regularizer=l2_reg,
    )
    # cellNumber = 100
    cellNumber = nNodes

    ##Input layer
    inputJets = tf.keras.layers.Input(shape=(None,5))
    inputOther = tf.keras.layers.Input(shape=(None,4))

    xJets = LSTM(inputJets, cellNumber, dropout, dense_kwargs)
    xOther = LSTM(inputOther, cellNumber, dropout, dense_kwargs)

    # xJets = tf.keras.layers.Flatten()(xJets)
    # xOther = tf.keras.layers.Flatten()(xOther)

    # xJets = residual_module_dense(xJets,nNodes,dense_kwargs,activation)
    # xOther = residual_module_dense(xOther,nNodes,dense_kwargs,activation)
    # xJets = residual_module_dense(xJets,nNodes,dense_kwargs,activation)
    # xOther = residual_module_dense(xOther,nNodes,dense_kwargs,activation)

    # z = tf.keras.layers.Add()([xJets, xOther])
    z = tf.keras.layers.concatenate(([xJets, xOther]))
    z = tf.keras.layers.Flatten()(z)
    z = tf.keras.layers.Dropout(dropout)(z)

    x=tf.keras.layers.Dense(nNodes,**dense_kwargs)(z)
    x=tf.keras.layers.Dropout(dropout)(x)
    for i in range(nDense-1):
        x=tf.keras.layers.Dense(nNodes,**dense_kwargs)(x)
        x=tf.keras.layers.Dropout(dropout)(x)

    # outputLayer = tf.keras.layers.Dense(20,activation='sigmoid')(z)
    # outputLayer = tf.keras.layers.Dense(1,activation='linear')(x)
    outputLayer = tf.keras.layers.Dense(1,activation='relu')(x)

    model = tf.keras.Model(inputs=[inputJets, inputOther], outputs=outputLayer, name="taggerModel")
    model.summary()

    return model

def getRhoRegModelFlat(regRate=1e-3,activation='selu',dropout=0.1,nDense=5,nNodes=200):
    l2_reg = tf.keras.regularizers.l2(regRate)
    dense_kwargs = dict(
        activation=activation,
        # kernel_initializer=tf.keras.initializers.lecun_normal(),
        kernel_regularizer=l2_reg,
    )
    # cellNumber = 100
    # cellNumber = nNodes

    ##Input layer
    # inputJets = tf.keras.layers.Input(shape=(None,5))
    # inputOther = tf.keras.layers.Input(shape=(None,4))
    # inputs = tf.keras.layers.Input(shape=(42))
    # inputs = tf.keras.layers.Input(shape=(36))
    # inputs = tf.keras.layers.Input(shape=(48))
    # inputs = tf.keras.layers.Input(shape=(36))
    # inputs = tf.keras.layers.Input(shape=(33))
    # inputs = tf.keras.layers.Input(shape=(25))
    inputs = tf.keras.layers.Input(shape=(24))
    # inputs = tf.keras.layers.Input(shape=(20))

    # xJets = LSTM(inputJets, cellNumber, dropout, dense_kwargs)
    # xOther = LSTM(inputOther, cellNumber, dropout, dense_kwargs)

    # xJets = tf.keras.layers.Flatten()(xJets)
    # xOther = tf.keras.layers.Flatten()(xOther)

    # xJets = residual_module_dense(xJets,nNodes,dense_kwargs,activation)
    # xOther = residual_module_dense(xOther,nNodes,dense_kwargs,activation)
    # xJets = residual_module_dense(xJets,nNodes,dense_kwargs,activation)
    # xOther = residual_module_dense(xOther,nNodes,dense_kwargs,activation)

    # z = tf.keras.layers.Add()([xJets, xOther])
    # z = tf.keras.layers.concatenate(([xJets, xOther]))
    # z = tf.keras.layers.Flatten()(z)
    # z = tf.keras.layers.Dropout(dropout)(z)

    x=tf.keras.layers.Dense(nNodes,**dense_kwargs)(inputs)
    x=tf.keras.layers.BatchNormalization()(x)
    # x=tf.keras.layers.Dense(nNodes,**dense_kwargs)(z)
    x=tf.keras.layers.Dropout(dropout)(x)
    for i in range(nDense-1):
        x=tf.keras.layers.Dense(nNodes,**dense_kwargs)(x)
        x=tf.keras.layers.Dropout(dropout)(x)

    outputLayer = tf.keras.layers.Dense(1,activation='sigmoid')(x)
    # outputLayer = tf.keras.layers.Dense(1,activation='linear')(x)
    # outputLayer = tf.keras.layers.Dense(1,activation='relu')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputLayer, name="rhoRegFlatModel")
    model.summary()

    return model


def getTaggerModelFlat(regRate=1e-3,activation='selu',dropout=0.1,nDense=5,nNodes=200):
    l2_reg = tf.keras.regularizers.l2(regRate)
    dense_kwargs = dict(
        activation=activation,
        kernel_initializer=tf.keras.initializers.lecun_normal(),
        kernel_regularizer=l2_reg,
    )
    inputs = tf.keras.layers.Input(shape=(36))
    # inputs = tf.keras.layers.Input(shape=(42))
    # inputs = tf.keras.layers.Input(shape=(48))
    x=tf.keras.layers.Dense(nNodes,**dense_kwargs)(inputs)
    x=tf.keras.layers.Dropout(dropout)(x)
    for i in range(nDense-1):
        x=tf.keras.layers.Dense(nNodes,**dense_kwargs)(x)
        x=tf.keras.layers.Dropout(dropout)(x)

    # outputLayer = tf.keras.layers.Dense(6,activation='sigmoid')(x)
    outputLayer = tf.keras.layers.Dense(4,activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputLayer, name="taggerFlatModel")
    model.summary()

    return model


def getTaggerModel(regRate=1e-3,activation='selu',dropout=0.1,nDense=5,nNodes=200):
    l2_reg = tf.keras.regularizers.l2(regRate)
    dense_kwargs = dict(
        activation=activation,
        kernel_initializer=tf.keras.initializers.lecun_normal(),
        kernel_regularizer=l2_reg,
    )

    cellNumber = 250

    ##Input layer
    inputJets = tf.keras.layers.Input(shape=(None,5))
    inputOther = tf.keras.layers.Input(shape=(None,4))

    xJets = LSTM(inputJets, cellNumber, dropout, dense_kwargs)
    xOther = LSTM(inputOther, cellNumber, dropout, dense_kwargs)

    # xJets=residual_module(xJets,64)
    # xOther=residual_module(xOther,64)
    stages=(2, 3, 4)
    filters=(16, 32, 64, 128)
    K=64
    chanDim=1
    bnEps=2e-5
    bnMom=0.9

    xJets = tf.keras.layers.BatchNormalization(axis=chanDim, epsilon=bnEps,
    	momentum=bnMom)(xJets)

    # apply CONV => BN => ACT => POOL to reduce spatial size
    xJets = tf.keras.layers.Conv1D(filters[0], 5, use_bias=False,
    	padding="same", kernel_regularizer=tf.keras.regularizers.l2(regRate))(xJets)
    xJets = tf.keras.layers.BatchNormalization(axis=chanDim, epsilon=bnEps,
    	momentum=bnMom)(xJets)
    xJets = tf.keras.layers.Activation("relu")(xJets)
    xJets = tf.keras.layers.ZeroPadding1D((1, 1))(xJets)
    xJets = tf.keras.layers.MaxPooling1D(3, strides=2)(xJets)

    xJets = tf.keras.layers.BatchNormalization(axis=chanDim, epsilon=bnEps,
    	momentum=bnMom)(xJets)

    # apply CONV => BN => ACT => POOL to reduce spatial size
    xOther = tf.keras.layers.Conv1D(filters[0], 5, use_bias=False,
    	padding="same", kernel_regularizer=tf.keras.regularizers.l2(regRate))(xOther)
    xOther = tf.keras.layers.BatchNormalization(axis=chanDim, epsilon=bnEps,
    	momentum=bnMom)(xOther)
    xOther = tf.keras.layers.Activation("relu")(xOther)
    xOther = tf.keras.layers.ZeroPadding1D((1, 1))(xOther)
    xOther = tf.keras.layers.MaxPooling1D(3, strides=2)(xOther)

	# loop over the number of stages
    for i in range(0, len(stages)):
        # initialize the stride, then apply a residual module
        # used to reduce the spatial size of the input volume
        stride = 1 if i == 0 else 1
        xJets = residual_module_new(xJets, filters[i + 1], stride,
        		chanDim, red=True, bnEps=2e-5, bnMom=0.9)
        xOther = residual_module_new(xOther, filters[i + 1], stride,
        		chanDim, red=True, bnEps=2e-5, bnMom=0.9)

        # loop over the number of layers in the stage
        for j in range(0, stages[i] - 1):
        	# apply a ResNet module
        	xJets = residual_module_new(xJets, filters[i + 1],
        		1, chanDim, bnEps=2e-5, bnMom=0.9)
        	xOther = residual_module_new(xOther, filters[i + 1],
        		1, chanDim, bnEps=2e-5, bnMom=0.9)

    # xJets=residual_module_new(xJets, K, stride, chanDim, red=False,
    # 		reg=regRate, bnEps=2e-5, bnMom=0.9)
    # xOther=residual_module_new(xOther, K, stride, chanDim, red=False,
    # 		reg=regRate, bnEps=2e-5, bnMom=0.9)

# residual_module_new

    z = tf.keras.layers.Add()([xJets, xOther])
    # z = tf.expand_dims(z, -1)
    # z = tf.expand_dims(z, -1)
    # z = inception_module(z, 16, 32, 64, 8, 16, 16)
    # z = inception_module(z, 128, 128, 192, 32, 96, 64)
    # z = residual_module(z,64)
    # z=tf.keras.layers.Conv1D(50, 2, activation=activation)(z)
    # z=tf.keras.layers.MaxPooling1D(pool_size=2)(z)
    # z=tf.keras.layers.Conv1D(100, 2, activation=activation)(z)
    # z=tf.keras.layers.MaxPooling1D(pool_size=2)(z)
    z = tf.keras.layers.Flatten()(z)
    z = tf.keras.layers.Dropout(dropout)(z)

    x=tf.keras.layers.Dense(nNodes,**dense_kwargs)(z)
    for i in range(nDense-1):
        x=tf.keras.layers.Dense(nNodes,**dense_kwargs)(x)
        x=tf.keras.layers.Dropout(dropout)(x)

    # outputLayer = tf.keras.layers.Dense(20,activation='sigmoid')(z)
    outputLayer = tf.keras.layers.Dense(10,activation='sigmoid')(x)

    model = tf.keras.Model(inputs=[inputJets, inputOther], outputs=outputLayer, name="taggerModel")
    model.summary()

    return model

def getJetModel(regRate=1e-3,activation='selu',dropout=0.1,nDense=3,nNodes=10):

    l2_reg = tf.keras.regularizers.l2(regRate)
    dense_kwargs = dict(
        activation=activation,
        kernel_initializer=tf.keras.initializers.lecun_normal(),
        kernel_regularizer=l2_reg,
    )

    # inputCharged = tf.keras.layers.Input(shape=(None,7))
    # inputNeutral = tf.keras.layers.Input(shape=(None,7))
    # inputSV = tf.keras.layers.Input(shape=(None,7))
    # visible = keras.layers.Input(shape=(11,4))
    visible = tf.keras.layers.Input(shape=(None,4))
    # visible = tf.keras.layers.Input(shape=(None,4*28))
    # visible = tf.keras.layers.Input(shape=(4*28))
    # visible = tf.keras.layers.Input(shape=(4*38))

    z = tf.keras.layers.Masking(mask_value=0.)(visible)
    # z = keras.layers.LSTM(38, kernel_regularizer=L2reg)(z)
    # z = tf.keras.layers.LSTM(250, **dense_kwargs)(z)
    # z = tf.keras.layers.LSTM(150, **dense_kwargs)(z)
    # z = tf.keras.layers.LSTM(50, **dense_kwargs)(z)
    z = tf.keras.layers.LSTM(4, **dense_kwargs, return_sequences=True)(z)
    z = tf.keras.layers.LSTM(38, **dense_kwargs)(z)

    # from keras.layers.wrappers import Bidirectional
    # z = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, activation=None), input_shape=(256,10))(z)
    # z = tf.keras.layers.BatchNormalization()(z)

    # z = tf.keras.layers.LSTM(200, **dense_kwargs)(z)
    # z = tf.keras.layers.LSTM(200, **dense_kwargs)(visible)
    # z = tf.keras.layers.LSTM(100)(z)
    # z = tf.keras.layers.LSTM(100)(z)
    # z = tf.keras.layers.Dropout(dropout)(z)
    # z = tf.expand_dims(z, -1)
    # z = tf.expand_dims(z, -1)
    # z = inception_module(z, 16, 32, 64, 8, 16, 16)
    # z = inception_module(z, 128, 128, 192, 32, 96, 64)
    # z = residual_module(z,64)
    # z=tf.keras.layers.Conv1D(50, 2, activation=activation)(z)
    # z=tf.keras.layers.MaxPooling1D(pool_size=2)(z)
    # z=tf.keras.layers.Conv1D(100, 2, activation=activation)(z)
    # z=tf.keras.layers.MaxPooling1D(pool_size=2)(z)

    # z = tf.keras.layers.concatenate(xCharged)
    # z = tf.keras.layers.Flatten()(z)
    # z = tf.keras.layers.Dropout(dropout)(z)
    # x=tf.keras.layers.Dense(nNodes,**dense_kwargs)(visible)
    x=tf.keras.layers.Dense(nNodes,**dense_kwargs)(z)
    x=tf.keras.layers.Dropout(dropout)(x)
    for i in range(nDense-1):
        x=tf.keras.layers.Dense(nNodes,**dense_kwargs)(x)
        x=tf.keras.layers.Dropout(dropout)(x)
    # z = tf.keras.layers.Dropout(dropout)(z)

    # outputLayer = tf.keras.layers.Dense(1, activation='sigmoid')(z)
    hidden = tf.keras.layers.Dense(4,activation="linear")(x)
    # hidden = tf.keras.layers.Dense(4)(x)

    # keras.Model(inputs=[inputCharged, inputNeutral, inputSV], outputs=outputLayer, name="JetModel")
    model = tf.keras.Model(inputs=visible, outputs=hidden)
    model.summary()

    return model
