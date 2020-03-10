import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import ROOT

import os
import tensorflow as tf
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras

def getRhoRegModel(regRate=1e-3,activation='selu',dropout=0.1,nDense=5,nNodes=200):
    l2_reg = tf.keras.regularizers.l2(regRate)
    dense_kwargs = dict(
        activation=activation,
        kernel_initializer=tf.keras.initializers.lecun_normal(),
        kernel_regularizer=l2_reg,
    )
    cellNumber = nNodes

    ##Input layer
    inputJets = tf.keras.layers.Input(shape=(None,5))
    inputOther = tf.keras.layers.Input(shape=(None,4))

    xJets = LSTM(inputJets, cellNumber, dropout, dense_kwargs)
    xOther = LSTM(inputOther, cellNumber, dropout, dense_kwargs)


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
        kernel_initializer=tf.keras.initializers.lecun_normal(),
        kernel_regularizer=l2_reg,
    )
    inputs = tf.keras.layers.Input(shape=(21))

    x=tf.keras.layers.Dense(nNodes,**dense_kwargs)(inputs)
    x=tf.keras.layers.BatchNormalization()(x)
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
    x=tf.keras.layers.Dense(nNodes,**dense_kwargs)(inputs)
    x=tf.keras.layers.Dropout(dropout)(x)
    for i in range(nDense-1):
        x=tf.keras.layers.Dense(nNodes,**dense_kwargs)(x)
        x=tf.keras.layers.Dropout(dropout)(x)

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
