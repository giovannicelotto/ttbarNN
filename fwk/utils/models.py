import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import ROOT

import os
import tensorflow as tf
from tensorflow.keras.layers import Layer
# from tensorflow.keras.engine import Layer
# from tensorflow.layers import Layer
import keras.backend as K
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
from tensorflow_smearing import network as smearLayer

def residual_module_dense(layer_in, n_filters, activation, dense_kwargs):
    merge_input = layer_in
    # check if the number of filters needs to be increase, assumes channels last format
    # if layer_in.shape[-1] != n_filters:
    # 	merge_input = tf.keras.layers.Conv1D(n_filters, 1, padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
    # conv1
    dense1 = tf.keras.layers.Dense(n_filters,**dense_kwargs)(layer_in)
    # dense1 = tf.keras.layers.LeakyReLU()(dense1)
    dense1 =tf.keras.layers.Activation(activation)(dense1)
    dense1 = tf.keras.layers.BatchNormalization()(dense1)
    # conv2
    dense2 = tf.keras.layers.Dense(n_filters,**dense_kwargs)(dense1)
    # dense2 = tf.keras.layers.LeakyReLU()(dense2)
    dense2 =tf.keras.layers.Activation(activation)(dense2)
    dense2 = tf.keras.layers.BatchNormalization()(dense2)
    # add filters, assumes filters/channels last
    layer_out = tf.keras.layers.add([dense2, merge_input])
    # activation function
    layer_out = tf.keras.layers.Activation(activation)(layer_out)
    # layer_out = tf.keras.layers.LeakyReLU()(layer_out)
    return layer_out

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

from tensorflow.python.keras import backend as K
from tensorflow.python.ops import array_ops
class SmearLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, nSmear, **kwargs):
        self.nSmear = nSmear
        self.input_dim = input_dim
        super(SmearLayer, self).__init__(**kwargs)
        # mean_init = tf.random_normal_initializer()
        mean_init = tf.constant_initializer(5.)
        self.m = tf.Variable(
            initial_value=mean_init(shape=(self.input_dim,), dtype="float32"),
            trainable=True, validate_shape=True
        )
        # std_init = tf.random_normal_initializer()
        std_init = tf.constant_initializer(0.1)
        self.s = tf.Variable(
            initial_value=std_init(shape=(self.input_dim,), dtype="float32"),
            trainable=True, validate_shape=True
        )

    def build(self, input_shape):
        self.batch_size = input_shape[0]
        super(SmearLayer, self).build(input_shape)

    def call(self, inputs):
        output = inputs
        print ("initial output",output)
        # output = array_ops.reshape(output,(array_ops.shape(output)[0]+1,))
        output = tf.expand_dims(output, -1)
        print ("expand dim output",output)
        output = K.repeat_elements(output, self.nSmear, axis=-1)
        print ("repeated output",output)
        # m = tf.expand_dims(self.m, -1)
        # s = tf.expand_dims(self.s, -1)
        # m = K.repeat_elements(m, self.nSmear, axis=-1)
        # s = K.repeat_elements(s, self.nSmear, axis=-1)
        # print ("extended means", m)
        # print ("extended sigmas", s)
        test = K.random_normal(shape=array_ops.shape(self.input_dim), mean=self.m, stddev=self.s)
        # output = output * K.random_normal(shape=array_ops.shape(output), mean=m, stddev=s)
        print ("final output",test)
        # print ("final output",output)
        return output

def getRhoRegModelFlat_new(n_particles,regRate=1e-3,batch_size=64, activation='selu', dropout=0.1, nDense=5, nNodes=200, inputDim = 21, outputActivation = 'linear'):
    l2_reg = tf.keras.regularizers.l2(regRate)
    dense_kwargs = dict(
        # activation = activation,
        # kernel_initializer = tf.keras.initializers.lecun_normal(),
        kernel_initializer = tf.keras.initializers.glorot_normal(),
        kernel_regularizer = l2_reg,
    )
    # inputs = tf.keras.layers.Input(shape = (inputDim))
    smearing_layer = smearLayer.NoiseSmearingLayer(built_in_noise=True)
    # input_particles = tf.keras.layers.Input(shape=(n_particles, 4), batch_size=batch_size)
    input_particles = tf.keras.layers.Input(shape=(n_particles, 4))
    inp_tensors = [input_particles]
    x = smearing_layer(input_particles)

    # x = tf.keras.layers.Flatten()(x)
    from lbn import LBN, LBNLayer
    lbn_layer = LBNLayer((6, 4), n_particles=9, boost_mode=LBN.PAIRS,
        features=["E", "px", "py", "pz"])

    x = lbn_layer(x)
    # x = lbn_layer(input_particles)

    for i in range(nDense-2):
        x = tf.keras.layers.Dense(nNodes, **dense_kwargs)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(nNodes, **dense_kwargs)(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    output = tf.keras.layers.Dense(1, activation = outputActivation)(x)

    model = tf.keras.Model(inputs = inp_tensors, outputs = output, name = "getRhoRegModelFlat_new")
    model.summary()

    return model

def getRhoRegModelFlat(regRate=1e-3, activation='selu', dropout=0.1, nDense=5, nNodes=200, inputDim = 21, outputActivation = 'sigmoid', printSummary=True):
    l2_reg = tf.keras.regularizers.l2(regRate)
    dense_kwargs = dict(
        # activation = activation,
        # kernel_initializer = tf.keras.initializers.lecun_normal(),
        kernel_initializer = tf.keras.initializers.glorot_normal(),
        kernel_regularizer = l2_reg,
        kernel_constraint = tf.keras.constraints.max_norm(5)
    )
    inputs = tf.keras.layers.Input(shape = (inputDim))
    # print (inputs)
    x = tf.keras.layers.BatchNormalization()(inputs)
    # x = tf.keras.layers.Dense(nNodes*nDense, **dense_kwargs)(inputs)
    # x = tf.keras.layers.Dense(nNodes*nDense, **dense_kwargs)(x)
    x = tf.keras.layers.Dense(nNodes, **dense_kwargs)(x)
    # x = tf.keras.layers.Dense(nNodes, **dense_kwargs)(inputs)
    # x = tf.keras.layers.Dense(nNodes, **dense_kwargs)(inputs)
    # x =tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x =tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # if activation==""
    # x = tf.keras.layers.Activation(activation)(x)
    # x =tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    # x =tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    for i in range(nDense-2):
        # x = tf.keras.layers.Dense(nNodes*(nDense-i-1), **dense_kwargs)(x)
        x = tf.keras.layers.Dense(nNodes, **dense_kwargs)(x)
        # x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        x = tf.keras.layers.Activation(activation)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.Activation(activation)(x)
        # x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        # x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(nNodes, **dense_kwargs)(x)
    # x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Activation(activation)(x)
    # outputLayer = tf.keras.layers.Dense(1, activation = outputActivation)(x)
    outputLayer = tf.keras.layers.Dense(1, activation = outputActivation, kernel_constraint=tf.keras.constraints.non_neg())(x)
    # outputLayer = tf.keras.layers.Dense(3, activation = outputActivation)(x)
    # outputLayer = tf.keras.layers.Dense(1,activation='sigmoid')(x)
    # outputLayer = tf.keras.layers.Dense(1,activation='linear')(x)
    # outputLayer = tf.keras.layers.Dense(1,activation='relu')(x)

    model = tf.keras.Model(inputs = inputs, outputs = outputLayer, name = "RhoRegModelFlat")
    if printSummary:
        model.summary()

    return model

def getRhoClassModelFlat(regRate=1e-3, activation='selu', dropout=0.1, nDense=5, nNodes=200, inputDim = 21, outputActivation = 'sigmoid'):
    l2_reg = tf.keras.regularizers.l2(regRate)
    dense_kwargs = dict(
        # activation = activation,
        # kernel_initializer = tf.keras.initializers.glorot_normal(),
        # kernel_regularizer = l2_reg,
        kernel_initializer = tf.keras.initializers.glorot_normal(),
        kernel_regularizer =  tf.keras.regularizers.l2(regRate),
        kernel_constraint = tf.keras.constraints.max_norm(5)
    )
    inputs = tf.keras.layers.Input(shape = (inputDim))
    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.Dense(nNodes, activation=activation, **dense_kwargs)(inputs)
    # x = tf.keras.layers.Activation(activation)(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    for i in range(nDense-2):
        # x = tf.keras.layers.Dense(nNodes*(nDense-i-1), **dense_kwargs)(x)
        x = tf.keras.layers.Dense(nNodes, activation=activation, **dense_kwargs)(x)
        # x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        # x = tf.keras.layers.Activation(activation)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.Activation(activation)(x)
        # x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        # x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(nNodes, activation=activation, **dense_kwargs)(x)
    # x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    # x = tf.keras.layers.Activation(activation)(x)
    outputLayer = tf.keras.layers.Dense(4, activation = outputActivation)(x)
    # outputLayer = tf.keras.layers.Dense(1,activation='sigmoid')(x)
    # outputLayer = tf.keras.layers.Dense(1,activation='linear')(x)
    # outputLayer = tf.keras.layers.Dense(1,activation='relu')(x)

    model = tf.keras.Model(inputs = inputs, outputs = outputLayer, name = "RhoRegModelFlat")
    model.summary()

    return model

def getClassificationModel(outputNodes,regRate=1e-3, activation='selu', dropout=0.1, nDense=5, nNodes=200, inputDim = 21, outputActivation = 'sigmoid', printSummary=True):
    dense_kwargs = dict(
        kernel_initializer = tf.keras.initializers.glorot_normal(),
        kernel_regularizer =  tf.keras.regularizers.l2(regRate),
        kernel_constraint = tf.keras.constraints.max_norm(5)
    )
    inputs = tf.keras.layers.Input(shape = (inputDim), name="input")
    x = tf.keras.layers.BatchNormalization(name="batchNorm_1")(inputs)
    # x = tf.keras.layers.Dense(nNodes, **dense_kwargs)(inputs)
    x = tf.keras.layers.Dense(nNodes, activation=activation, name="dense_1", **dense_kwargs)(x)
    # x =tf.keras.layers.Activation(activation, name="activation_1")(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout_1")(x)
    for i in range(nDense-2):
        # x = tf.keras.layers.Dense(nNodes*(nDense-i-1), name="dense_"+str(i+2), **dense_kwargs)(x)
        x = tf.keras.layers.Dense(nNodes, activation=activation, name="dense_"+str(i+2), **dense_kwargs)(x)
        # x = tf.keras.layers.Dense(int(nNodes*0.5**(i+1)), **dense_kwargs)(x)
        # x = tf.keras.layers.Dense(nNodes, **dense_kwargs)(x)
        # x = residual_module_dense(x, nNodes, activation, dense_kwargs)
        # x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        # x = tf.keras.layers.Activation(activation, name="activation_"+str(i+2))(x)
        x = tf.keras.layers.BatchNormalization(name="batchNorm_"+str(i+2))(x)
        # x = tf.keras.layers.Activation(activation)(x)
        # x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        # x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(dropout, name="dropout_"+str(i+2))(x)
    x = tf.keras.layers.Dense(nNodes, activation=activation, name="dense_"+str(nDense), **dense_kwargs)(x)
    # x = tf.keras.layers.Dense(int(nNodes*0.5**(nDense-1)), **dense_kwargs)(x)
    # x = residual_module_dense(x, nNodes, activation, dense_kwargs)
    # x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    # x = tf.keras.layers.Activation(activation, name="activation_"+str(nDense))(x)
    outputLayer = tf.keras.layers.Dense(outputNodes, activation = outputActivation, name="output")(x)
    # outputLayer = tf.keras.layers.Dense(1,activation='sigmoid')(x)
    # outputLayer = tf.keras.layers.Dense(1,activation='linear')(x)
    # outputLayer = tf.keras.layers.Dense(1,activation='relu')(x)

    model = tf.keras.Model(inputs = inputs, outputs = outputLayer, name = "SimpleMultiClassifier")
    if printSummary:
        model.summary()

    return model

def getClassificationModelWithInvariance(outputNodes,regRate=1e-3, activation='selu', dropout=0.1, nDense=5, nNodes=200, inputDim = 21, outputActivation = 'sigmoid', printSummary=True, hp_lambda=1.):
    dense_kwargs = dict(
        kernel_initializer = tf.keras.initializers.glorot_normal(),
        kernel_regularizer =  tf.keras.regularizers.l2(regRate),
        kernel_constraint = tf.keras.constraints.max_norm(5)
    )

    inputs = tf.keras.layers.Input(shape = (inputDim), name="input")
    x = tf.keras.layers.BatchNormalization(name="batchNorm_1")(inputs)
    x = tf.keras.layers.Dense(nNodes, activation=activation, name="dense_1", **dense_kwargs)(x)
    # x =tf.keras.layers.Activation(activation, name="activation_1")(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout_1")(x)
    for i in range(nDense-2):
        # x = tf.keras.layers.Dense(nNodes*(nDense-i-1), name="dense_"+str(i+2), **dense_kwargs)(x)
        x = tf.keras.layers.Dense(nNodes, activation=activation, name="dense_"+str(i+2), **dense_kwargs)(x)
        # x = tf.keras.layers.Activation(activation, name="activation_"+str(i+2))(x)
        x = tf.keras.layers.BatchNormalization(name="batchNorm_"+str(i+2))(x)
        x = tf.keras.layers.Dropout(dropout, name="dropout_"+str(i+2))(x)
    x = tf.keras.layers.Dense(nNodes, activation=activation, name="dense_"+str(nDense), **dense_kwargs)(x)
    # x1 = tf.keras.layers.Activation(activation, name="activation_"+str(nDense))(x)
    outputLayer = tf.keras.layers.Dense(outputNodes, activation = outputActivation, name="output")(x)

    y = GradientReversal(hp_lambda=hp_lambda, name="gradient_reversal")(x)
    # y = GradReverse()(x)
    y = tf.keras.layers.Dense(nNodes, activation=activation, name="gradRev_dense_1")(y)
    y = tf.keras.layers.BatchNormalization(name="gradRev_batchNorm_1")(y)
    y = tf.keras.layers.Dropout(rate=dropout, name="gradRev_dropout_1")(y)
    # y = tf.keras.layers.Dense(nNodes, activation=activation, name="gradRev_dense_2")(y)
    # y = tf.keras.layers.BatchNormalization(name="gradRev_batchNorm_2")(y)
    # y = tf.keras.layers.Dropout(rate=dropout, name="gradRev_dropout_2")(y)
    # y = tf.keras.layers.Dense(nNodes, activation=activation, name="gradRev_dense_3")(y)
    # y = tf.keras.layers.BatchNormalization(name="gradRev_batchNorm_3")(y)
    # y = tf.keras.layers.Dropout(rate=dropout, name="gradRev_dropout_3")(y)
    y = tf.keras.layers.Dense(nNodes, activation=activation, name="gradRev_dense_4")(y)
    gradient_reversal = tf.keras.layers.Dense(1, activation='linear', name="gradRev_output")(y)


    model = tf.keras.Model(inputs = inputs, outputs = [outputLayer,gradient_reversal], name = "SimpleMultiClassifierWithInvariance")
    if printSummary:
        model.summary()

    return model

def getRhoRegModelFlat_res(regRate=1e-3, activation='selu', dropout=0.1, nDense=5, nNodes=200, inputDim = 21, outputActivation = 'sigmoid'):
    l2_reg = tf.keras.regularizers.l2(regRate)
    dense_kwargs = dict(
        kernel_initializer = tf.keras.initializers.glorot_normal(),
        kernel_regularizer = l2_reg,
    )
    inputs = tf.keras.layers.Input(shape = (inputDim))
    x = tf.keras.layers.Dense(nNodes, **dense_kwargs)(inputs)
    x = tf.keras.layers.LeakyReLU()(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = residual_module_dense(x, nNodes, dense_kwargs)
    x = tf.keras.layers.Dropout(dropout)(x)
    for i in range(nDense-1):
        # x = tf.keras.layers.Dense(nNodes, **dense_kwargs)(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        # x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
        x = residual_module_dense(x, nNodes, dense_kwargs)
        x = tf.keras.layers.Dropout(dropout)(x)

    outputLayer = tf.keras.layers.Dense(1, activation = outputActivation)(x)
    # outputLayer = tf.keras.layers.Dense(1,activation='linear')(x)

    model = tf.keras.Model(inputs = inputs, outputs = outputLayer, name = "RhoRegModelFlat")
    model.summary()

    return model

def getMassRegModelFlat(regRate=1e-3, activation='selu', dropout=0.1, nDense=5, nNodes=200, inputDim = 21, outputActivation = 'sigmoid'):
    l2_reg = tf.keras.regularizers.l2(regRate)
    dense_kwargs = dict(
        activation = activation,
        kernel_initializer = tf.keras.initializers.lecun_normal(),
        kernel_regularizer = l2_reg,
    )
    inputs = tf.keras.layers.Input(shape = (inputDim))

    x = tf.keras.layers.Dense(nNodes, **dense_kwargs)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    for i in range(nDense-1):
        x = tf.keras.layers.Dense(nNodes, **dense_kwargs)(x)
        x = tf.keras.layers.Dropout(dropout)(x)

    outputLayer = tf.keras.layers.Dense(1, activation = outputActivation)(x)

    model = tf.keras.Model(inputs = inputs, outputs = outputLayer, name = "MassRegModelFlat")

    return model


def getTaggerModelFlat(regRate=1e-3, activation='selu', dropout=0.1, nDense=5, nNodes=200, inputDim = 36):
    l2_reg = tf.keras.regularizers.l2(regRate)
    dense_kwargs = dict(
        activation=activation,
        kernel_initializer=tf.keras.initializers.lecun_normal(),
        kernel_regularizer=l2_reg,
    )
    inputs = tf.keras.layers.Input(shape=(inputDim))
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


# import tensorflow as tf
# import tensorflow.keras as k
# from tensorflow.keras.layers import Layer

# global_layer_list = {}

# def flip_gradient(x, l=1.0):
# 	positive_path = tf.stop_gradient(x * tf.cast(1 + l, tf.float32))
# 	negative_path = -x * tf.cast(l, tf.float32)
# 	return positive_path + negative_path

def reverse_gradient(X, hp_lambda):
    '''Flips the sign of the incoming gradient during training.'''
    try:
        reverse_gradient.num_calls += 1
    except AttributeError:
        reverse_gradient.num_calls = 1

    grad_name = "GradientReversal%d" % reverse_gradient.num_calls

    @tf.RegisterGradient(grad_name)
    def _flip_gradients(op, grad):
        return [tf.negative(grad) * hp_lambda]

    # g = k.backend.get_session().graph
    g = tf.compat.v1.keras.backend.get_session().graph
    with g.gradient_override_map({'Identity': grad_name}):
        y = tf.identity(X)
    return y

class GradientReversal(Layer):
    '''Flip the sign of gradient during training.'''

    def __init__(self, hp_lambda=1., **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.supports_masking = False
        self.hp_lambda = hp_lambda

    def build(self, input_shape):
       self._trainable_weights = []

    def call(self, x, mask=None):
        return reverse_gradient(x, self.hp_lambda)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'hp_lambda': self.hp_lambda}
        base_config = super(GradientReversal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



# global_layer_list['GradientReversal'] = GradientReversal
#
@tf.custom_gradient
def grad_reverse(x):
    y = tf.identity(x)
    def custom_grad(dy):
        return -dy
    return y, custom_grad
#
class GradReverse(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x):
        return grad_reverse(x)
