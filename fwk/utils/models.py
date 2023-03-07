import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import ROOT

import os
import tensorflow as tf
from tensorflow.keras.layers import Layer
import keras.backend as K
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
#from tensorflow_smearing import network as smearLayer

def residual_module_dense(layer_in, n_filters, activation, dense_kwargs):
    merge_input = layer_in
    # check if the number of filters needs to be increase, assumes channels last format
    # if layer_in.shape[-1] != n_filters:
    # 	merge_input = tf.keras.layers.Conv1D(n_filters, 1, padding='same', activation='relu', kernel_initializer='he_normal')(layer_in)
    # conv1
    dense1 = tf.keras.layers.Dense(n_filters,**dense_kwargs)(layer_in)
    dense1 =tf.keras.layers.Activation(activation)(dense1)
    dense1 = tf.keras.layers.BatchNormalization()(dense1)
    # conv2
    dense2 = tf.keras.layers.Dense(n_filters,**dense_kwargs)(dense1)
    dense2 =tf.keras.layers.Activation(activation)(dense2)
    dense2 = tf.keras.layers.BatchNormalization()(dense2)
    # add filters, assumes filters/channels last
    layer_out = tf.keras.layers.add([dense2, merge_input])
    # activation function
    layer_out = tf.keras.layers.Activation(activation)(layer_out)
    return layer_out

def getMRegModel(regRate=1e-3,activation='selu',dropout=0.1,nDense=5,nNodes=200):
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

    outputLayer = tf.keras.layers.Dense(1,activation='relu')(x)

    model = tf.keras.Model(inputs=[inputJets, inputOther], outputs=outputLayer, name="taggerModel")
    model.summary()

    return model

from tensorflow.python.keras import backend as K
from tensorflow.python.ops import array_ops

def getMRegModelFlat(regRate=1e-3, activation='selu', dropout=0.1, nDense=5, nNodes=200, inputDim = 21, outputActivation = 'sigmoid', printSummary=True):
    l2_reg = tf.keras.regularizers.l2(regRate)		#cost = oroginal cost function + REGRATE * sum of squared weights (we want weights to be as small as possible)
# large weights are penalized with L2 with respect to L1

    dense_kwargs = dict(
        kernel_initializer = tf.keras.initializers.glorot_normal(),
        kernel_regularizer = l2_reg,
        kernel_constraint = tf.keras.constraints.max_norm(5)
    )
# Input Layer
    inputs = tf.keras.layers.Input(shape = (inputDim))
# Batch normalization layer: Layer that normalizes its inputs.
    x = tf.keras.layers.BatchNormalization()(inputs)
# First Dense Layer with activation function, batch normalization and drop out
    x = tf.keras.layers.Dense(nNodes, **dense_kwargs)(x)
    x = tf.keras.layers.Activation(activation)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    for i in range(nDense-2):				#never takes place if nDense = 2
        x = tf.keras.layers.Dense(nNodes, **dense_kwargs)(x)
        x = tf.keras.layers.Activation(activation)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout)(x)
# Dropout is a technique where randomly selected neurons are ignored during training.
# Second Dense layer: Q? Why don't we put a batch normalization? Isn't it needed in the ouput? Why not a dropout?
    x = tf.keras.layers.Dense(nNodes, **dense_kwargs)(x)
    x = tf.keras.layers.Activation(activation)(x)
# Output layer
    outputLayer = tf.keras.layers.Dense(1, activation = outputActivation, kernel_constraint=tf.keras.constraints.non_neg())(x)

    model = tf.keras.Model(inputs = inputs, outputs = outputLayer, name = "MRegModelFlat")
    if printSummary:
        model.summary()

    return model
'''
def getRhoClassModelFlat(regRate=1e-3, activation='selu', dropout=0.1, nDense=5, nNodes=200, inputDim = 21, outputActivation = 'sigmoid'):
    l2_reg = tf.keras.regularizers.l2(regRate)
    dense_kwargs = dict(
        kernel_initializer = tf.keras.initializers.glorot_normal(),
        kernel_regularizer =  tf.keras.regularizers.l2(regRate),
        kernel_constraint = tf.keras.constraints.max_norm(5)
    )
    inputs = tf.keras.layers.Input(shape = (inputDim))
    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.Dense(nNodes, activation=activation, **dense_kwargs)(inputs)
    x = tf.keras.layers.Dropout(dropout)(x)
    for i in range(nDense-2):
        x = tf.keras.layers.Dense(nNodes, activation=activation, **dense_kwargs)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(nNodes, activation=activation, **dense_kwargs)(x)
    outputLayer = tf.keras.layers.Dense(4, activation = outputActivation)(x)

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
    x = tf.keras.layers.Dense(nNodes, activation=activation, name="dense_1", **dense_kwargs)(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout_1")(x)
    for i in range(nDense-2):
        x = tf.keras.layers.Dense(nNodes, activation=activation, name="dense_"+str(i+2), **dense_kwargs)(x)
        x = tf.keras.layers.BatchNormalization(name="batchNorm_"+str(i+2))(x)
        x = tf.keras.layers.Dropout(dropout, name="dropout_"+str(i+2))(x)
    x = tf.keras.layers.Dense(nNodes, activation=activation, name="dense_"+str(nDense), **dense_kwargs)(x)
    outputLayer = tf.keras.layers.Dense(outputNodes, activation = outputActivation, name="output")(x)

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
    x = tf.keras.layers.Dropout(dropout, name="dropout_1")(x)
    for i in range(nDense-2):
        x = tf.keras.layers.Dense(nNodes, activation=activation, name="dense_"+str(i+2), **dense_kwargs)(x)
        x = tf.keras.layers.BatchNormalization(name="batchNorm_"+str(i+2))(x)
        x = tf.keras.layers.Dropout(dropout, name="dropout_"+str(i+2))(x)
    x = tf.keras.layers.Dense(nNodes, activation=activation, name="dense_"+str(nDense), **dense_kwargs)(x)
    outputLayer = tf.keras.layers.Dense(outputNodes, activation = outputActivation, name="output")(x)

    y = GradientReversal(hp_lambda=hp_lambda, name="gradient_reversal")(x)
    y = tf.keras.layers.Dense(nNodes, activation=activation, name="gradRev_dense_1")(y)
    y = tf.keras.layers.BatchNormalization(name="gradRev_batchNorm_1")(y)
    y = tf.keras.layers.Dropout(rate=dropout, name="gradRev_dropout_1")(y)
    y = tf.keras.layers.Dense(nNodes, activation=activation, name="gradRev_dense_4")(y)
    gradient_reversal = tf.keras.layers.Dense(1, activation='linear', name="gradRev_output")(y)


    model = tf.keras.Model(inputs = inputs, outputs = [outputLayer,gradient_reversal], name = "SimpleMultiClassifierWithInvariance")
    if printSummary:
        model.summary()

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

    # residual_module_new

    z = tf.keras.layers.Add()([xJets, xOther])
    z = tf.keras.layers.Flatten()(z)
    z = tf.keras.layers.Dropout(dropout)(z)

    x=tf.keras.layers.Dense(nNodes,**dense_kwargs)(z)
    for i in range(nDense-1):
        x=tf.keras.layers.Dense(nNodes,**dense_kwargs)(x)
        x=tf.keras.layers.Dropout(dropout)(x)

    outputLayer = tf.keras.layers.Dense(10,activation='sigmoid')(x)

    model = tf.keras.Model(inputs=[inputJets, inputOther], outputs=outputLayer, name="taggerModel")
    model.summary()

    return model

'''
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


@tf.custom_gradient
def grad_reverse(x):
    y = tf.identity(x)
    def custom_grad(dy):
        return -dy
    return y, custom_grad

class GradReverse(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x):
        return grad_reverse(x)
