import tensorflow as tf
from tensorflow.keras.layers import Layer


'''def getMRegModel(regRate=1e-3,activation='selu',dropout=0.1,nDense=5,nNodes=200):
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

    return model'''

#from tensorflow.python.keras import backend as K
#from tensorflow.python.ops import array_ops

def getMRegModelFlat(regRate, activation, dropout, nDense, nNodes, inputDim, outputActivation, printSummary=True):
    l2_reg = tf.keras.regularizers.l2(regRate)		#cost = oroginal cost function + REGRATE * sum of squared weights (we want weights to be as small as possible)
# large weights are penalized with L2 with respect to L1

    dense_kwargs = dict(
        kernel_initializer = tf.keras.initializers.glorot_normal(),             # initializes the weights of the dense layer with Glorot normal distribution, which is a commonly used weight initialization method for deep learning models.
        kernel_regularizer = l2_reg,                                            # using L2 regularization
        kernel_constraint = tf.keras.constraints.max_norm(5)                    # the max_norm constraint is used to limit the maximum norm of the weight vector to 5,
    )
# Input Layer
    inputs = tf.keras.layers.Input(shape = (inputDim))
    x = tf.keras.layers.BatchNormalization()(inputs)
# First Dense Layer with activation function, batch normalization and drop out
    #x = tf.keras.layers.Dense(nNodes, **dense_kwargs)(x)
    #x = tf.keras.layers.Activation(activation)(x)
    #x = tf.keras.layers.BatchNormalization()(x)
    #x = tf.keras.layers.Dropout(dropout)(x)
    for i in range(nDense):				#never takes place if nDense = 2
        x = tf.keras.layers.Dense(nNodes, **dense_kwargs)(x)
        x = tf.keras.layers.Activation(activation)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout)(x)

#GC    x = tf.keras.layers.Dense(nNodes, **dense_kwargs)(x)
#GC    x = tf.keras.layers.Activation(activation)(x)
# Output layer
    outputLayer = tf.keras.layers.Dense(1, activation = outputActivation, kernel_constraint=tf.keras.constraints.non_neg())(x)

    model = tf.keras.Model(inputs = inputs, outputs = outputLayer, name = "MRegModelFlat")
    if printSummary:
        model.summary()

    return model

def baseline_model(inputDim):
 # create model
 model = tf.keras.Sequential()
 model.add(tf.keras.Dense(inputDim, input_shape=(inputDim,), kernel_initializer='normal', activation='relu'))
 model.add(tf.keras.Dense(1, kernel_initializer='normal'))
 # Compile model
 model.compile(loss='mean_squared_error', optimizer='adam')
 return model