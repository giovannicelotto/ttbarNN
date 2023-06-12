import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow import keras
from sklearn.metrics import mean_squared_error

def getModel(regRate, activation, nDense, nNodes, inputDim, outputActivation='linear', printSummary=True):
    l2_reg = tf.keras.regularizers.l2(regRate)
    
    dense_kwargs = dict(
        kernel_initializer = tf.keras.initializers.glorot_normal( seed=1999),            
        kernel_regularizer = l2_reg,                                           
        #kernel_constraint = tf.keras.constraints.max_norm(4)                    # the max_norm constraint is used to limit the maximum norm of the weight vector
    )

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape = (inputDim))) 
    #model.add(tf.keras.layers.Dropout(0.2))
    for i in range(nDense):				
        model.add(tf.keras.layers.Dense(nNodes[i], **dense_kwargs))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation(activation))
        #model.add(tf.keras.layers.Dropout(0.5))
    

    model.add(tf.keras.layers.Dense(1, activation = outputActivation))

    #model = tf.keras.Model(inputs = inputs, outputs = outputLayer, name = "MRegModelFlat")
    if printSummary:
        model.summary()

    return model


def getModelRandom(lasso, ridge, activation, nDense, nNodes, inputDim, outputActivation='linear', printSummary=True):
    reg = tf.keras.regularizers.L1L2(l1 = lasso, l2 = ridge)
    
    dense_kwargs = dict(
        kernel_initializer = tf.keras.initializers.glorot_normal( seed=1999),            
        kernel_regularizer = reg,                                           
        #kernel_constraint = tf.keras.constraints.max_norm(4)                    # the max_norm constraint is used to limit the maximum norm of the weight vector
    )

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape = (inputDim))) 
    #model.add(tf.keras.layers.Dropout(0.2))
    for i in range(nDense):				
        model.add(tf.keras.layers.Dense(nNodes[i], **dense_kwargs))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation(activation))

    model.add(tf.keras.layers.Dense(1, activation = outputActivation))

    if printSummary:
        model.summary()

    return model


def getSimpleModel(lasso, ridge, activation, nDense=2, nNodes=32, inputDim=58, outputActivation='linear', printSummary=True):
    reg = tf.keras.regularizers.L1L2(l1 = lasso, l2 = ridge)
    
    dense_kwargs = dict(
        kernel_initializer = tf.keras.initializers.glorot_normal( seed=1999),        
        kernel_regularizer = reg,                                           
        #kernel_constraint = tf.keras.constraints.max_norm(4)                    # the max_norm constraint is used to limit the maximum norm of the weight vector
    )

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape = (inputDim))) 
    #model.add(tf.keras.layers.Dropout(0.2))
    for i in range(nDense):				
        model.add(tf.keras.layers.Dense(nNodes[i], **dense_kwargs))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation(activation))
        #model.add(tf.keras.layers.Dropout(0.5))
    

    model.add(tf.keras.layers.Dense(1, activation = outputActivation))

    #model = tf.keras.Model(inputs = inputs, outputs = outputLayer, name = "MRegModelFlat")
    if printSummary:
        model.summary()

    return model

'''def getModelForBayes(regRate, activation, nDense, nNode1, nNode2, nNode3, inputDim, outputActivation='linear', printSummary=True):
    l2_reg = tf.keras.regularizers.l2(regRate)
    
    dense_kwargs = dict(
        kernel_initializer = tf.keras.initializers.glorot_normal( seed=1999),            
        kernel_regularizer = l2_reg,                                           
        kernel_constraint = tf.keras.constraints.max_norm(3)                    # the max_norm constraint is used to limit the maximum norm of the weight vector
    )

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape = (inputDim))) 
    nNodes = [nNode1, nNode2, nNode3]
    #
    for i in range(nDense):				#never takes place if nDense = 2
        model.add(tf.keras.layers.Dense(nNodes[i], **dense_kwargs))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation(activation))
    

    model.add(tf.keras.layers.Dense(1, activation = outputActivation, kernel_constraint=tf.keras.constraints.non_neg()))

    #model = tf.keras.Model(inputs = inputs, outputs = outputLayer, name = "MRegModelFlat")
    if printSummary:
        model.summary()

    return model

def getModelForBayesNew(regRate, activation, nDense, nNode1, nNode2, inputDim, outputActivation='linear', printSummary=True):
    l2_reg = tf.keras.regularizers.l2(regRate)
    
    dense_kwargs = dict(
        kernel_initializer = tf.keras.initializers.glorot_normal( seed=1999),            
        kernel_regularizer = l2_reg,                                           
        #kernel_constraint = tf.keras.constraints.max_norm(1)                    # the max_norm constraint is used to limit the maximum norm of the weight vector to 5,
    )

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape = (inputDim))) 
    nNodes = [nNode1, nNode2]
    for i in range(nDense):				#never takes place if nDense = 2
        model.add(tf.keras.layers.Dense(nNodes[i], **dense_kwargs))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation(activation))
    

    model.add(tf.keras.layers.Dense(1, activation = outputActivation, kernel_constraint=tf.keras.constraints.non_neg()))

    #model = tf.keras.Model(inputs = inputs, outputs = outputLayer, name = "MRegModelFlat")
    if printSummary:
        model.summary()

    return model'''


    
