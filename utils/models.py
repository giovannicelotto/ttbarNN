import tensorflow as tf
from tensorflow.keras.layers import Layer

def getMRegModelFlat(regRate, activation, dropout, nDense, nNodes, inputDim, outputActivation, printSummary=True):
    l2_reg = tf.keras.regularizers.l2(regRate)
    
    dense_kwargs = dict(
        kernel_initializer = tf.keras.initializers.glorot_normal( seed=1999),            
        kernel_regularizer = l2_reg,                                           
        kernel_constraint = tf.keras.constraints.max_norm(2)                    # the max_norm constraint is used to limit the maximum norm of the weight vector to 5,
    )

    #inputs = 
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape = (inputDim))) 
    model.add(tf.keras.layers.BatchNormalization())
    for i in range(nDense):				#never takes place if nDense = 2
        model.add(tf.keras.layers.Dense(nNodes, **dense_kwargs))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation(activation))
        #x=tf.keras.layers.Dropout(dropout)(x)

    model.add(tf.keras.layers.Dense(1, activation = outputActivation, kernel_constraint=tf.keras.constraints.non_neg()))

    #model = tf.keras.Model(inputs = inputs, outputs = outputLayer, name = "MRegModelFlat")
    if printSummary:
        model.summary()

    return model

#def baseline_model(inputDim):
 # create model
 #model = tf.keras.Sequential()
 #model.add(tf.keras.Dense(inputDim, input_shape=(inputDim,), kernel_initializer='normal', activation='relu'))
 #model.add(tf.keras.Dense(1, kernel_initializer='normal'))
 # Compile model
 #model.compile(loss='mean_squared_error', optimizer='adam')
 #return model