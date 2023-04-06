# *******************************
# *                             *
# *        Neural Network       *
# *                             *
# *******************************
def getModel():
    print ("getting model")
    model = getMRegModelFlat(regRate = regRate_, activation = activation_, dropout = dropout_, nDense = nDense_,
                                      nNodes = nNodes_, inputDim = inX_train.shape[1], outputActivation = outputActivation_)


    optimizer = keras.optimizers.Adam(   learning_rate = learningRate_,
                                            beta_1=0.9, beta_2=0.999,   # memory lifetime of the first and second moment
                                            epsilon=1e-07,              # regularization constant to avoid divergences
                                            #weight_decay=None,
                                            #use_ema=False, bema_momentum=0.99, ema_overwrite_frequency=None,
                                            name="Adam"
                                            )

    print ("compiling model")
    #model.compile(optimizer=optimizer, loss = keras.losses.MeanSquaredError(), metrics=['mean_absolute_error','mean_absolute_percentage_error'])
    model.compile(optimizer=optimizer, loss = keras.losses.MeanSquaredError())


    callbacks=[]
    #reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', patience=patiencelR_, factor=reduceLR_factor)
    earlyStop = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=patienceeS_, verbose = 1, restore_best_weights=True)
    #modelCheckpoint = keras.callbacks.ModelCheckpoint(outFolder + '/model_best.h5', monitor='val_loss', save_best_only=True)
    callbacks.append(earlyStop)
    #callbacks.append(modelCheckpoint)
    return model
