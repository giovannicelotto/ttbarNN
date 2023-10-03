# file to be suubmitted to condor repeated 100 times
# serves as initial points for the bayesian optimization
import os
import sys
import numpy as np

from utils.helpers import *
from utils.models import *
from utils.plot import *
from utils.stat import *
from utils.splitTrainValid import splitTrainValid
from utils.defName import getNamesForBayes
import matplotlib.pyplot as plt
import matplotlib
from array import array
from tensorflow import keras
from utils.data import *
from scipy.stats.stats import pearsonr
from scipy.stats import pearsonr
matplotlib.use('agg')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'        # TensorFlow will only display error messages and suppress all other messages including these warnings.


maxEvents_          = None
hp = {
    #'learningRate':     0.0001, 
    'batchSize':        128,
    'validBatchSize':   256,
    #'nNodes':           [32, 64, 128], #32, 64, 256 ,  21, 428, 276 -- 
    #'regRate':          0.01,
    'activation':       'elu',
    'scale' :           'standard',
    'nFiles':           3,
    'testFraction':     0.2,
    'validation_split': 0.2,
    'epochs':           10,
    'patienceeS':       150,
    #'alpha':            180,
    #'nNodes2':          [64, 32 ],
    #'learningRate2':    0.01,
    #'epochs2':          12,
    #'regRate2':         0.02
}


def objective(inX_train, outY_train, weights_train, inX_test, outY_test, weights_test, nNodes, activation, lasso, ridge, learningRate):
    
    dnnMask_train  = inX_train[:,0]>-998
    dnnMask_test  = inX_test[:,0]>-998
    
    
   
# Get the model, optmizer,         
    model = getModelRandom(lasso = lasso, ridge = ridge, activation = activation,
                        nDense = len(nNodes),  nNodes = nNodes,
                        inputDim = inX_train.shape[1], outputActivation = 'linear')
    optimizer = keras.optimizers.Adam(   learning_rate = learningRate, beta_1=0.9, beta_2=0.999, epsilon=1e-01, name="Adam")
    model.compile(optimizer=optimizer, loss = keras.losses.MeanSquaredError(), weighted_metrics=[])
    callbacks=[]
    earlyStop = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = hp['patienceeS'], verbose = 1, restore_best_weights=True)
    callbacks.append(earlyStop)
    
# Fit the model
    fit = model.fit(    x = inX_train[dnnMask_train, :], y = outY_train[dnnMask_train, :], batch_size = hp['batchSize'], epochs = hp['epochs'], verbose = 2,
                        callbacks = callbacks, validation_split = 0.2, # The validation data is selected from the last samples in the x and y data provided, before shuffling. 
                        validation_batch_size = hp['validBatchSize'],
                        shuffle = False, sample_weight = np.abs(weights_train[dnnMask_train]), initial_epoch=2)


    dnnMask_test = inX_test[:,0]>-998
    y_predicted__ = model.predict(inX_test[dnnMask_test,:])
    y_predicted = np.ones(len(inX_test))*-999
    y_predicted[dnnMask_test] = y_predicted__[:,0]
    assert ((y_predicted[dnnMask_test]>0).all()), "Predicted negative values in the first NN"


    #y_predicted_train, y_predicted_valid, y_predicted = computePredicted(inX_train[mask_train, :], inX_valid[mask_valid, :], inX_test[mask_test, :], npyDataFolder, model)
    
# SECOND NN
    dnn2Mask_train = inX_train[:,0]<-4998
    dnn2Mask_valid = inX_valid[:,0]<-4998
    dnn2Mask_test  = inX_test[:,0]<-4998
    assert ((outY_train[dnn2Mask_train, :]>0).all()), "outY_train used for the second training is wrong"
# Select only events that satisfy kinematic cuts (Njets, nbjets, passCuts) but for which the analytical solutions do not exist (or are not adequate)
    SinX_train       = inX_train[dnn2Mask_train,14:]
    Sweights_train   = weights_train[dnn2Mask_train]
    SinX_valid       = inX_valid[dnn2Mask_valid,14:]
    Sweights_valid   = weights_valid[dnn2Mask_valid]
    SinX_test        = inX_test[dnn2Mask_test, 14:]
    
# Get the model, optmizer, 
    
    simpleModel = getSimpleModel(regRate = hp['regRate2'], activation = 'elu',
                        nDense = len(hp['nNodes2']),  nNodes = hp['nNodes2'],
                        inputDim = SinX_train.shape[1], outputActivation = 'linear')
    optimizer = keras.optimizers.Adam(   learning_rate = hp['learningRate2'], beta_1=0.9, beta_2=0.999, epsilon=1e-01, name="Adam") 
    simpleModel.compile(optimizer=optimizer, loss = keras.losses.MeanSquaredError(), weighted_metrics=[])
    callbacks=[]
    earlyStop = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = hp['patienceeS'], verbose = 1, restore_best_weights=True)
    callbacks.append(earlyStop)
    
# Fit the model
    simpleFit = simpleModel.fit(    x = SinX_train[:, :], y = outY_train[dnn2Mask_train, :], batch_size = hp['batchSize'], epochs = hp['epochs2'], verbose = 2,
                        callbacks = callbacks, validation_batch_size = hp['validBatchSize'],
                        validation_data = (SinX_valid[:, :], outY_valid[dnn2Mask_valid, :], np.abs(Sweights_valid[:])),
                        shuffle = False, sample_weight = np.abs(Sweights_train[:]), initial_epoch=2)


# predict the valid samples and produce plots    
    y_predicted__ = simpleModel.predict(SinX_test[:,:])
    y_predicted[dnn2Mask_test] = y_predicted__[:,0]
    assert ((y_predicted[dnn2Mask_test]>0).all()), "Predicted negative values in the Second NN"
    
    mse = mean_squared_error(outY_test[dnnMask_test ,:], y_predicted[dnnMask_test ], sample_weight = weights_test)            # | dnn2Mask_test
    return mse


def main():

    # Get paths of data based on nFiles and maxEvents
    npyDataFolder = getNamesForBayes(hp['nFiles'], maxEvents_) #10*None
    print(npyDataFolder)
    outFolder = '/nfs/dust/cms/user/celottog/mttNN/outputs/tempRandom'+"/model"
    # Load scaled and splitted quantities.
    inX_train, inX_test, outY_train, outY_test, weights_train, weights_test, lkrM_train, lkrM_test, krM_train, krM_test, totGen_train, totGen_test, mask_train, mask_test= loadData(npyDataFolder = npyDataFolder,
                                                                            testFraction = hp['testFraction'],
                                                                            maxEvents = maxEvents_,
                                                                            minbjets = 1,
                                                                            nFiles = hp['nFiles'], scale=hp['scale'], outFolder = outFolder, output=False)
    dnnMask_train = inX_train[:,0]>-998
    dnn2Mask_train = inX_train[:,0]<-4998
    tau = np.random.normal(loc = 174, scale = 5)
    weights_train[dnnMask_train], weights_train_original = getWeightsTrain(outY_train[dnnMask_train], weights_train[dnnMask_train], outFolder=outFolder, alpha = tau, output=False)
    weights_train[dnn2Mask_train], weights_train_original = getWeightsTrain(outY_train[dnn2Mask_train], weights_train[dnn2Mask_train], outFolder=outFolder, alpha = tau, output=False)
    
# nNode generation
    learningRate = 0.000467
    activation = hp['activation']
    nNodes = [23, 26, 60]
    lasso  = 0.14874
    ridge  = 0.00307
    det = []
    for i in range(3):
        det__ = objective(inX_train = inX_train, outY_train = outY_train, weights_train = weights_train, inX_test = inX_test, outY_test = outY_test, weights_test = weights_test,
                          nNodes=nNodes, activation=activation, lasso=lasso, ridge = ridge, learningRate = learningRate)
        det.append(det__)

    det = np.array(det)
    with open("/nfs/dust/cms/user/celottog/mttNN/outputs/comparisons/newRandomWeight.txt", "a+") as f:
                string = "\n{:.3f}\t{:.5f}\t".format(det.mean(), det.std())
                string = string + str(nNodes) + "\t%.5f"%ridge + "\t%.1f"%tau
                

                f.write(string)



if __name__ == "__main__":
    main()


'''
For generating lognormal distribution
mean = 0.01
sigma = 0.01
mu = np.log(mean)
sigma_norm = np.sqrt(np.log(1 + (sigma**2 / mean**2)))
num = np.random.normal(mu, sigma_norm)'''