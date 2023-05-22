# file to be suubmitted to condor repeated 100 times
# serves as initial points for the bayesian optimization
import os
import sys
import numpy as np

from utils.helpers import *
from utils.models import getModelRandom
from utils.plot import getRecoBin
import matplotlib.pyplot as plt
import matplotlib
from array import array
from tensorflow import keras
from scipy.stats.stats import pearsonr
from scipy.stats import pearsonr
matplotlib.use('agg')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'        # TensorFlow will only display error messages and suppress all other messages including these warnings.



#learningRate =  0.0001
# Get paths of data based on nFiles and maxEvents
 #10*None

# Load scaled and splitted quantities.
def loadDataSaved(npyDataFolder):
    print("loading Data from : ", npyDataFolder, "...")
    inX_train = np.load(npyDataFolder     + "/testing/inX_train.npy")
    inX_valid = np.load(npyDataFolder     + "/testing/inX_valid.npy")
    inX_test = np.load(npyDataFolder      + "/testing/inX_test.npy")
    outY_train = np.load(npyDataFolder    + "/testing/outY_train.npy")
    outY_valid = np.load(npyDataFolder    + "/testing/outY_valid.npy")
    outY_test = np.load(npyDataFolder     + "/testing/outY_test.npy")
    weights_train = np.load(npyDataFolder + "/testing/weights_train.npy")
    weights_valid = np.load(npyDataFolder + "/testing/weights_valid.npy")
    weights_test = np.load(npyDataFolder  + "/testing/weights_test.npy")
    totGen_test = np.load(npyDataFolder   + "/testing/totGen_test.npy")

    print("data loaded succesfully")
    return inX_train, inX_valid, inX_test, outY_train, outY_valid, outY_test, weights_train, weights_valid, weights_test, totGen_test







def objective(inX_train, inX_valid, inX_test, outY_train, outY_valid, outY_test, weights_train, weights_valid, weights_test, totGen_test, nNodes, activation, lasso, ridge, learningRate, hp):
    
    dnnMask_train = inX_train[:,0]>-998
    dnnMask_valid = inX_valid[:,0]>-998
    
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
                        callbacks = callbacks, #validation_split = validation_split_, # The validation data is selected from the last samples in the x and y data provided, before shuffling. 
                        validation_data = (inX_valid[dnnMask_valid, :], outY_valid[dnnMask_valid, :], np.abs(weights_valid[dnnMask_valid])), validation_batch_size = hp['validBatchSize'],
                        shuffle = False, sample_weight = np.abs(weights_train[dnnMask_train]), initial_epoch=2)


    dnnMask_test = inX_test[:,0]>-998
    y_predicted__ = model.predict(inX_test[dnnMask_test,:])
    y_predicted = np.ones(len(inX_test))*-999
    y_predicted[dnnMask_test] = y_predicted__[:,0]
    assert ((y_predicted[dnnMask_test]>0).all()), "Predicted negative values in the first NN"

    recoBin = getRecoBin()
    
# RESPONSE MATRIX
    nbin = len(recoBin)-1
    matrix = np.empty((nbin, nbin))
    for i in range(nbin):
        maskGen = (totGen_test >= recoBin[i]) & (totGen_test < recoBin[i+1])
        # maskNorm = (totGen >= recoBin[i]) & (totGen < recoBin[i+1])
        normalization = np.sum(weights_test[maskGen])
        for j in range(nbin):
            # print(f"Bin generated #{i}\tBin reconstructed #{j}\r", end="")
            maskRecoGen = (y_predicted >= recoBin[j]) & (y_predicted < recoBin[j+1]) & maskGen
            entries = np.sum(weights_test[maskRecoGen])
            # print("entries",entries)
            if normalization == 0:
                matrix[j, i] = 0
            else:
                matrix[j, i] = entries/normalization

# UNFOLDING
    dnnCounts, b_ = np.histogram(y_predicted[dnnMask_test] , bins=getRecoBin(), weights=weights_test[dnnMask_test], density=False)
    #dnnUnfolded   = np.linalg.inv(matrix).dot(dnnCounts)

# COVARIANCE MATRIX
    dy = np.diag(dnnCounts)
    try:
        A_inv = np.linalg.inv(matrix)
        cov = (A_inv.dot(dy)).dot(A_inv.T)
        #print(cov)
        corr = np.ones((nbin, nbin))

        for i in range(nbin):
             for j in range(nbin):
  
                corr[i, j] = cov[i, j]/np.sqrt(cov[i, i] * cov[j, j])
    except:
        print("matrix not computed")
    try:
        det = np.linalg.det(corr)
    except:
        det = -999
        print("det not computed")

    

    
    return det


    #mse = mean_squared_error(outY_test[dnnMask_test ,:], y_predicted[dnnMask_test ], sample_weight=weights_test)            # | dnn2Mask_test


def main():
    maxEvents_          = None
    hp = {
        'batchSize':        128,
        'validBatchSize':   256,
        'activation':       'elu',
        'nFiles':           3,
        'epochs':           3500,
        'patienceeS':       70,
        'numTrainings':     3
        #'learningRate':     0.0001, 
        #'nNodes':           [32, 64, 128], #32, 64, 256 ,  21, 428, 276 -- 
        #'regRate':          0.01,
        #'scale' :           'standard',
        #'testFraction':     0.2,
        #'validation_split': 0.2,
        #'alpha':            180,
        #'nNodes2':          [64, 32 ],
        #'learningRate2':    0.01,
        #'epochs2':          12,
        #'regRate2':         0.02
    }
    learningRate = 0.0001
    npyDataFolder = "/nfs/dust/cms/user/celottog/mttNN/npyData/currentRandom"
    #npyDataFolder = "/nfs/dust/cms/user/celottog/mttNN/npyData/1*None"
    print("npyDataFolder \t:\t", npyDataFolder)
    inX_train, inX_valid, inX_test, outY_train, outY_valid, outY_test, weights_train, weights_valid, weights_test, totGen_test = loadDataSaved(npyDataFolder)
  

# nNode generation
    nDense = np.random.randint(2, 5)
    nNodes = []
    for i in range(nDense):
        t = np.random.randint(1, 64)*3
        nNodes.append(t)
# activation generation
    #activation_index = np.random.randint(0, 4)
    #activation = ['relu', 'gelu', 'selu', 'elu'][activation_index]
    activation = 'elu'
#regRate generation: mean in 10**-2
    #regRate = 0.00658
    
    mean = np.log10(0.01)
    sigma = 0.666
    lasso = 10**(np.random.normal(loc = mean, scale = sigma))

    mean = np.log10(0.01)
    sigma = 0.666
    ridge = 10**(np.random.normal(loc = mean, scale = sigma))
    det = []

    for i in range(hp['numTrainings']):
        det__ = objective(inX_train, inX_valid, inX_test, outY_train, outY_valid, outY_test, weights_train, weights_valid, weights_test, totGen_test,
                          nNodes=nNodes, activation=activation, lasso = lasso, ridge = ridge, learningRate = learningRate, hp =hp)
        det.append(det__)

    det = np.array(det)
    det = det[det>-998]
    with open("/nfs/dust/cms/user/celottog/mttNN/outputs/comparisons/newRandom.txt", "a+") as f:
                string = "\n{:.6f}\t{:.6f}\t".format(det.mean(), det.std()/np.sqrt(len(det)))
                string = string + str(nNodes) + "\t%.5f"%lasso + "\t%.5f"%ridge
                #+"\t"+ str(activation) +"\t%.7f"%learningRate 

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


# SECOND NN
'''dnn2Mask_train = inX_train[:,0]<-4998
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
    assert ((y_predicted[dnn2Mask_test]>0).all()), "Predicted negative values in the Second NN"'''
    