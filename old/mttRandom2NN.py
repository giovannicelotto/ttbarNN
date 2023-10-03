# use the error of the first bin as criterion
import os
import sys
import numpy as np
from sklearn.metrics import mean_squared_error
from utils.helpers import *
from utils.models import getModelRandom, getSimpleModel
from utils.plot import getRecoBin
import matplotlib.pyplot as plt
import matplotlib
from array import array
from tensorflow import keras
from scipy.stats.stats import pearsonr
from scipy.stats import pearsonr
matplotlib.use('agg')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'        # TensorFlow will only display error messages and suppress all other messages including these warnings.



# Load scaled and splitted quantities.
def loadDataSaved(npyDataFolder):
    print("loading Data from : ", npyDataFolder, "...")
    inX_train = np.load(npyDataFolder     + "/inX_train.npy")
    inX_valid = np.load(npyDataFolder     + "/inX_valid.npy")
    inX_test = np.load(npyDataFolder      + "/inX_test.npy")
    outY_train = np.load(npyDataFolder    + "/outY_train.npy")
    outY_valid = np.load(npyDataFolder    + "/outY_valid.npy")
    outY_test = np.load(npyDataFolder     + "/outY_test.npy")
    weights_train = np.load(npyDataFolder + "/weights_train.npy")
    weights_valid = np.load(npyDataFolder + "/weights_valid.npy")
    weights_test = np.load(npyDataFolder  + "/weights_test.npy")
    totGen_test = np.load(npyDataFolder   + "/totGen_test.npy")

    print("data loaded succesfully")
    return inX_train, inX_valid, inX_test, outY_train, outY_valid, outY_test, weights_train, weights_valid, weights_test, totGen_test







def objective(inX_train, inX_valid, inX_test, outY_train, outY_valid, outY_test, weights_train, weights_valid, weights_test, totGen_test, nNodes, activation,
              lasso, ridge, learningRate, nNodes_2, lasso_2, ridge_2, learningRate_2, hp):
    
    dnnMask_train = inX_train[:,0]>-998
    dnnMask_valid = inX_valid[:,0]>-998
    dnnMask_test = inX_test[:,0]>-998
    dnn2Mask_train = inX_train[:,0]<-4998
    dnn2Mask_valid = inX_valid[:,0]<-4998
    dnn2Mask_test = inX_test[:,0]<-4998

    maskTrain = dnnMask_train | dnn2Mask_train
    maskTest  = dnnMask_test | dnn2Mask_test
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


    y_predicted__ = model.predict(inX_test[dnnMask_test,:])
    y_predicted = np.ones(len(inX_test))*-999
    y_predicted[dnnMask_test] = y_predicted__[:,0]



    assert ((outY_train[dnn2Mask_train, :]>0).all()), "outY_train used for the second training is wrong"
# Select only events that satisfy kinematic cuts (Njets, nbjets, passCuts) but for which the analytical solutions do not exist (or are not adequate)
    SinX_train       = inX_train[dnn2Mask_train,15:]
    Sweights_train   = weights_train[dnn2Mask_train]
    SinX_valid       = inX_valid[dnn2Mask_valid,15:]
    Sweights_valid   = weights_valid[dnn2Mask_valid]
    SinX_test        = inX_test[dnn2Mask_test, 15:]
    
# Get the model, optmizer, 
    
    simpleModel = getSimpleModel(lasso = lasso_2, ridge=ridge_2, activation = 'elu',
                    nDense = len(nNodes_2),  nNodes = nNodes_2,
                    inputDim = SinX_train.shape[1], outputActivation = 'linear')
    optimizer = keras.optimizers.Adam(   learning_rate = learningRate_2, beta_1=0.9, beta_2=0.999, epsilon=1e-01, name="Adam") 
    simpleModel.compile(optimizer=optimizer, loss = keras.losses.MeanSquaredError(), weighted_metrics=[])
    callbacks=[]
    earlyStop = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = hp['patienceeS'], verbose = 1, restore_best_weights=True)
    callbacks.append(earlyStop)
                
        # Fit the model
    simpleFit = simpleModel.fit(x = SinX_train[:, :], y = outY_train[dnn2Mask_train, :], batch_size = hp['batchSize'], epochs = hp['epochs'], verbose = 2,
                    callbacks = callbacks, validation_batch_size = hp['validBatchSize'],
                    validation_data = (SinX_valid[:, :], outY_valid[dnn2Mask_valid, :], np.abs(Sweights_valid[:])),
                    shuffle = False, sample_weight = np.abs(Sweights_train[:]), initial_epoch=2)

    

# predict the valid samples and produce plots    
    y_predicted__ = simpleModel.predict(SinX_test[:,:])
    y_predicted[dnn2Mask_test] = y_predicted__[:,0]

    assert ((y_predicted[dnnMask_test]>0).all()), "Predicted negative values in the first NN"
    assert ((y_predicted[dnn2Mask_test]>0).all()), "Predicted negative values in the first NN"

    print("Unweighted mse:", mean_squared_error(y_true=totGen_test[maskTest], y_pred=y_predicted[maskTest], sample_weight=weights_test[maskTest]) )
# RESPONSE MATRIX
    recoBin = getRecoBin()
    nbin = len(recoBin)-1
    matrix = np.empty((nbin, nbin))
    for i in range(nbin):
        maskGen = (totGen_test >= recoBin[i]) & (totGen_test < recoBin[i+1])
        
        normalization = np.sum(weights_test[maskGen])
        for j in range(nbin):
            
            maskRecoGen = (y_predicted >= recoBin[j]) & (y_predicted < recoBin[j+1]) & maskGen
            entries = np.sum(weights_test[maskRecoGen])
            
            if normalization == 0:
                matrix[j, i] = 0
            else:
                matrix[j, i] = entries/normalization

# UNFOLDING
    dnnCounts, b_ = np.histogram(y_predicted[maskTest] , bins=getRecoBin(), weights=weights_test[maskTest], density=False)
    
# COVARIANCE MATRIX
    dy = np.diag(dnnCounts)
    try:
        A_inv = np.linalg.inv(matrix)
        cov = (A_inv.dot(dy)).dot(A_inv.T)
        res = np.linalg.det(cov)

    except:
        res = -999

        print("det not computed")
    return res

def main():
    maxEvents_          = None
    index = 4
    hp = {
        'batchSize':        128*index,
        'validBatchSize':   256*index,
        'activation':       'elu',
        'nFiles':           3,
        'epochs':           3500,
        'patienceeS':       70,
        'numTrainings':     3

    }

    npyDataFolder = "/nfs/dust/cms/user/celottog/mttNN/npyData/currentRandom/3File"
    print("npyDataFolder \t:\t", npyDataFolder)
    inX_train, inX_valid, inX_test, outY_train, outY_valid, outY_test, weights_train, weights_valid, weights_test, totGen_test = loadDataSaved(npyDataFolder)
  

# nNode generation
    nDense = 2
    #nDense = np.random.randint(2, 4)
    nNodes_2 = []
    for i in range(nDense):
        if i ==0:
            t = np.random.randint(4, 80)
        elif i == 1:
            t = np.random.randint(4, 80)
    
        nNodes_2.append(t)

    activation = 'elu'

    mean = np.log10(0.06903)
    sigma = 1
    lasso_2 = 10**(np.random.normal(loc = mean, scale = sigma))

    mean = np.log10(0.00383)
    sigma = 1
    ridge_2 = 10**(np.random.normal(loc = mean, scale = sigma))
    
    mean = np.log10(0.0053)
    sigma = 0.666
    learningRate_2 = 10**(np.random.normal(loc = mean, scale = sigma))

# to be changed
    nNodes = [110, 105, 123]
    lasso  = 0.04885  
    ridge  = 0.00001  
    learningRate = 0.01237

    res = []

    for i in range(hp['numTrainings']):
        res__ = objective(inX_train, inX_valid, inX_test, outY_train, outY_valid, outY_test, weights_train, weights_valid, weights_test, totGen_test,
                          nNodes=nNodes, activation=activation, lasso = lasso, ridge = ridge, learningRate = learningRate, 
                          nNodes_2=nNodes_2,lasso_2 = lasso_2, ridge_2 = ridge_2, learningRate_2 = learningRate_2, hp =hp)
        res.append(res__)

    res = np.array(res)
    res = res[res>-998]
    with open("/nfs/dust/cms/user/celottog/mttNN/outputs/comparisons/2006_newW_2NN.txt", "a+") as f:
                string = "\n{:.6f}\t{:.6f}\t".format(res.mean(), res.std()/np.sqrt(len(res)))
                string = string + str(nNodes_2) + "\t%.5f"%lasso_2 + "\t%.5f"%ridge_2+ "\t%.7f"%learningRate_2


                f.write(string)



if __name__ == "__main__":
    main()
