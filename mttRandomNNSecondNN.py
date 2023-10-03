import os
import sys
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from utils.helpers import *
from utils.models import getModelRandom
from utils.plot import getRecoBin
import matplotlib.pyplot as plt
import matplotlib
from array import array
from tensorflow import keras
from scipy.stats import loguniform
matplotlib.use('agg')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'        # TensorFlow will only display error messages and suppress all other messages including these warnings.

# Load scaled and splitted quantities.
def loadDataSaved(npyDataFolder):
    print("loading Data from : ", npyDataFolder, "...")
    inX_train   = np.load(npyDataFolder     + "/inX_train.npy")
    inX_valid   = np.load(npyDataFolder     + "/inX_valid.npy")
    inX_test    = np.load(npyDataFolder     + "/inX_test.npy")
    outY_train  = np.load(npyDataFolder     + "/outY_train.npy")
    outY_valid  = np.load(npyDataFolder     + "/outY_valid.npy")
    outY_test   = np.load(npyDataFolder     + "/outY_test.npy")
    weights_train   = np.load(npyDataFolder + "/weights_train.npy")
    weights_valid   = np.load(npyDataFolder + "/weights_valid.npy")
    weights_test    = np.load(npyDataFolder + "/weights_test.npy")
    totGen_test     = np.load(npyDataFolder + "/totGen_test.npy")

    print("data loaded succesfully")
    return inX_train, inX_valid, inX_test, outY_train, outY_valid, outY_test, weights_train, weights_valid, weights_test, totGen_test


def objective(inX_train, inX_valid, inX_test, outY_train, outY_valid, outY_test, weights_train, weights_valid, weights_test, totGen_test, nNodes, activation, lasso, ridge, learningRate, hp):
    modelNN1 = "/nfs/dust/cms/user/celottog/mttNN/outputs/15*None_[256_128_8_8]SingleNN_simple/model/mttRegModel.h5"
    modelNN1 = keras.models.load_model(modelNN1)
    
    dnn2Mask_train = inX_train[:,0]<-4998
    dnn2Mask_valid = inX_valid[:,0]<-4998
    dnn2Mask_test = inX_test[:,0]<-4998
    dnnMask_test = inX_test[:,0]>-998

# Get the model, optmizer,         
    model = getModelRandom(lasso = lasso, ridge = ridge, activation = activation, nDense = len(nNodes),  nNodes = nNodes, 
                           inputDim = inX_train.shape[1], outputActivation = 'linear')
    optimizer = keras.optimizers.Adam(   learning_rate = learningRate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Adam")
    model.compile(optimizer=optimizer, loss = keras.losses.MeanSquaredError(), weighted_metrics=[])
    callbacks=[]
    earlyStop = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = hp['patienceeS'], verbose = 1, restore_best_weights=True)
    callbacks.append(earlyStop)
    
# Fit the model
    fit = model.fit(    x = inX_train[dnn2Mask_train, :], y = outY_train[dnn2Mask_train, :], batch_size = hp['batchSize'], epochs = hp['epochs'], verbose = 2,
                        callbacks = callbacks, #validation_split = validation_split_, # The validation data is selected from the last samples in the x and y data provided, before shuffling. 
                        validation_data = (inX_valid[dnn2Mask_valid, :], outY_valid[dnn2Mask_valid, :], np.abs(weights_valid[dnn2Mask_valid])), validation_batch_size = hp['validBatchSize'],
                        shuffle = False, sample_weight = np.abs(weights_train[dnn2Mask_train]), initial_epoch=2)


    y_predicted__ = model.predict(inX_test[dnn2Mask_test,:])
    y_predicted = np.ones(len(inX_test))*-999
    y_predicted[dnn2Mask_test] = y_predicted__[:,0]
    y_predicted__ = modelNN1.predict(inX_test[dnnMask_test,:])
    y_predicted[dnnMask_test] = y_predicted__[:,0]
    #assert ((y_predicted[maskTest]>0).all()), "Predicted negative values in the first NN"


    recoBin = getRecoBin()
    
# RESPONSE MATRIX
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
    dnnCounts, b_ = np.histogram(y_predicted[dnnMask_test] , bins=getRecoBin(), weights=weights_test[dnnMask_test], density=False)

# COVARIANCE MATRIX
    dy = np.diag(dnnCounts)
    try:
        A_inv = np.linalg.inv(matrix)
        cov = (A_inv.dot(dy)).dot(A_inv.T)
        res = np.linalg.det(cov)
        
    except:
        res = -999
    
        print("det not computed")

    normalizedMeanSquared = np.sqrt(np.average(((y_predicted[dnn2Mask_test]-totGen_test[dnn2Mask_test])**2)/(totGen_test[dnn2Mask_test]**2), weights = weights_test[dnn2Mask_test]))
    
    # JS
    p = y_predicted[dnn2Mask_test]
    q = totGen_test[dnn2Mask_test]
    x1, x2, nbin = 300, 1200, 50
    bins = np.linspace(x1, x2, nbin)
    p[p>x2]=x2-1
    p[p<x1]=x1+1
    q[q>x2]=x2-1
    q[q<x1]=x1+1

    histP = np.histogram(p, weights=weights_test[dnn2Mask_test], bins=bins)[0]
    histQ = np.histogram(q, weights=weights_test[dnn2Mask_test], bins=bins)[0]
    
    m = (histP + histQ) / 2

    jd0 = np.sqrt(entropy(histP, m)/2 + entropy(histQ, m)/2)
    
    return res, normalizedMeanSquared, jd0



def main():
    maxEvents_          = None
    size = 2**np.random.randint(low=5, high=14) #4
    hp = {
        'batchSize':        size,
        'validBatchSize':   size,
        'activation':       'elu',
        'nFiles':           15,
        'epochs':           3500,
        'patienceeS':       50,
        'numTrainings':     3
        
    }

    npyDataFolder = "/nfs/dust/cms/user/celottog/mttNN/npyData/currentRandom/"+str(hp['nFiles'])+"File"
    print("npyDataFolder \t:\t", npyDataFolder)
    inX_train, inX_valid, inX_test, outY_train, outY_valid, outY_test, weights_train, weights_valid, weights_test, totGen_test = loadDataSaved(npyDataFolder)

# nNode generation
    nDense = np.random.randint(1, 3)        # 1
    nNodes = []
    for idx in range(nDense):
        t = np.random.randint(4, 64)         # 2
        nNodes.append(t)
    
    activation = 'elu'
    
    lasso = 0 #loguniform.rvs(10**-6, 10**-1)
    ridge = loguniform.rvs(10**-6, 10**-1)
    
    learningRate = 0.001512
    res = []
    mse = []
    jd0 = []
    for i in range(hp['numTrainings']):
        det, mse_, jd0_,  = objective(inX_train, inX_valid, inX_test, outY_train, outY_valid, outY_test, weights_train, weights_valid, weights_test, totGen_test,
                          nNodes=nNodes, activation=activation, lasso = lasso, ridge = ridge, learningRate = learningRate, hp =hp)
        res.append(det)
        mse.append(mse_)
        jd0.append(jd0_)
        
    res = np.array(res)
    res = res[res>-998]
    mse = np.array(mse)
    jd0 = np.array(jd0)
    
    with open("/nfs/dust/cms/user/celottog/mttNN/outputs/comparisons/0708_phase2_afterRebin.txt", "a+") as f:
                string = "%d\t%d\t" %(res.mean(), np.std(res, ddof=1)/np.sqrt(len(res)))
                string = string + str(nNodes) + "\t%.2f"%lasso + "\t%.7f"%ridge+ "\t%.5f"%learningRate + "\t%d"%size+ "\t%.7f"%(mse.mean()) + "\t%.7f"%(np.std(mse, ddof=1)/np.sqrt(len(mse)))  + "\t%.4f"%jd0.mean() + "\t%.7f"%(np.std(jd0, ddof=1)/np.sqrt(len(jd0))) +"\n"
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
    
