# file to be suubmitted to condor repeated 100 times
# serves as initial points for the bayesian optimization
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from utils.helpers import *
from utils.models import *
from utils.plot import *
from utils.stat import *
from utils.defName import *
from utils.splitTrainValid import *
from npyData.checkFeatures import checkFeatures
import matplotlib.pyplot as plt
import matplotlib
from array import array
from tensorflow import keras
from utils.data import *
from scipy.stats.stats import pearsonr
from scipy.spatial.distance import jensenshannon
from sklearn.utils import shuffle
from scipy.stats import gaussian_kde
from scipy.stats import entropy
from scipy.stats import pearsonr
import joblib
matplotlib.use('agg')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'        # TensorFlow will only display error messages and suppress all other messages including these warnings.

nFiles_             = 10
maxEvents_          = None
testFraction_       = 0.3
validationSplit     = 0.3
epochs              = 1000
patienceeS          = 50
batchSize, validBatchSize = [ 128, 512]
learningRate =  0.0001
# Get paths of data based on nFiles and maxEvents
dataPathFolder = getNamesForBayes(nFiles_, maxEvents_) #10*None
# Load data from Numpy Arrays or create them if not available. Scale data using different techinques (standard, maxmin, power-law)
inX_train, inX_test, outY_train, outY_test, weights_train, weights_test, lkrM_train, lkrM_test, krM_train, krM_test, totGen_train, totGen_test, mask_train, mask_test = loadData(dataPathFolder = dataPathFolder,
                                                                                                                                        testFraction = testFraction_,
                                                                                                                                        maxEvents = maxEvents_,
                                                                                                                                        minbjets = 1,  nFiles_ = nFiles_, scale='multi')
weights_train[mask_train], weights_train_original = getWeightsTrain(outY_train[mask_train], weights_train[mask_train], outFolder=None, exp_tau = 150, output=False)
inX_train, outY_train, weights_train, lkrM_train, krM_train, mask_train, inX_valid, outY_valid, weights_valid, lkrM_valid, krM_valid, mask_valid = splitTrainvalid(inX_train, outY_train, weights_train, lkrM_train, krM_train, totGen_train, mask_train, validationSplit, dataPathFolder=dataPathFolder)



def objective(nDense, nNodes, activation, regRate):
    
    model = getModel(regRate, activation, nDense, nNodes, inputDim = inX_train.shape[1])
    optimizer = keras.optimizers.Adam(   learning_rate = learningRate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Adam")
    model.compile(optimizer=optimizer, loss = keras.losses.MeanSquaredError(),  weighted_metrics=[])
    callbacks=[]
    earlyStop = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = patienceeS, verbose = 1, restore_best_weights=True)
    callbacks.append(earlyStop)
    fit = model.fit(    x = inX_train[mask_train, :], y = outY_train[mask_train, :], batch_size = batchSize, epochs = epochs, verbose = 2,
                        callbacks = callbacks,
                        validation_data = (inX_valid[mask_valid, :], outY_valid[mask_valid, :], weights_valid[mask_valid]),
                        validation_batch_size = validBatchSize,
                        shuffle = False, sample_weight = weights_train[mask_train], initial_epoch=2)
    y_predicted = model.predict(inX_test[mask_test, :],)
    mse = mean_squared_error(outY_test[mask_test,:], y_predicted)
    corr = pearsonr(outY_test[mask_test,:].reshape(outY_test[mask_test,:].shape[0]), y_predicted.reshape(y_predicted.shape[0]))[0]
    return mse, corr


def main():
    nDense = 3
# nNode generation
    nNodes = []
    for i in range(nDense):
        t = np.random.randint(8, 513)
        nNodes.append(t)
# activation generation
    activation_index = np.random.randint(0, 4)
    activation = ['relu', 'gelu', 'selu', 'elu'][activation_index]
#regRate generation: mean in 10**-2
    mean = -2
    sigma = 0.666
    regRate = 10**(np.random.normal(loc = mean, scale = sigma))

    mse, corr = objective(nDense=nDense, nNodes=nNodes, activation=activation, regRate = regRate)

    
    with open("/nfs/dust/cms/user/celottog/mttNN/randomResults2004.txt", "a+") as f:
                string = "\n{:.3f}\t{:.5f}\t".format(mse, corr)
                string = string + str(nNodes) +"\t"+ str(activation) +"\t%.4f"%regRate

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