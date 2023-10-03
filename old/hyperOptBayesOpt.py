import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
import sys
folder_path = '/nfs/dust/cms/user/celottog/mttNN'
sys.path.append(folder_path)
from utils.helpers import *
from utils.models import *
from utils.plot import *
from utils.stat import *
from utils.defName import *
from utils.splitTrainValid import *
from npyData.checkFeatures import checkFeatures
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from array import array
from tensorflow import keras
from utils.data import *
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
from sklearn.utils import shuffle
from scipy.stats import gaussian_kde
from scipy.stats import entropy
from hyperopt import tpe, hp, fmin, Trials
import pandas as pd
import itertools

matplotlib.use('agg')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'        # TensorFlow will only display error messages and suppress all other messages including these warnings.
nFiles_             = 10
maxEvents_          = None
testFraction_       = 0.3
patienceeS          = 50
nDense              = 3
validationSplit     = 0.3
epochs              = 10
max_evals           = 2
activation = 'elu' #['selu', 'relu', 'tanh', 'elu', 'sigmoid']
batchSize = 1024
validBatchSize = 1024
learningRate = 0.001
regRate = 0.00658

# Get paths of data based on nFiles and maxEvents
npyDataFolder = getNamesForBayes(nFiles_, maxEvents_)
# Load data from Numpy Arrays or create them if not available. Scale data using different techinques (standard, maxmin, power-law)
inX_train, inX_test, outY_train, outY_test, weights_train, weights_test, lkrM_train, lkrM_test, krM_train, krM_test, totGen_train, totGen_test, mask_train, mask_test,meanOutY,sigmaOutY = loadData(npyDataFolder = npyDataFolder,
                                                                                                                                                testFraction = testFraction_,
                                                                                                                                                maxEvents = maxEvents_,
                                                                                                                                                minbjets = 1,
                                                                                                                                                nFiles_ = nFiles_, scale='multi')
inX_train = inX_train[mask_train, :]
outY_train = outY_train[mask_train, :]
weights_train = weights_train[mask_train]
inX_test = inX_test[mask_test,:]
outY_test = outY_test[mask_test, :]
weights_train, weights_train_original = getWeightsTrain(outY_train, weights_train, outFolder=None, exp_tau = 150, output=False)

def objective(params):

    print("objective called")
    #GC
    nNode1, nNode2, nNode3 = params['nNode1'], params['nNode2'], params['nNode3']
    nNodes = [nNode1, nNode2, nNode3]
    model = getModel(regRate, activation, nDense, nNodes, inputDim = inX_train.shape[1])
    
    optimizer = keras.optimizers.Adam(   learning_rate = learningRate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Adam") #use_ema=False, bema_momentum=0.99, ema_overwrite_frequency=None, 
    model.compile(optimizer=optimizer, loss = keras.losses.MeanSquaredError(),  weighted_metrics=[])
    callbacks=[]
    earlyStop = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = patienceeS, verbose = 1, restore_best_weights=True)
    callbacks.append(earlyStop)


    fit = model.fit(    x = inX_train, y = (outY_train - meanOutY)/sigmaOutY, batch_size = batchSize, epochs = epochs, verbose = 2,
                        callbacks = callbacks, validation_split = validationSplit, validation_batch_size = validBatchSize,
                        shuffle = False, sample_weight = weights_train, initial_epoch=2)
    
    y_predicted = model.predict(inX_test)
    y_predicted = (y_predicted * sigmaOutY) + meanOutY
    mse = mean_squared_error(outY_test, y_predicted)
    return mse 


# evaluate correalation coefficient and mse
def main():
    
    print("********************************************\n*					   *\n*        Main function started             *\n*					   *\n********************************************")
    


    
    print("define space")
    space = {
    #'learningRate'  : hp.lognormal('learningRate', 0, 2.6),
    'nNode1'        : hp.randint('nNode1', 16, 512),
    'nNode2'        : hp.randint('nNode2', 16, 512),
    'nNode3'        : hp.randint('nNode3', 16, 512)
    #GC,
    #'regRate'       : hp.lognormal('regRate', 0, 2.3),
    #'activation'    : hp.choice('activation', activationList),
    #'batchSize'    : hp.choice('batchSize', batchList),
    #'validBatchSize'    : hp.choice('validBatchSize', validBatchList)
    }
    
    # Perform Bayesian optimization
    print("minimize")
    trials = Trials()
    best = fmin(fn=objective,               # Objective Function to optimize
                space=space,                # Hyperparameter's Search Space
                algo=tpe.suggest,           # Optimization algorithm (representative TPE)
                trials = trials,
                max_evals= max_evals              # Number of optimization attempts
                )
    
    history = []
    for trial in trials.trials:
        hyperparams = trial['misc']['vals']
        result = trial['result']
        history.append(hyperparams)
        #hyperparams['result'] = result
    df = pd.DataFrame(history)
    print(df)

    for i in df.keys():
        df[i] = np.concatenate(df[i].values).ravel()
    p = sns.PairGrid(df, diag_sharey=False)
    p.map_diag(plt.hist, edgecolor="k", lw=0.5)
    # Loop through off-diagonal elements
    for i, j in zip(*np.tril_indices_from(p.axes, -1)):
        ax = p.axes[i, j]
        ax.scatter(df.iloc[:, j], df.iloc[:, i], alpha=0.7, c = df.index, cmap="coolwarm")
                    # Hide upper triangle axes
    for i in range(0, df.shape[1]-1):
        for j in range(i+1, df.shape[1]):
            p.axes[i, j].set_visible(False)

    # Show the plot
    p.savefig('/nfs/dust/cms/user/celottog/mttNN/outputs/bayesianOptimization/HyperOptNodes.pdf', bbox_inches='tight')


    nNode1, nNode2, nNode3 = best['nNode1'], best['nNode2'], best['nNode3']
    model = getModelForBayes(regRate, activation, nDense, nNode1, nNode2, nNode3, inX_train.shape[1])
    optimizer = keras.optimizers.Adam(   learning_rate = learningRate, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-07, name="Adam") #use_ema=False, bema_momentum=0.99, ema_overwrite_frequency=None, 
    model.compile(optimizer=optimizer, loss = keras.losses.MeanSquaredError(),  weighted_metrics=[])
    callbacks=[]
    earlyStop = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = patienceeS, verbose = 1, restore_best_weights=True)
    callbacks.append(earlyStop)
    fit = model.fit(    x = inX_train, y = outY_train, batch_size = batchSize, epochs = epochs, verbose = 2,
                        callbacks = callbacks, validation_split = validationSplit, validation_batch_size = validBatchSize,
                        shuffle = False, sample_weight = weights_train, initial_epoch=2)
    y_predicted = model.predict(inX_test)
    mse = mean_squared_error(outY_test, y_predicted)
    corr = pearsonr(outY_test.reshape(outY_test.shape[0]), y_predicted.reshape(y_predicted.shape[0]))
    
    with open("/nfs/dust/cms/user/celottog/mttNN/outputs/bayesianOptimization/HyperOptNodes.txt", "w+") as f:
        print(df, file  =f)
        print(best, file=f)
        print("\nLoss:\t %.2f"%(mse), file=f)
        print("correlation:%.4f"%(corr[0]), file=f)
        
        
if __name__ == "__main__":
    main()
