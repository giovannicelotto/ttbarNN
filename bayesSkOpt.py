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
from scipy.stats.stats import pearsonr
from skopt.space import Real, Integer, Categorical
from skopt import gp_minimize
import skopt.plots
import itertools
import joblib
from skopt import callbacks
from skopt import dump, load
from skopt.callbacks import CheckpointSaver

matplotlib.use('agg')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'        # TensorFlow will only display error messages and suppress all other messages including these warnings.

nFiles_             = 10
maxEvents_          = None
testFraction_       = 0.3
# Get paths of data based on nFiles and maxEvents
dataPathFolder = getNamesForBayes(nFiles_, maxEvents_)
    # Load data from Numpy Arrays or create them if not available. Scale data using different techinques (standard, maxmin, power-law)
inX_train, inX_test, outY_train, outY_test, weights_train, weights_test, lkrM_train, lkrM_test, krM_train, krM_test, totGen_train, totGen_test, mask_train, mask_test = loadData(dataPathFolder = dataPathFolder,
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
nDense              = 3                     # GC
validationSplit     = 0.3
epochs              = 1000                    # GC
patienceeS          = 50                    # GC
n_calls             = 40                     # GC
batchSize, validBatchSize = [ 128, 512]     # GC
learningRate        = 0.0001                       # GC


def objective(params):
    print("objective called")
    nNodes  = params[:nDense]  #GC
    activation, regRate = params[nDense:]
    print("nNodes", nNodes)
    print("activation", activation)
    
    model = getModel(regRate, activation, nDense, nNodes, inputDim = inX_train.shape[1])
    optimizer = keras.optimizers.Adam(   learning_rate = learningRate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Adam")
    model.compile(optimizer=optimizer, loss = keras.losses.MeanSquaredError(),  weighted_metrics=[])
    callbacks=[]
    earlyStop = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = patienceeS, verbose = 1, restore_best_weights=True)
    callbacks.append(earlyStop)
    fit = model.fit(    x = inX_train, y = outY_train, batch_size = batchSize, epochs = epochs, verbose = 2,
                        callbacks = callbacks, validation_split = validationSplit, validation_batch_size = validBatchSize,
                        shuffle = False, sample_weight = weights_train, initial_epoch=2)
    y_predicted = model.predict(inX_test)
    mse = mean_squared_error(outY_test, y_predicted)
    with open("/nfs/dust/cms/user/celottog/mttNN/outputs/bayesianOptimization/SkOpt_2004.txt", "a+") as file:
        print(str(params)+"\t"+str(mse)+"\n", file=file) 
        
    return mse 


def main():
    
    print("********************************************\n*					   *\n*        Main function started             *\n*					   *\n********************************************")

    
    print("define space")
    space = [   
                Integer(low = 8, high = 513,  name = 'nNode1'),
                Integer(low = 8, high = 513,  name = 'nNode2'),
                Integer(low = 8, high = 513,  name = 'nNode3'),
                Categorical(['relu', 'gelu', 'elu', 'selu', 'sigmoid'], name='activation'),
                Real(low= 0.001, high = 0.5, name='regRate', prior='log-uniform')
                
                #Categorical([64, 128, 256], name='batchSize'),
                #Categorical([128, 256, 512, 1024], name='validBatchSize'),
                ]
    

    print("define checkpoint")
    checkpoint_saver = CheckpointSaver("/nfs/dust/cms/user/celottog/mttNN/outputs/bayesianOptimization/check2004.pkl", compress=9)
    print("check defined")
    try:
        res = load('/nfs/dust/cms/user/celottog/mttNN/outputs/bayesianOptimization/check2004.pkl')
        print("Resuming from checkpoint...")
        print("Evaluated points", len(res.x_iters))
        print(res.x_iters)
    except FileNotFoundError:
        res = None

    if res is None:
        res = gp_minimize(objective, space, x0 = [32, 64, 384, 'relu', 0.01], n_calls=n_calls, n_initial_points=15, random_state=42, callback=[checkpoint_saver])
    else:
        x0 = res.x_iters
        y0 = res.func_vals
        res = gp_minimize(objective, space, x0=x0, y0=y0, random_state=42, n_initial_points=0, callback=[checkpoint_saver], n_calls=n_calls)


    print("Done...\n\n")
    nNodes = res.x[:nDense]
    activation, regRate = res.x[nDense:]
    model = getModel(regRate, activation, nDense, nNodes, inputDim=inX_train.shape[1])
    optimizer = keras.optimizers.Adam(   learning_rate = learningRate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Adam")
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
    
    with open("/nfs/dust/cms/user/celottog/mttNN/outputs/bayesianOptimization/SkOpt_2004.txt", "a+") as file:
        print("Best parameters: ", res.x, file=file) 
        print("Best objective value: ", res.fun, file=file)
        print("\nLoss:\t %.2f"%(mse), file=file)
        print("correlation:%.4f"%(corr[0]), file=file)


    _ = skopt.plots.plot_evaluations(res, plot_dims=['nNode1', 'nNode2', 'nNode3', 'activation', 'regRate']) #GC
    plt.savefig("/nfs/dust/cms/user/celottog/mttNN/outputs/bayesianOptimization/Eval_2004.pdf", bbox_inches='tight')

    _ = skopt.plots.plot_objective(res, plot_dims=['nNode1', 'activation']) #gC
    plt.savefig("/nfs/dust/cms/user/celottog/mttNN/outputs/bayesianOptimization/Obj_2004.pdf", bbox_inches='tight')




if __name__ == "__main__":
    main()
