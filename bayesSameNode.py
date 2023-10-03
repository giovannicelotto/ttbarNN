import os
import numpy as np
from utils.helpers import *
from utils.models import getModelRandom, getSimpleModel
from utils.plot import getRecoBin
from utils.stat import *
import math
import matplotlib.pyplot as plt
import matplotlib
from array import array
from tensorflow import keras
from utils.data import *
from skopt.space import Real, Integer, Categorical
from skopt import gp_minimize
import skopt.plots
from skopt import callbacks
from skopt import dump, load
from skopt.callbacks import CheckpointSaver

matplotlib.use('agg')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'        # TensorFlow will only display error messages and suppress all other messages including these warnings.


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


def objective(params):
    #print("Here")
    nNodes  = params[:3]  #GC
    lasso, ridge, learningRate, index = params[3:]
    activation = 'elu'
    hp = {
    'batchSize':        2**index,
    'validBatchSize':   2**index,
    'activation':       'elu',
    'nFiles':           5,
    'epochs':           3500,
    'patienceeS':       120,
    'numTrainings':     3
}
    

    npyDataFolder = "/nfs/dust/cms/user/celottog/mttNN/npyData/currentRandom/5File"
    print("npyDataFolder \t:\t", npyDataFolder)
    inX_train, inX_valid, inX_test, outY_train, outY_valid, outY_test, weights_train, weights_valid, weights_test, totGen_test = loadDataSaved(npyDataFolder)

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
    #assert ((y_predicted[dnnMask_test]>0).all()), "Predicted negative values in the first NN"

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
        det = np.linalg.det(cov)
    except:
        det = 10**100
        print("matrix not computed")
    with open("/nfs/dust/cms/user/celottog/mttNN/outputs/bayesianOptimization/2807_bayesian.txt", "a+") as f:
                string = "%d\t"%(det)
                string = string + str(nNodes) + "\t%.5f"%lasso + "\t%.5f"%ridge+ "\t%.5f"%learningRate + "\t%d"%index+"\n"
                f.write(string)
    return det

def getPreviousPoints(file_path, x0, y0, columns):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        data, y0 = [], []
        
        for line in lines:
            values = line.strip().split('\t')
            column1, column2, column3, column4, column5 = values
            column1 = float(column1)
            column2 = float(column2)
            column3 = int(column3) 
            column4 = int(column4)
            column5 = float(column5)
            column6 = int(column6)
            
            assert 2<= column3 <= 4
            assert 3<= column4 <= 9
            assert 0.0001 <= column6 <= 1
            assert 5 <= column6 <= 14
                
            data.append([column1, column2, column3, column4, column5, column6])
            
        # Create the DataFrame
        df = pd.DataFrame(data, columns=columns)
        df = df.reset_index(drop=True)
        x0=[]
        for row in df.values:
            temp = []
            temp.append(row[2])
            temp.append(row[3])
            temp.append(row[4])
            
            x0.append(temp)
            y0.append(row[0])
        
        print(df['L2reg'].max())
        print(df['L2reg'].min())
        print(df['index'].max())
        print(df['index'].min())
        print(df['nNodes'].max())
        print(df['nNodes'].min())
        print(df['nDense'].max())
        print(df['nDense'].min())
    return x0, y0

def main():
    file_path = '/nfs/dust/cms/user/celottog/mttNN/outputs/comparisons/W-2_simple.txt'
    

    print("1. Defining space...")
    space = [
                Integer(low = 2,  high = 4,   name = 'nLayers'),
                Integer(low = 3,  high = 9,   name = 'nNodes'),
                Real(low= 0.0001, high = 1,   name = 'L2',            prior='log-uniform'),
                Integer(low = 5,  high = 14,   name = 'size')
                ]
    columns = ['det', 'Err_det', 'nDense', 'nNodes',  'L2reg', 'index']
    print(" Space defined...")
    print("Getting random searches...")
    
    
    
    x0, y0 = []
    if os.path.exists(file_path):
        print("Previous points found")
        x0, y0 = getPreviousPoints(file_path, x0, y0, columns)

    


    print("define checkpoint")
    checkpoint_saver = CheckpointSaver("/nfs/dust/cms/user/celottog/mttNN/outputs/bayesianOptimization/check.pkl", compress=9)
    print("check defined")
    try:
        res = load('/nfs/dust/cms/user/celottog/mttNN/outputs/bayesianOptimization/check2807.pkl')
        print("Resuming from checkpoint...")
        print("Evaluated points", len(res.x_iters))
        print(res.x_iters)
    except FileNotFoundError:
        res = None
    
    if res is None:
        res = gp_minimize(objective, space, x0 = x0, y0=y0, n_calls = 50, n_initial_points=0, random_state=42, callback=[checkpoint_saver])
    else:
        x0 = res.x_iters
        y0 = res.func_vals
        res = gp_minimize(objective, space, x0=x0, y0=y0, random_state=42, n_initial_points=0, callback=[checkpoint_saver], n_calls = 15)


    
    with open("/nfs/dust/cms/user/celottog/mttNN/outputs/bayesianOptimization/2807_bayesian.txt", "a+") as file:
        print("Best parameters: ", res.x, file=file)
        print("Best objective value: ", res.fun, file=file)




    _ = skopt.plots.plot_evaluations(res, plot_dims=['nNode1', 'nNode2', 'nNode3', 'lasso', 'ridge', 'learningRate'])
    plt.savefig("/nfs/dust/cms/user/celottog/mttNN/outputs/bayesianOptimization/Eval_0707.pdf", bbox_inches='tight')

    _ = skopt.plots.plot_objective(res, plot_dims=['nNode1', 'nNode2', 'nNode3', 'lasso', 'ridge', 'learningRate'])
    plt.savefig("/nfs/dust/cms/user/celottog/mttNN/outputs/bayesianOptimization/Obj_0707.pdf", bbox_inches='tight')




if __name__ == "__main__":
    main()
