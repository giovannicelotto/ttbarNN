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
import itertools

matplotlib.use('agg')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'        # TensorFlow will only display error messages and suppress all other messages including these warnings.


# evaluate correalation coefficient and mse
def main(lr, bs, vbs, nN, rR, af, index):
    print("********************************************\n*					   *\n*        Main function started             *\n*					   *\n********************************************")
    
    nFiles_             = 7
    maxEvents_          = None
    testFraction_       = 0.3
    validation_split_   = 0.3
    epochs_             = 500
    patienceeS_         = 50

    

    hp = {
    'learningRate': [0.001, 0.0005, 0.0001][lr],
    'batchSize': [64, 128, 256][bs],
    'validBatchSize': [256, 512, 1024][vbs],
    'nNodes': [[150, 70, 30], [150, 70, 30, 15, 6], [32, 32]][nN],
    'regRate': [0.005, 0.001, 0.0005][rR],
    'activation': ['relu', 'selu'][af]
}
    hp['nDense']= len(hp['nNodes'])
    printStatus(hp)
    
    inputName, dataPathFolder, outFolder =  getNamesForHyperSearch(nFiles_, maxEvents_, hp['nDense'], hp['nNodes'])
    

    if not os.path.exists(outFolder):
        os.makedirs(outFolder)
    
# Load data from Numpy Arrays or create them if not available. Scale data using different techinques (standard, maxmin, power-law)
    inX_train, inX_test, outY_train, outY_test, weights_train, weights_test, lkrM_train, lkrM_test, krM_train, krM_test, totGen = loadData(dataPathFolder = dataPathFolder,
                                                                                                                                            testFraction = testFraction_,
                                                                                                                                            maxEvents = maxEvents_,
                                                                                                                                            minbjets = 1,
                                                                                                                                            nFiles_ = nFiles_, output=False)
# Create weights for training
    weights_train, weights_train_original = getWeightsTrain(outY_train, weights_train, outFolder=outFolder, exp_tau = 150, output=False)
# Split train and validation
    inX_train, outY_train, weights_train, lkrM_train, krM_train, inX_valid, outY_valid, weights_valid, lkrM_valid, krM_valid = splitTrainvalid(inX_train, outY_train, weights_train, lkrM_train, krM_train, validation_split_, dataPathFolder=dataPathFolder, saveData = False)
# Get the model, optmizer       
    model = getModel(regRate = hp['regRate'], activation = hp['activation'],
                        nDense = hp['nDense'],  nNodes = hp['nNodes'],
                        inputDim = inX_train.shape[1], outputActivation = 'linear')
    optimizer = keras.optimizers.Adam(   learning_rate = hp['learningRate'], beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Adam") #use_ema=False, bema_momentum=0.99, ema_overwrite_frequency=None, 
    model.compile(optimizer=optimizer, loss = keras.losses.MeanSquaredError())
    callbacks=[]
    earlyStop = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=patienceeS_, verbose = 1, restore_best_weights=True)
    callbacks.append(earlyStop)

# Fit the model
    fit = model.fit(    x = inX_train, y = outY_train, batch_size = hp['batchSize'], epochs = epochs_, verbose = 2,
                        callbacks = callbacks, #validation_split = validation_split_, # The validation data is selected from the last samples in the x and y data provided, before shuffling. 
                        validation_data = (inX_valid, outY_valid, weights_valid), validation_batch_size = hp['validBatchSize'],
                        shuffle = False, sample_weight = weights_train, initial_epoch=2)

    modelName = "mttRegModel"
    model.save(outFolder+modelName+".h5")
    keras.backend.clear_session()
    keras.backend.set_learning_phase(0)
    model = keras.models.load_model(outFolder+modelName+".h5")

# predict the valid samples and produce plots    
    y_predicted_train, y_predicted_valid, y_predicted = computePredicted(inX_train, inX_valid, inX_test, dataPathFolder, model)
    mseTest = np.average((outY_test - y_predicted)**2)
    corr = pearsonr(outY_test.reshape(outY_test.shape[0]), y_predicted.reshape(y_predicted.shape[0]))
    with open(outFolder+"modelSummary7F_2.txt", "a+") as f:
        print(index, hp['learningRate'],"\t",hp['batchSize'],"\t",hp['validBatchSize'],"\t",hp['nNodes'],"\t",hp['nDense'],"\t",hp['regRate'],"\t",hp['activation'], "\t", mseTest, "\t", corr[0], file=f)
    #doPlotLoss(fit = fit, outFolder=outFolder+"/model")

# print statistic and model summary
    #printStat(outY_test, y_predicted, outFolder, model)
        
    #totGen_train = totGen[ :int((1-testFraction_)*len(totGen))]
    #totGen_test  = totGen[ :int(testFraction_*len(totGen))]
    
    #doEvaluationPlots(outY_train, y_predicted_train, weights_train_original, lkrM_train, krM_train, outFolder = outFolder+"/train", totGen = totGen_train, write=False)
    #print("Plots for testing")
    #doEvaluationPlots(outY_test, y_predicted, weights_test, lkrM_test, krM_test, outFolder = outFolder+"/test", totGen = totGen_test, write = True)
    #featureNames = getFeatureNames()
    #print('Features:', featureNames)
    #doPlotShap(featureNames, model, inX_test, outFolder = outFolder)


if __name__ == "__main__":
    lr = int(sys.argv[1])
    bs = int(sys.argv[2])
    vbs = int(sys.argv[3])
    nN = int(sys.argv[4])
    rR = int(sys.argv[5])
    af = int(sys.argv[6])
    index = int(sys.argv[7])
    main(lr, bs, vbs, nN, rR, af, index)
