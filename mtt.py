import os
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

matplotlib.use('agg')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'        # TensorFlow will only display error messages and suppress all other messages including these warnings.


def justEvaluate(dataPathFolder, modelDir, modelName, outFolder, testFraction , maxEvents, minbjets, nFiles_):
    print("Calling justEvaluate ...")
    inX_train, inX_test, outY_train, outY_test, weights_train, weights_test, lkrM_train, lkrM_test, krM_train, krM_test, totGen = loadData(
                                    dataPathFolder, testFraction , maxEvents, minbjets, nFiles_) 

    model = keras.models.load_model(modelDir+"/"+modelName+".h5")
    y_predicted = model.predict(inX_test)

    doEvaluationPlots(outY_test, y_predicted, weights_test, lkrM_test, krM_test, outFolder = outFolder+"/test", totGen=totGen[:int(testFraction_*len(totGen))], write=True)

def main():
    print("********************************************\n*					   *\n*        Main function started             *\n*					   *\n********************************************")
    nFiles_             = 3
    maxEvents_          = None
    testFraction_       = 0.3
    validation_split_   = 0.3
    epochs_             = 1500
    patienceeS_         = 300
    hp = {
    'learningRate': 0.0005,
    'batchSize': 128,
    'validBatchSize': 512,
    'nNodes': [150, 70, 30],
    'regRate': 0.005,
    'activation': 'relu'
}
    hp['nDense']= len(hp['nNodes'])

    printStatus(hp)


    
    inputName, additionalName, dataPathFolder, outFolder =  getNames(nFiles_, maxEvents_, hp['nDense'], hp['nNodes'])
    additionalName = additionalName+"+Batch"
    doEvaluate=False
    if not os.path.exists(outFolder):
        os.makedirs(outFolder)
    if not os.path.exists(outFolder+ "/model"):
        os.makedirs(outFolder+ "/model")
    modelName = "mttRegModel"
    
    if (doEvaluate):
        justEvaluate(dataPathFolder = "/nfs/dust/cms/user/celottog/mttNN/npyData/"+inputName, modelDir = "/nfs/dust/cms/user/celottog/mttNN/outputs/"+additionalName+"/model",
        modelName = "mttRegModel", outFolder="/nfs/dust/cms/user/celottog/mttNN/outputs/"+additionalName,
                     testFraction = testFraction_ , maxEvents = maxEvents_, minbjminbjets = 1, nFiles_ = nFiles_)
    
    else:
        inX_train, inX_test, outY_train, outY_test, weights_train, weights_test, lkrM_train, lkrM_test, krM_train, krM_test, totGen = loadData(dataPathFolder = dataPathFolder,
                                                                                                                                                testFraction = testFraction_,
                                                                                                                                                maxEvents = maxEvents_,
                                                                                                                                                minbjets = 1,
                                                                                                                                                nFiles_ = nFiles_)

# Create weights for training
        weights_train, weights_train_original = getWeightsTrain(outY_train, weights_train, outFolder=outFolder, exp_tau = 150, output=True)
# Split train and validation
        inX_train, outY_train, weights_train, lkrM_train, krM_train, inX_valid, outY_valid, weights_valid, lkrM_valid, krM_valid = splitTrainvalid(inX_train, outY_train, weights_train, lkrM_train, krM_train, validation_split_, dataPathFolder="/nfs/dust/cms/user/celottog/mttNN/npyData/"+inputName)
# Get the model, optmizer,         
        model = getModel(regRate = hp['regRate'], activation = hp['activation'],
                            nDense = hp['nDense'],  nNodes = hp['nNodes'],
                            inputDim = inX_train.shape[1], outputActivation = 'linear')
        optimizer = keras.optimizers.Adam(   learning_rate = hp['learningRate'], beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Adam") #use_ema=False, bema_momentum=0.99, ema_overwrite_frequency=None, 
        model.compile(optimizer=optimizer, loss = keras.losses.MeanSquaredError())
        callbacks=[]
        earlyStop = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=patienceeS_, verbose = 1, restore_best_weights=True)
        callbacks.append(earlyStop)

# Fit the model
        fit = model.fit(    x = inX_train, y = outY_train, batch_size = hp['batchSize'], epochs = epochs_, verbose = 1,
                            callbacks = callbacks, #validation_split = validation_split_, # The validation data is selected from the last samples in the x and y data provided, before shuffling. 
                            validation_data = (inX_valid, outY_valid, weights_valid), validation_batch_size = hp['validBatchSize'],
                            shuffle = False, sample_weight = weights_train)

        model.save(outFolder+"/model/"+modelName+".h5")
        keras.backend.clear_session()
        keras.backend.set_learning_phase(0)
        model = keras.models.load_model(outFolder+"/model/"+modelName+".h5")
        #print('inputs: ', [input.op.name for input in model.inputs])
        #print('outputs: ', [output.op.name for output in model.outputs])


# predict the valid samples and produce plots    
        y_predicted_train, y_predicted_valid, y_predicted = computePredicted(inX_train, inX_valid, inX_test, dataPathFolder, model)
        doPlotLoss(fit = fit, outFolder=outFolder+"/model")
    
# print statistic and model summary
        printStat(outY_test, y_predicted, outFolder, model)
            
        totGen_train = totGen[ :int((1-testFraction_)*len(totGen))]
        totGen_test  = totGen[ :int(testFraction_*len(totGen))]
        
        doEvaluationPlots(outY_train, y_predicted_train, weights_train_original, lkrM_train, krM_train, outFolder = outFolder+"/train", totGen = totGen_train, write=False)
        print("Plots for testing")
        doEvaluationPlots(outY_test, y_predicted, weights_test, lkrM_test, krM_test, outFolder = outFolder+"/test", totGen = totGen_test, write = True)
        featureNames = getFeatureNames()
        print('Features:', featureNames)
        doPlotShap(featureNames, model, inX_test, outFolder = outFolder)


if __name__ == "__main__":
	main()