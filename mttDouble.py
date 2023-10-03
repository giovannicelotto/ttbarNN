# remove correlated features
# check data mc
# remove scaling of input features

import os
import numpy as np
from utils.helpers import *
from utils.models import *
from utils.plot import linearCorrelation,  doEvaluationPlots, doPlotLoss, doPlotShap, checkForOverTraining
from utils.stat import getWeightsTrain, printStat,  scaleNonAnalytical
from utils.data import loadData
from utils import defName
from utils.splitTrainValid import saveTrainValidTest, splitTrainValid
import matplotlib
import pickle as pkl
from tensorflow import keras
from sklearn.preprocessing import QuantileTransformer
import time
matplotlib.use('agg')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

def weightedStd(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- NumPy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return np.sqrt(variance)


def justEvaluate(npyDataFolder, modelDir, modelName, outFolder, testFraction , maxEvents, minbjets, nFiles, scale, doubleNN, smallNN=False) :
        print("Calling justEvaluate ...")
        inX_train, inX_test, outY_train, outY_test, weights_train, weights_test, lkrM_train, lkrM_test, krM_train, krM_test, totGen_train, totGen_test, mask_train, mask_test = loadData(npyDataFolder = npyDataFolder,
                                                                                                                                                testFraction = testFraction, maxEvents = maxEvents,
                                                                                                                                                minbjets = 1, nFiles = nFiles, doubleNN=doubleNN, 
                                                                                                                                                scale = 'standard', outFolder = outFolder+"/model")
        featuresNN1 = 15
        if smallNN:
                featuresNN1=6
                toKeep = [0, 1, 3, 7, 8, 14, 15, 18, 21, 22, 24, 28, 33, 36, 37, 40, 41, 44, 45, 48, 50, 51, 54, 56, 57, 60, 62, 64, 65, 67, 68, 69, 70, 71, 72]
                inX_train = inX_train[:, toKeep]
                inX_test = inX_test[:,   toKeep]

        dnn2Mask_train = inX_train[:,0]<-4998
        dnn2Mask_test = inX_test[:,0]<-4998
        dnnMask_test = inX_test[:,0]>-998
        dnnMask_train = inX_train[:,0]>-998
        model       = keras.models.load_model(modelDir+"/"+modelName+".h5")
        y_predicted = np.ones(len(inX_test))*-999
        trainPrediction = np.ones(len(inX_train))*-999
        

        assert (y_predicted==-999).all(), "check je"
        if (doubleNN):
                simpleModel = keras.models.load_model(modelDir+"/"+modelName+"_simple.h5")
                y_predicted[dnn2Mask_test] = simpleModel.predict(inX_test[dnn2Mask_test,featuresNN1:])[:,0]
                trainPrediction[dnn2Mask_train] = simpleModel.predict(inX_train[dnn2Mask_train,featuresNN1:])[:,0]


        y_predicted[dnnMask_test] = model.predict(inX_test[dnnMask_test,:])[:,0]
        trainPrediction[dnnMask_train] =  model.predict(inX_train[dnnMask_train,:])[:,0]
        print(y_predicted[(dnnMask_test)|(dnn2Mask_test)].min(), y_predicted[(dnnMask_test)|(dnn2Mask_test)].max())
        print("Starting with plots...")
        
        SinX_test = inX_test[dnn2Mask_test,featuresNN1:]
        inX_test = inX_test[inX_test[:,0]>-998,:]
        printStat(totGen_test, [y_predicted, lkrM_test, krM_test], weights_test, outFolder, model)
        linearCorrelation(yPredicted = y_predicted[:], lkrM = lkrM_test, krM = krM_test, totGen = totGen_test, outFolder = outFolder+"/test",  weights = weights_test)
        doEvaluationPlots(y_predicted[:], lkrM_test, krM_test, outFolder = outFolder+"/test", totGen = totGen_test, mask_test = mask_test, weights = weights_test, dnn2Mask = dnn2Mask_test, write=True)
        doPlotShap(np.array(getFancyNames())[toKeep], model, [inX_test[:1000,:]], outName = outFolder+"/model/model_shapGE.pdf")
        if doubleNN:
                doPlotShap(np.array(getFancyNames())[toKeep][featuresNN1:], simpleModel, [SinX_test[:1000,:]], outName = outFolder+"/model/simpleModel_shapGE.pdf")
                checkForOverTraining(y_predicted[y_predicted>-998], trainPrediction[(dnn2Mask_train)|(dnnMask_train)], weights_test[y_predicted>-998], weights_train[(dnn2Mask_train)|(dnnMask_train)], outFolder=outFolder)
        else:
              checkForOverTraining(y_predicted[y_predicted>-998], trainPrediction[(dnnMask_train)], weights_test[y_predicted>-998], weights_train[(dnnMask_train)], outFolder=outFolder)
              
        

        
def main(doubleNN = True, smallNN = True, doEvaluate = False, scaleOut = False, nFiles=None, skipFirst = False):
    start_time = time.time()
    print("********************************************\n*					   *\n*        Main function started             *\n*					   *\n********************************************")
    
    maxEvents_  = None
    hp = {
# Events
    'nFiles':           15 if nFiles == None else nFiles,
    'testFraction':     0.15,
    'validation_split': 0.15,
# Hyperparameters NN1
    'learningRate':     0.001512,       #0.01025,
    'batchSize':        4096,
    'validBatchSize':   4096,
    'nNodes':           [256, 128, 8, 8], #[64, 8, 256, 32], 				
    'lasso':            0,       #0.04648
    'ridge':            2.9235e-02,         #0.00138
    'activation':       'elu',
    'scale' :           'standard',
    'epochs':           3500,
    'patienceeS':       40,
    'alpha':            0,
# Hyperparameters NN2
    'nNodes2':          [20],
    'learningRate2':    0.00151,
    'epochs2':          3500,
    'lasso2':           0,
    'ridge2':           0.007720,
    'batchSize2':        2048,
    'validBatchSize2':   2048,
}
    
    hp['nDense']= len(hp['nNodes'])
    
    additionalInputName = ""
    additionalOutputName= "DoubleNN_rebinNN2" if doubleNN else "SingleNN" 
    if smallNN:
           additionalOutputName=additionalOutputName+"_simple"

# Define input folder and output folder
    npyDataFolder, outFolder =  defName.getNames(hp['nFiles'], maxEvents_, hp['nDense'], hp['nNodes'])
    npyDataFolder = npyDataFolder + additionalInputName         # npyFolder
    outFolder     = outFolder + additionalOutputName            # output of the model
    print("Input Folder: ", npyDataFolder)
    print("Outpot Folder:", outFolder)
    
    if not os.path.exists(outFolder+ "/model"):
        os.makedirs(outFolder+ "/model")
        print("Created folder " + outFolder+"/model")
    modelName = "mttRegModel"
    defName.printStatus(hp, outFolder+"/model")         # print the arhitecture NN on the file info.txt    
    
    
    if (doEvaluate):
        justEvaluate(   npyDataFolder = npyDataFolder, modelDir = outFolder+"/model", modelName = modelName, outFolder=outFolder,
                        testFraction = hp['testFraction'] , maxEvents = maxEvents_, minbjets = 1, nFiles = hp['nFiles'], scale=hp['scale'], doubleNN=doubleNN, smallNN=smallNN)
    
    else:
        inX_train, inX_test, outY_train, outY_test, weights_train, weights_test, lkrM_train, lkrM_test, krM_train, krM_test, totGen_train, totGen_test, mask_train, mask_test = loadData(npyDataFolder = npyDataFolder,
                                                                                                                                                testFraction = hp['testFraction'],
                                                                                                                                                maxEvents = maxEvents_,
                                                                                                                                                minbjets = 1,
                                                                                                                                                nFiles = hp['nFiles'], scale=hp['scale'], outFolder = outFolder+"/model", doubleNN=doubleNN)

# 1NN
        dnnMask_train = inX_train[:,0]>-998
        dnnMask_test  = inX_test[:,0]>-998
        weights_train_original = weights_train.copy()
        weights_train[dnnMask_train] = getWeightsTrain(outY_train[dnnMask_train], weights_train[dnnMask_train], outFolder=outFolder, alpha = hp['alpha'], output=True)

# 2NN SCALING and weighting       
        if (doubleNN):
                dnn2Mask_train = inX_train[:,0]<-4998
                dnn2Mask_test  = inX_test[:,0]<-4998
                weights_train[dnn2Mask_train] = getWeightsTrain(outY_train[dnn2Mask_train],  weights_train[dnn2Mask_train], outFolder=outFolder, alpha = hp['alpha'], output=True, outFix = '2NN')
# End Of 2NN
      
        
        featuresNN1 = 15
        if smallNN:
                featuresNN1 = 6
                #toKeep = [3, 7, 14, 18, 21, 24, 33, 36, 37, 40, 41, 44, 45, 50, 51, 56, 57, 60, 61, 64, 67, 68, 71]
                toKeep = [0, 1, 3, 7, 8, 14, 15, 18, 21, 22, 24, 28, 33, 36, 37, 40, 41, 44, 45, 48, 50, 51, 54, 56, 57, 60, 62, 64, 65, 67, 68, 69, 70, 71, 72]
                
                inX_train = inX_train[:, toKeep]
                inX_test = inX_test[:,   toKeep]
        else:
               toKeep = [True for i in range(inX_test.shape[1])]
                
                
# Split train and validation and save training and validation in testing folder
        if scaleOut:
                meanY = outY_train[dnnMask_train].mean()
                sigmaY = outY_train[dnnMask_train].std()
                meanY_NN2 = outY_train[dnn2Mask_train].mean()
                sigmaY_NN2 = outY_train[dnn2Mask_train].std()
                outY_train[dnnMask_train] = (outY_train[dnnMask_train] - meanY)/sigmaY
                outY_test[dnnMask_test] = (outY_test[dnnMask_test] - meanY)/sigmaY
                outY_train[dnn2Mask_train] = (outY_train[dnn2Mask_train] - meanY_NN2)/sigmaY_NN2
                outY_test[dnn2Mask_test] = (outY_test[dnn2Mask_test] - meanY_NN2)/sigmaY_NN2
                



        inX_train, outY_train, weights_train, weights_train_original, lkrM_train, krM_train, mask_train, inX_valid, outY_valid, weights_valid, weights_valid_original, lkrM_valid, krM_valid, mask_valid = splitTrainValid(inX_train, outY_train, weights_train, weights_train_original, lkrM_train, krM_train, totGen_train, mask_train, hp['validation_split'], npyDataFolder)
        saveTrainValidTest(npyDataFolder, inX_train, outY_train, weights_train, inX_valid, outY_valid, weights_valid,inX_test, outY_test, weights_test, totGen_test)


# FIRST NN 
        dnnMask_train = inX_train[:,0]>-998
        dnnMask_valid = inX_valid[:,0]>-998
        
# Get the model, optmizer,         
        if (not skipFirst):
                model = getModelRandom(lasso = hp['lasso'], ridge = hp['ridge'], activation = hp['activation'], nDense = hp['nDense'], nNodes = hp['nNodes'],  
                                inputDim = inX_train.shape[1], outputActivation = 'linear')
                optimizer = keras.optimizers.Adam(   learning_rate = hp['learningRate'], beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Adam") #use_ema=False, bema_momentum=0.99, ema_overwrite_frequency=None, 
                model.compile(optimizer=optimizer, loss = keras.losses.MeanSquaredError(), weighted_metrics=[])
                callbacks=[]
                earlyStop = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = hp['patienceeS'], verbose = 1, restore_best_weights=True)
                callbacks.append(earlyStop)
                
        # Fit the model
                fit = model.fit(    x = inX_train[dnnMask_train, :], y = outY_train[dnnMask_train, :], batch_size = hp['batchSize'], epochs = hp['epochs'], verbose = 2,
                                callbacks = callbacks, #validation_split = validation_split_, # The validation data is selected from the last samples in the x and y data provided, before shuffling. 
                                validation_data = (inX_valid[dnnMask_valid, :], outY_valid[dnnMask_valid, :], np.abs(weights_valid[dnnMask_valid])), validation_batch_size = hp['validBatchSize'],
                                shuffle = False, sample_weight = np.abs(weights_train[dnnMask_train]), initial_epoch=2)

                model.save(outFolder+"/model/"+modelName+".h5")
                keras.backend.clear_session()
                
                model = keras.models.load_model(outFolder+"/model/"+modelName+".h5") # custom_objects={'mean_weighted_squared_percentage_error': mean_weighted_squared_percentage_error}
                
                y_predicted__ = model.predict(inX_test[dnnMask_test,:])
                y_predicted = np.ones(len(inX_test))*-999
                y_predicted[dnnMask_test] = y_predicted__[:,0]

                trainPrediction__ = model.predict(inX_train[dnnMask_train,:])
                trainPrediction = np.ones(len(inX_train))*-999
                trainPrediction[dnnMask_train] = trainPrediction__[:,0]
                
                if scaleOut:  
                        y_predicted[dnnMask_test] = y_predicted[dnnMask_test] * sigmaY + meanY
                        outY_train[dnnMask_train] = (outY_train[dnnMask_train] * sigmaY) + meanY
                        outY_test[dnnMask_test] = (outY_test[dnnMask_test] * sigmaY) + meanY

                        y_predicted[dnn2Mask_test] = y_predicted[dnn2Mask_test] * sigmaY_NN2 + meanY_NN2
                        outY_train[dnn2Mask_train] = (outY_train[dnn2Mask_train] * sigmaY_NN2) + meanY_NN2
                        outY_test[dnn2Mask_test] = (outY_test[dnn2Mask_test] * sigmaY_NN2) + meanY_NN2
                        
        
        #assert ((y_predicted[dnnMask_test]>0).all()), "Predicted negative values in the first NN"



        
                doPlotLoss(fit = fit, outName=outFolder+"/model"+"/loss.pdf", earlyStop=earlyStop, patience=hp['patienceeS'])
        if skipFirst:
                model       = keras.models.load_model(outFolder+"/model/"+modelName+".h5")
                y_predicted = np.ones(len(inX_test))*-999
                trainPrediction = np.ones(len(inX_train))*-999
                y_predicted[dnnMask_test] = model.predict(inX_test[dnnMask_test,:])[:,0]
                trainPrediction[dnnMask_train] =  model.predict(inX_train[dnnMask_train,:])[:,0]
# SECOND NN
        if (doubleNN):
                dnn2Mask_train = inX_train[:,0]<-4998
                dnn2Mask_valid = inX_valid[:,0]<-4998
                dnn2Mask_test  = inX_test[:,0]<-4998
                assert ((outY_train[dnn2Mask_train, :]>0).all()), "outY_train used for the second training is wrong"
        # Select only events that satisfy kinematic cuts (Njets, nbjets, passCuts) but for which the analytical solutions do not exist (or are not adequate)
                SinX_train       = inX_train[dnn2Mask_train,featuresNN1:]
                Sweights_train   = weights_train[dnn2Mask_train]
                SinX_valid       = inX_valid[dnn2Mask_valid,featuresNN1:]
                Sweights_valid   = weights_valid[dnn2Mask_valid]
                SinX_test        = inX_test[dnn2Mask_test, featuresNN1:]
                
        # Get the model, optmizer, 
                
                simpleModel = getSimpleModel(lasso = hp['lasso2'], ridge=hp['ridge2'], activation = 'elu',
                                nDense = len(hp['nNodes2']),  nNodes = hp['nNodes2'],
                                inputDim = SinX_train.shape[1], outputActivation = 'linear')
                optimizer = keras.optimizers.Adam(   learning_rate = hp['learningRate2'], beta_1=0.9, beta_2=0.999, epsilon=1e-01, name="Adam") 
                simpleModel.compile(optimizer=optimizer, loss = keras.losses.MeanSquaredError(), weighted_metrics=[])
                callbacks=[]
                earlyStopNN2 = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = hp['patienceeS'], verbose = 1, restore_best_weights=True)
                callbacks.append(earlyStopNN2)
                
        # Fit the model
                simpleFit = simpleModel.fit(    x = SinX_train[:, :], y = outY_train[dnn2Mask_train, :], batch_size = hp['batchSize2'], epochs = hp['epochs2'], verbose = 2,
                                callbacks = callbacks, validation_batch_size = hp['validBatchSize2'],
                                validation_data = (SinX_valid[:, :], outY_valid[dnn2Mask_valid, :], np.abs(Sweights_valid[:])),
                                shuffle = False, sample_weight = np.abs(Sweights_train[:]), initial_epoch=2)
                
                simpleModel.save(outFolder+"/model/"+modelName+"_simple.h5")
                keras.backend.clear_session()
                simpleModel = keras.models.load_model(outFolder+"/model/"+modelName+"_simple.h5") #,  custom_objects={'mean_weighted_squared_percentage_error':mean_weighted_squared_percentage_error}
                

        # predict the valid samples and produce plots    
                y_predicted__ = simpleModel.predict(SinX_test[:,:])
                
                doPlotLoss(fit = simpleFit, outName=outFolder+"/model"+"/simpleLoss.pdf", earlyStop=earlyStopNN2, patience=hp['patienceeS'])
                featureNames = getFeatureNames()

                featureNames = np.array(featureNames)[toKeep]
                doPlotShap(featureNames[featuresNN1:], simpleModel, [SinX_test[:1000,:]], outName = outFolder+"/model/simpleModel_shapGE.pdf")
                y_predicted[dnn2Mask_test] = y_predicted__[:,0]
                trainPrediction__ = simpleModel.predict(SinX_train[:,:])
                trainPrediction[dnn2Mask_train] = trainPrediction__[:,0]

        #GC
        np.save(npyDataFolder+"/testing/y_predicted"    + "_test.npy", y_predicted)
        print("8. Plots")
        printStat(totGen_test, [y_predicted, lkrM_test, krM_test], weights_test, outFolder, model)
        linearCorrelation(yPredicted = y_predicted[:], lkrM = lkrM_test, krM = krM_test, totGen = totGen_test, outFolder = outFolder+"/test",  weights = weights_test)
        dnn2Mask_test = inX_test[:,0]<-4998
        doEvaluationPlots(y_predicted[:], lkrM_test, krM_test, outFolder = outFolder+"/test", totGen = totGen_test,    mask_test = mask_test,  dnn2Mask = dnn2Mask_test,  weights = weights_test, write=True)
        if doubleNN:
                checkForOverTraining(y_predicted[y_predicted>-998], trainPrediction[(dnn2Mask_train) | (dnnMask_train)], weights_test[y_predicted>-998], weights_train_original[(dnn2Mask_train) | (dnnMask_train)], outFolder=outFolder)
        else:
                checkForOverTraining(y_predicted[y_predicted>-998], trainPrediction[ (dnnMask_train)], weights_test[y_predicted>-998], weights_train_original[ (dnnMask_train)], outFolder=outFolder)
        
        print('Plotting SHAP:')
        assert (inX_test.shape[1]==len(featureNames)), "Check len featureNames and number of features"
        inX_test = inX_test[dnnMask_test,:]
        doPlotShap(featureNames, model, [inX_test[:1000,:]], outName = outFolder+"/model/model_shapGE.pdf")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Elapsed time: {:.2f} seconds".format(elapsed_time))
        return  

if __name__ == "__main__":
        if len(sys.argv) > 1:
                doubleNN = bool(int(sys.argv[1]))
        if len(sys.argv) > 2:
                smallNN = bool(int(sys.argv[2]))
        if len(sys.argv) > 3:
                doEvaluate = bool(int(sys.argv[3]))
        if len(sys.argv) > 4:
                scaleOut = bool(int(sys.argv[4]))
        if len(sys.argv) > 5:
                nFiles = int(sys.argv[5])
        if len(sys.argv) > 6:
                skipFirst = bool(int(sys.argv[6]))
        
        main(doubleNN, smallNN, doEvaluate, scaleOut, nFiles, skipFirst)
        #else:
        #       main()