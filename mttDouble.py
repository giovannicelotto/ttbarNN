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
from tensorflow import keras
import time
matplotlib.use('agg')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf

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

        print("Starting with plots...")
        
        checkForOverTraining(y_predicted[y_predicted>-998], trainPrediction[(dnn2Mask_train) | (dnnMask_train)], weights_test[y_predicted>-998], weights_train[(dnn2Mask_train) | (dnnMask_train)], outFolder=outFolder)
        SinX_test = inX_test[dnn2Mask_test,featuresNN1:]
        inX_test = inX_test[inX_test[:,0]>-998,:]
        printStat(totGen_test, [y_predicted, lkrM_test, krM_test], weights_test, outFolder, model)
        linearCorrelation(yPredicted = y_predicted[:], lkrM = lkrM_test, krM = krM_test, totGen = totGen_test, outFolder = outFolder+"/test",  weights = weights_test)
        doEvaluationPlots(y_predicted[:], lkrM_test, krM_test, outFolder = outFolder+"/test", totGen = totGen_test, mask_test = mask_test, weights = weights_test, dnn2Mask = dnn2Mask_test, write=True)
        doPlotShap(np.array(getFeatureNames())[toKeep], model, [inX_test[:1000,:]], outName = outFolder+"/model/model_shapGE.pdf")
        doPlotShap(np.array(getFeatureNames())[toKeep][featuresNN1:], simpleModel, [SinX_test[:1000,:]], outName = outFolder+"/model/simpleModel_shapGE.pdf")
        

        
def main(doubleNN = True, smallNN = True, doEvaluate = False, scaleOut = False):
    start_time = time.time()
    print("********************************************\n*					   *\n*        Main function started             *\n*					   *\n********************************************")
    

    
    maxEvents_  = None
    hp = {
# Events
    'nFiles':           15,
    'testFraction':     0.1,
    'validation_split': 0.1/0.9,
# Hyperparameters NN1
    'learningRate':     0.001512,       #0.01025,
    'batchSize':        512,
    'validBatchSize':   512,
    'nNodes':           [27, 41, 21], 				
    'lasso':            0.04648 ,       #0.04648
    'ridge':            0.00138,         #0.00138
    'activation':       'elu',
    'scale' :           'standard',
    'epochs':           3500,
    'patienceeS':       120,
    'alpha':            0,
# Hyperparameters NN2
    'nNodes2':          [12, 4],
    'learningRate2':    0.000495,
    'epochs2':          3500,
    'lasso2':           0.05683,
    'ridge2':           0.05323
}
    hp['nDense']= len(hp['nNodes'])
    
    additionalInputName = ""
    additionalOutputName= "DoubleNN_W-2" if doubleNN else "SingleNN" 
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
# pca
        #pca = wpca.WPCA(n_components=44)
        #print("Pre", inX_train.shape, inX_test.shape )
        
        #pca.fit(inX_train[dnnMask_train,:], weights=np.tile(weights_train_original, (inX_train.shape[1], 1)).T)      

        #print("Post", inX_train.shape, inX_test.shape )
        inX_train, outY_train, weights_train, weights_train_original, lkrM_train, krM_train, mask_train, inX_valid, outY_valid, weights_valid, weights_valid_original, lkrM_valid, krM_valid, mask_valid = splitTrainValid(inX_train, outY_train, weights_train, weights_train_original, lkrM_train, krM_train, totGen_train, dnnMask_train, hp['validation_split'], npyDataFolder)
        saveTrainValidTest(npyDataFolder, inX_train, outY_train, weights_train, inX_valid, outY_valid, weights_valid,inX_test, outY_test, weights_test, totGen_test)


# FIRST NN        
        dnnMask_train = inX_train[:,0]>-998
        dnnMask_valid = inX_valid[:,0]>-998
        
# Get the model, optmizer,         
        model = getModelRandom(lasso = hp['lasso'], ridge = hp['ridge'], activation = hp['activation'], nDense = hp['nDense'], nNodes = hp['nNodes'],
                            inputDim = inX_train.shape[1], outputActivation = 'linear')
        optimizer = keras.optimizers.Adam(   learning_rate = hp['learningRate'], beta_1=0.9, beta_2=0.999, epsilon=1e-01, name="Adam") #use_ema=False, bema_momentum=0.99, ema_overwrite_frequency=None, 
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
        
        assert ((y_predicted[dnnMask_test]>0).all()), "Predicted negative values in the first NN"



        
        doPlotLoss(fit = fit, outName=outFolder+"/model"+"/loss.pdf", earlyStop=earlyStop, patience=hp['patienceeS'])
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
                simpleFit = simpleModel.fit(    x = SinX_train[:, :], y = outY_train[dnn2Mask_train, :], batch_size = hp['batchSize'], epochs = hp['epochs2'], verbose = 2,
                                callbacks = callbacks, validation_batch_size = hp['validBatchSize'],
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


        
        
        print("8. Plots")
        printStat(totGen_test, [y_predicted, lkrM_test, krM_test], weights_test, outFolder, model)
        linearCorrelation(yPredicted = y_predicted[:], lkrM = lkrM_test, krM = krM_test, totGen = totGen_test, outFolder = outFolder+"/test",  weights = weights_test)
        dnn2Mask_test = inX_test[:,0]<-4998
        doEvaluationPlots(y_predicted[:], lkrM_test, krM_test, outFolder = outFolder+"/test", totGen = totGen_test,    mask_test = mask_test,  dnn2Mask = dnn2Mask_test,  weights = weights_test, write=True)
        
        checkForOverTraining(y_predicted[y_predicted>-998], trainPrediction[(dnn2Mask_train) | (dnnMask_train)], weights_test[y_predicted>-998], weights_train_original[(dnn2Mask_train) | (dnnMask_train)], outFolder=outFolder)
        
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
                smallNN = bool(int(sys.argv[2]))
                doEvaluate = bool(int(sys.argv[3]))
                scaleOut = bool(int(sys.argv[4]))
        
                main(doubleNN, smallNN, doEvaluate, scaleOut)
        else:
               main()


#pcaw = wpca.WPCA(n_components=1)
                #print("Len weights train\t", len(weights_train_original))
                #print("shape inX_train  \t", inX_train.shape)
                #print("len dnn mask train\t", len(dnnMask_train))
                #print("Len dnn mask true \t", len(dnnMask_train[dnnMask_train]))
                #print("Values shape      \t", (inX_train[:,[0, 4, 25]][dnnMask_train]).shape)
                #print("Weights shape     \t", np.tile(weights_train_original, (3, 1)).T)
                #print(inX_train[dnnMask_train,0].shape)
                #print(pcaw.fit_transform(X=inX_train[:,[0, 4, 25]][dnnMask_train], weights=np.tile(weights_train_original, (3, 1)).T).reshape(-1).shape)
                #print("\n\n\n\n")
                #print(np.std(inX_train[dnnMask_train,0]), np.std(inX_test[dnnMask_test,0]))


                #inX_train[dnnMask_train,0] = pcaw.fit_transform(X=inX_train[:,[0, 4, 25]][dnnMask_train], weights=np.tile(weights_train_original, (3, 1)).T).reshape(-1)
                #inX_test[dnnMask_test,0]  = pcaw.transform(X=inX_test[:,[0, 4, 25]][dnnMask_test]).reshape(-1)
                #inX_train[dnnMask_train,1] = pcaw.fit_transform(X=inX_train[:,[1, 5, 26]][dnnMask_train], weights=np.tile(weights_train_original, (3, 1)).T).reshape(-1)
                #inX_test[dnnMask_test,1]  = pcaw.transform(X=inX_test[:,[1, 5, 26]][dnnMask_test]).reshape(-1)
                #inX_train[dnnMask_train,8] = pcaw.fit_transform(X=inX_train[:,[8, 11]][dnnMask_train], weights=np.tile(weights_train_original, (2, 1)).T).reshape(-1)
                #inX_test[dnnMask_test,8]  = pcaw.transform(X=inX_test[:,[8, 11]][dnnMask_test]).reshape(-1)
                
                #for ss in [0, 1, 8]:
                #        sigma = weightedStd(inX_train[dnnMask_train, ss], weights=weights_train_original)
                #        inX_train[dnnMask_train, ss] = inX_train[dnnMask_train, ss]/sigma
                #        inX_test[dnnMask_test, ss] = inX_test[dnnMask_test, ss]/sigma