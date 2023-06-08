

import os
import numpy as np
from utils.helpers import *
from utils.models import *
from utils.plot import linearCorrelation,  doEvaluationPlots, doPlotLoss, doPlotShap
from utils.stat import getWeightsTrain, printStat,  scaleNonAnalytical
from utils.data import loadData
from utils import defName
from utils.splitTrainValid import saveTrainValidTest, splitTrainValid
import matplotlib
from tensorflow import keras
import time

matplotlib.use('agg')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'        # TensorFlow will only display error messages and suppress all other messages including these warnings.


def justEvaluate(npyDataFolder, modelDir, modelName, outFolder, testFraction , maxEvents, minbjets, nFiles, scale, doubleNN) :
        print("Calling justEvaluate ...")
        inX_train, inX_test, outY_train, outY_test, weights_train, weights_test, lkrM_train, lkrM_test, krM_train, krM_test, totGen_train, totGen_test, mask_train, mask_test, meanOutY, sigmaOutY = loadData(npyDataFolder = npyDataFolder,
                                                                                                                                                testFraction = testFraction,
                                                                                                                                                maxEvents = maxEvents,
                                                                                                                                                minbjets = 1,
                                                                                                                                                nFiles = nFiles, scale=scale, outFolder = outFolder+"/model")
        dnn2Mask_train = inX_train[:,0]<-4998
        dnn2Mask_test = inX_test[:,0]<-4998
        dnnMask_test = inX_test[:,0]>-998
        model       = keras.models.load_model(modelDir+"/"+modelName+".h5")
        y_predicted = np.ones(len(inX_test))*-999
        if (doubleNN):
                inX_train[dnn2Mask_train, 15:], inX_test[dnn2Mask_test, 15:] = scaleNonAnalytical(getFeatureNames()[15:], inX_train, inX_test, npyDataFolder, outFolder)
                simpleModel = keras.models.load_model(modelDir+"/"+modelName+"_simple.h5")
                y_predicted[dnn2Mask_test] = simpleModel.predict(inX_test[dnn2Mask_test,15:])[:,0]


        y_predicted[dnnMask_test] = model.predict(inX_test[dnnMask_test,:])[:,0]

        print("Starting with plots...")
        

        inX_test = inX_test[inX_test[:,0]>-998,:]
        doPlotShap(getFeatureNames(), model, [inX_test[:1000,:]], outName = outFolder+"/model/model_shapGE.pdf")

        linearCorrelation(yPredicted = y_predicted[:], lkrM = lkrM_test, krM = krM_test, totGen = totGen_test, outFolder = outFolder+"/test",  weights = weights_test)
        doEvaluationPlots(outY_test[mask_test,:], y_predicted[:], lkrM_test, krM_test, outFolder = outFolder+"/test", totGen = totGen_test, mask_test = mask_test, weights = weights_test, dnn2Mask = dnn2Mask_test, write=True)
        
def main():
    start_time = time.time()
    print("********************************************\n*					   *\n*        Main function started             *\n*					   *\n********************************************")
    

    maxEvents_          = None
    hp = {
# Events
    'nFiles':           3,
    'testFraction':     0.3,
    'validation_split': 0.2,
# Hyperparameters NN1
    'learningRate':     0.000467, 
    'batchSize':        128,
    'validBatchSize':   256,
    'nNodes':           [23, 26, 60], #[27, 27, 78], #[24, 36, 63], # [3, 174]      0.05061  
    'lasso':            0.14874,
    'ridge':            0.00307,
    'activation':       'elu',
    'scale' :           'standard',
    'epochs':           3500,
    'patienceeS':       30,
    'alpha':            174.3,
# Hyperparameters NN2
    'nNodes2':          [12, 24],
    'learningRate2':    0.000467,
    'epochs2':          3500,
    'lasso2':           0.14874,
    'ridge2':           0.00307,
    'numTrainings':     1
}
    doubleNN = True
    doEvaluate = False
    additionalInputName = ""
    additionalOutputName= "DoubleNN" if doubleNN else "SingleNN" 

# Define input folder and output folder
    hp['nDense']= len(hp['nNodes'])
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
    defName.printStatus(hp)

    
    
    
    if (doEvaluate):
        justEvaluate(   npyDataFolder = npyDataFolder, modelDir = outFolder+"/model", modelName = modelName, outFolder=outFolder,
                        testFraction = hp['testFraction'] , maxEvents = maxEvents_, minbjets = 1, nFiles = hp['nFiles'], scale=hp['scale'], doubleNN=doubleNN)
    
    else:
        #print("Going to loadData...")
        #input()  
        #print("Script resumed.")
        inX_train, inX_test, outY_train, outY_test, weights_train, weights_test, lkrM_train, lkrM_test, krM_train, krM_test, totGen_train, totGen_test, mask_train, mask_test = loadData(npyDataFolder = npyDataFolder,
                                                                                                                                                testFraction = hp['testFraction'],
                                                                                                                                                maxEvents = maxEvents_,
                                                                                                                                                minbjets = 1,
                                                                                                                                                nFiles = hp['nFiles'], scale=hp['scale'], outFolder = outFolder+"/model")



# Create weights for training I want only modify the real weights
        #weights_train[mask_train] = getWeightsTrainNew(outY_train[mask_train], weights_train[mask_train], alpha=hp['alpha'], outFolder=outFolder, output=True)
# 1NN
        dnnMask_train = inX_train[:,0]>-998
        dnnMask_test  = inX_test[:,0]>-998
        weights_train[dnnMask_train], weights_train_original = getWeightsTrain(outY_train[dnnMask_train], weights_train[dnnMask_train], outFolder=outFolder, alpha = hp['alpha'], output=True)

# 2NN SCALING and weighting
        # scale data without analytical solutions and fill the corresponding vectors        
        if (doubleNN):
                dnn2Mask_train = inX_train[:,0]<-4998
                dnn2Mask_test  = inX_test[:,0]<-4998
                inX_train[dnn2Mask_train, 15:], inX_test[dnn2Mask_test, 15:] = scaleNonAnalytical(getFeatureNames()[15:], inX_train, inX_test, npyDataFolder, outFolder)
                weights_train[dnn2Mask_train], Sweights_train_original = getWeightsTrain(outY_train[dnn2Mask_train],  weights_train[dnn2Mask_train], outFolder=outFolder, alpha = hp['alpha'], output=True, outFix = '2NN')
# End Of 2NN
# Now where inX_train[:,0] has a value >-4999 the two approaches worked. This dataset is scaled in one way. The corresponding testing dataset is scaled with the same functions
# Where it is <-4999, the two approaches did not work (one or both) and this subset is scaled in another way
# Now the whole training sample is splitted in training and validation. Since vents are randomly distributed the validation may have more events of the second NN or viceversa 
# but this is not relevant in large number limit for the training. Like setting 0.18 of validation instead of 0.2
        
        
        
# Split train and validation and save training and validation in testing folder
        inX_train, outY_train, weights_train, lkrM_train, krM_train, mask_train, inX_valid, outY_valid, weights_valid, lkrM_valid, krM_valid, mask_valid = splitTrainValid(inX_train, outY_train, weights_train, lkrM_train, krM_train, totGen_train, dnnMask_train, hp['validation_split'], npyDataFolder)
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
        model = keras.models.load_model(outFolder+"/model/"+modelName+".h5")
        y_predicted__ = model.predict(inX_test[dnnMask_test,:])
        y_predicted = np.ones(len(inX_test))*-999
        y_predicted[dnnMask_test] = y_predicted__[:,0]
        if (hp['numTrainings'] > 1):
                print(y_predicted[dnnMask_test][:10])
                for i in range(hp['numTrainings']-1):
                        print("Training number %d"%(i+2))
                        fit = model.fit(    x = inX_train[dnnMask_train, :], y = outY_train[dnnMask_train, :], batch_size = hp['batchSize'], epochs = hp['epochs'], verbose = 0,
                                callbacks = callbacks, #validation_split = validation_split_, # The validation data is selected from the last samples in the x and y data provided, before shuffling. 
                                validation_data = (inX_valid[dnnMask_valid, :], outY_valid[dnnMask_valid, :], np.abs(weights_valid[dnnMask_valid])), validation_batch_size = hp['validBatchSize'],
                                shuffle = False, sample_weight = np.abs(weights_train[dnnMask_train]), initial_epoch=2)

                        model.save(outFolder+"/model/"+modelName+"%d.h5"%(i+1))
                        keras.backend.clear_session()
                        model = keras.models.load_model(outFolder+"/model/"+modelName+"%d.h5"%(i+1))
                        y_predicted__ = model.predict(inX_test[dnnMask_test,:])
                        print(y_predicted__[:10, 0])
                        y_predicted[dnnMask_test] = y_predicted[dnnMask_test] + y_predicted__[:,0]
                y_predicted[dnnMask_test] = y_predicted[dnnMask_test]/hp['numTrainings']
                print(y_predicted[dnnMask_test][:10])
        #assert ((y_predicted[dnnMask_test]>0).all()), "Predicted negative values in the first NN"



        #y_predicted_train, y_predicted_valid, y_predicted = computePredicted(inX_train[mask_train, :], inX_valid[mask_valid, :], inX_test[mask_test, :], npyDataFolder, model)
        
# SECOND NN
        if (doubleNN):
                dnn2Mask_train = inX_train[:,0]<-4998
                dnn2Mask_valid = inX_valid[:,0]<-4998
                dnn2Mask_test  = inX_test[:,0]<-4998
                assert ((outY_train[dnn2Mask_train, :]>0).all()), "outY_train used for the second training is wrong"
        # Select only events that satisfy kinematic cuts (Njets, nbjets, passCuts) but for which the analytical solutions do not exist (or are not adequate)
                SinX_train       = inX_train[dnn2Mask_train,15:]
                Sweights_train   = weights_train[dnn2Mask_train]
                SinX_valid       = inX_valid[dnn2Mask_valid,15:]
                Sweights_valid   = weights_valid[dnn2Mask_valid]
                SinX_test        = inX_test[dnn2Mask_test, 15:]
                
        # Get the model, optmizer, 
                
                simpleModel = getSimpleModel(lasso = hp['lasso2'], ridge=hp['ridge2'], activation = 'elu',
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
                
                simpleModel.save(outFolder+"/model/"+modelName+"_simple.h5")
                keras.backend.clear_session()
                simpleModel = keras.models.load_model(outFolder+"/model/"+modelName+"_simple.h5")
                

        # predict the valid samples and produce plots    
                y_predicted__ = simpleModel.predict(SinX_test[:,:])
                doPlotLoss(fit = simpleFit, outName=outFolder+"/model"+"/simpleLoss.pdf")
                featureNames = getFeatureNames()
                doPlotShap(featureNames[15:], simpleModel, [SinX_test[:1000,:]], outName = outFolder+"/model/simpleModel_shapGE.pdf")
                y_predicted[dnn2Mask_test] = y_predicted__[:,0]


        
        
        print("8. Plots")
        doPlotLoss(fit = fit, outName=outFolder+"/model"+"/loss.pdf")
    
# print statistic (rho, Loss, JS) and model summary on screen and in /model/Info.txt
        printStat(totGen_test, y_predicted, weights_test, outFolder, model)
              
        print("Plots for testing...")
#       nota loose and kin sono passati con la maschera (quando non funzionano assumono -999)
        
        linearCorrelation(yPredicted = y_predicted[:], lkrM = lkrM_test, krM = krM_test, totGen = totGen_test, outFolder = outFolder+"/test",  weights = weights_test)
        dnn2Mask_test = inX_test[:,0]<-4998
        doEvaluationPlots(outY_test[mask_test,:], y_predicted[:], lkrM_test, krM_test, outFolder = outFolder+"/test", totGen = totGen_test,    mask_test = mask_test,  dnn2Mask = dnn2Mask_test,  weights = weights_test, write=True)
        featureNames = getFeatureNames()
        
        print('Plotting SHAP:')
        assert (inX_test.shape[1]==len(featureNames)), "Check len featureNames and number of features"
        inX_test = inX_test[dnnMask_test,:]
        doPlotShap(featureNames, model, [inX_test[:1000,:]], outName = outFolder+"/model/model_shapGE.pdf")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Elapsed time: {:.2f} seconds".format(elapsed_time))
        return  

if __name__ == "__main__":
	main()

# ypredicted is only for DNN cuts passed
# lkrM and krM have as many elements as totgen, -999 when the single reconstruction does not work or when cuts not passed
# outY_test generated masses when cuts and dnn cuts satisfied
# totGen is just the list of all generated masses without any cut
# totGen[mask_test]=out_test
