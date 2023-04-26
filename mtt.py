import os
import numpy as np
from sklearn.model_selection import train_test_split
from utils.helpers import *
from utils.models import *
from utils.plot import *
from utils.stat import *
from utils.data import loadData
from utils import defName
from utils.splitTrainValid import *
import matplotlib
from tensorflow import keras
import tensorflow as tf
import shap

matplotlib.use('agg')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'        # TensorFlow will only display error messages and suppress all other messages including these warnings.


def justEvaluate(dataPathFolder, modelDir, modelName, outFolder, testFraction , maxEvents, minbjets, nFiles_, scale) :
        print("Calling justEvaluate ...")
        inX_train, inX_test, outY_train, outY_test, weights_train, weights_test, lkrM_train, lkrM_test, krM_train, krM_test, totGen_train, totGen_test, mask_train, mask_test = loadData(dataPathFolder = dataPathFolder,
                                                                                                                                                testFraction = testFraction,
                                                                                                                                                maxEvents = maxEvents,
                                                                                                                                                minbjets = 1,
                                                                                                                                                   nFiles_ = nFiles_, scale=scale)
    #inX_test     = np.load(dataPathFolder + "/testing/flat_inX_test.npy")
    #outY_test    = np.load(dataPathFolder + "/testing/flat_outY_test.npy")
    #weights_test = np.load(dataPathFolder + "/testing/flat_weights_test.npy")
    #lkrM_test    = np.load(dataPathFolder + "/testing/flat_lkrM_test.npy")
    #krM_test     = np.load(dataPathFolder + "/testing/flat_krM_test.npy")
    #totGen       = np.load(dataPathFolder + "/flat_totGen.npy")
        # def my_l2_regularizer(weight_matrix):
            # Define your custom L2 regularization logic here
         #       return l2(weight_matrix, l=0.01)

        # Register the custom L2 regularizer
        #get_custom_objects().update({'my_l2_regularizer': my_l2_regularizer})
        model = keras.models.load_model(modelDir+"/"+modelName+".h5")
        y_predicted = model.predict(inX_test[mask_test,:])
        print("Starting with plots...")

        explainer = shap.GradientExplainer(model, inX_train[mask_train, :])



        #linearCorrelation(yPredicted= y_predicted[:,0], lkrM = lkrM_test, krM = krM_test, totGen = totGen_test, outFolder = outFolder+"/test", mask_test = mask_test)
        #doEvaluationPlots(outY_test[mask_test,:], y_predicted, lkrM_test, krM_test, outFolder = outFolder+"/test", totGen = totGen_test, mask_test=mask_test, write=True)
        
def main():
    print("********************************************\n*					   *\n*        Main function started             *\n*					   *\n********************************************")
    nFiles_             = 1
    maxEvents_          = None
    testFraction_       = 0.3
    validation_split_   = 0.3
    epochs_             = 10
    patienceeS_         = 50
    scale               = 'multi'
    additionalInput     = ""
    additionalName      = "2404afterBayes"
    hp = {
    'learningRate': 0.01, 
    'batchSize': 128,
    'validBatchSize': 512,
    'nNodes': [32], #46, 38, 111
    'regRate': 0.00658,
    'activation': 'elu'
}
    hp['nDense']= len(hp['nNodes'])
    doEvaluate = True



    inputName, dataPathFolder, outFolder =  defName.getNames(nFiles_, maxEvents_, hp['nDense'], hp['nNodes'])
    dataPathFolder = dataPathFolder + additionalInput   # datapathfolder = npyData/10*None + additionalinput
    outFolder = outFolder + additionalName              # outFolder=outputs/10*None_3*[70_30_10] + addname
    if not os.path.exists(outFolder):
        os.makedirs(outFolder)
    if not os.path.exists(outFolder+ "/model"):
        os.makedirs(outFolder+ "/model")
        print("Created folder " + outFolder+"/model")
    modelName = "mttRegModel"
    defName.printStatus(hp, outFolder+"/model")         # print the arhitecture NN on the file info.txt
    
    if (doEvaluate):
        justEvaluate(   dataPathFolder = dataPathFolder, modelDir = outFolder+"/model", modelName = modelName, outFolder=outFolder,
                        testFraction = testFraction_ , maxEvents = maxEvents_, minbjets = 1, nFiles_ = nFiles_, scale=scale)
    
    else:
        inX_train, inX_test, outY_train, outY_test, weights_train, weights_test, lkrM_train, lkrM_test, krM_train, krM_test, totGen_train, totGen_test, mask_train, mask_test = loadData(dataPathFolder = dataPathFolder,
                                                                                                                                                testFraction = testFraction_,
                                                                                                                                                maxEvents = maxEvents_,
                                                                                                                                                minbjets = 1,
                                                                                                                                                nFiles_ = nFiles_, scale=scale)

        

# Create weights for training I want only modify the real weights
        weights_train[mask_train], weights_train_original = getWeightsTrain(outY_train[mask_train], weights_train[mask_train], outFolder=outFolder, exp_tau = 150, output=True)
# Split train and validation
        inX_train, outY_train, weights_train, lkrM_train, krM_train, mask_train, inX_valid, outY_valid, weights_valid, lkrM_valid, krM_valid, mask_valid = splitTrainvalid(inX_train, outY_train, weights_train, lkrM_train, krM_train, totGen_train, mask_train, validation_split_, dataPathFolder=dataPathFolder)
        
# Get the model, optmizer,         
        model = getModel(regRate = hp['regRate'], activation = hp['activation'],
                            nDense = hp['nDense'],  nNodes = hp['nNodes'],
                            inputDim = inX_train.shape[1], outputActivation = 'linear')
        optimizer = keras.optimizers.Adam(   learning_rate = hp['learningRate'], beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Adam") #use_ema=False, bema_momentum=0.99, ema_overwrite_frequency=None, 
        model.compile(optimizer=optimizer, loss = keras.losses.MeanSquaredError(), weighted_metrics=[])
        callbacks=[]
        earlyStop = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=patienceeS_, verbose = 1, restore_best_weights=True)
        callbacks.append(earlyStop)

# Fit the model
        fit = model.fit(    x = inX_train[mask_train, :], y = outY_train[mask_train, :], batch_size = hp['batchSize'], epochs = epochs_, verbose = 1,
                            callbacks = callbacks, #validation_split = validation_split_, # The validation data is selected from the last samples in the x and y data provided, before shuffling. 
                            validation_data = (inX_valid[mask_valid, :], outY_valid[mask_valid, :], weights_valid[mask_valid]), validation_batch_size = hp['validBatchSize'],
                            shuffle = False, sample_weight = weights_train[mask_train], initial_epoch=2)

        model.save(outFolder+"/model/"+modelName+".h5")
        keras.backend.clear_session()
        #keras.backend.set_learning_phase(False)
        model = keras.models.load_model(outFolder+"/model/"+modelName+".h5")
        #print('inputs: ', [input.op.name for input in model.inputs])
        #print('outputs: ', [output.op.name for output in model.outputs])


# predict the valid samples and produce plots    
        y_predicted_train, y_predicted_valid, y_predicted = computePredicted(inX_train[mask_train, :], inX_valid[mask_valid, :], inX_test[mask_test, :], dataPathFolder, model)
        doPlotLoss(fit = fit, outFolder=outFolder+"/model")
    
# print statistic (rho, Loss, JS) and model summary on screen and in /model/Info.txt
        print(outY_test.shape)
        print(y_predicted.shape)
        print(mask_test.shape)
        printStat(outY_test[mask_test,:], y_predicted, outFolder, model)
        
        
        #doEvaluationPlots(outY_train, y_predicted_train, lkrM_train, krM_train, outFolder = outFolder+"/train", totGen = totGen_train, write=False)
        print("Plots for testing...")
#       nota loose and kin sono passati con la maschera (quando non funzionano assumono -999)
        linearCorrelation(yPredicted= y_predicted[:,0], lkrM = lkrM_test, krM = krM_test, totGen = totGen_test, outFolder = outFolder+"/test", mask_test = mask_test)
        doEvaluationPlots(outY_test[mask_test,:], y_predicted, lkrM_test, krM_test, outFolder = outFolder+"/test", totGen = totGen_test, mask_test=mask_test, write = True)
        featureNames = getFeatureNames()
        print('Plotting SHAP:')
        assert (inX_test.shape[1]==len(featureNames)), "Check len featureNames and number of features"
        
        print(inX_train.shape)
        print(len(featureNames))
        doPlotShap(featureNames, model, inX_test[mask_test,:], outFolder = outFolder)


if __name__ == "__main__":
	main()

# ypredicted is only for DNN cuts passed
# lkrM and krM have as many elements as totgen, -999 when the single reconstruction does not work or when cuts not passed
# outY_test generated masses when cuts and dnn cuts satisfied
# totGen is just the list of all generated masses without any cut
# totGen[mask_test]=out_test