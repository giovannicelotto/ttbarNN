import os
import numpy as np
from sklearn.model_selection import train_test_split
from utils.helpers import *
from utils.models import *
from utils.plot import *
from utils.stat import *
from npyData.checkFeatures import checkFeatures
import matplotlib.pyplot as plt
import matplotlib
from array import array
from tensorflow import keras

from scipy.stats.stats import pearsonr
from scipy.spatial.distance import jensenshannon
from sklearn.utils import shuffle
from scipy.stats import gaussian_kde
from scipy.stats import entropy

matplotlib.use('agg')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'        # TensorFlow will only display error messages and suppress all other messages including these warnings.


nFiles_             = 1
maxEvents_          = 100000
testFraction_       = 0.3
validation_split_   = 0.3
epochs_             = 50
learningRate_       = 0.01  #tb0
batchSize_          = 128    #tbo
validBatchSize_     = 512    #tbo
nNodes_             = [12, 6] # tbo
nDense_             = len(nNodes_)
regRate_            = 0.005     #tbo
activation_         = 'relu'
outputActivation_   = 'linear'
patienceeS_         = 200
reduceLR_factor     = 0.
dropout_            = 0.
doEvaluate          = False
exp_tau_             = 150
minbjets_           = 1
#inputName           = str(nFiles_)+"*"+((str(int(maxEvents_/1000))+"k") if maxEvents_ is not None else 'None')
#additionalName_     = str(nFiles_)+"*"+((str(int(maxEvents_/1000))+"k") if maxEvents_ is not None else 'None')+"_"+str(nDense_)+"*"+str(nNodes_).replace(", ", "_")
inputName           = "1*None"
additionalName_     = "new"
#inputName           = "emu_AtLeast"+str(minbjets_)+"Bj"+str(nFiles_)+"*"+((str(int(maxEvents_/1000))+"k") if maxEvents_ is not None else 'None')+"Ev"
#additionalName_     = str(nDense_)+"*"+str(nNodes_)+"_Reg"+str(regRate_)+"_Batch"+str(batchSize_)+"Ev"+((str(int(maxEvents_/1000))+"k") if maxEvents_ is not None else 'None')




def loadData(dataPathFolder , testFraction, maxEvents):
    '''
    Load data from saved numpy arrays or create them if not available (using loadRegressionData)
    '''
    
    print ("LoadData from "+dataPathFolder+"/flat_[..].npy")
    print ("\t nFiles       = "+str(nFiles_))
    print ("\t maxEvents    = "+str(maxEvents))
    print ("\t testFraction = "+str(testFraction))
    print ("\t valid split  = "+str(validation_split_))
    print ("\t Epochs       = "+str(epochs_))
    print ("\t learningRate = "+str(learningRate_))  
    print ("\t batchSize    = "+str(batchSize_))
    print ("\t nDense       = "+str(nDense_))
    print ("\t nNodes       = "+str(nNodes_))
    print ("\t regRate      = "+str(regRate_))
    print ("\t activation   = "+str(activation_))
    #print ("\t patiencelR   = "+str(patiencelR_))
    print ("\t patienceeS   = "+str(patienceeS_))
    print ("\t dropout      = "+str(dropout_))
    print ("\t expTau       = "+str(exp_tau_))


    createNewData = False
    if not os.path.exists(dataPathFolder+ "/flat_inX.npy"):
        createNewData = True
        print("*** Not found the data in the directory "+dataPathFolder+ "/flat_inX.npy")
    else:
        print("\nData found in the directory :" + dataPathFolder)
        createNewData = False

    if createNewData:
        inX, outY, weights, lkrM, krM, totGen = loadRegressionData("/nfs/dust/cms/user/celottog/ttbarSignalFromDilepton", "miniTree", nFiles=nFiles_, minbjets=minbjets_, maxEvents=maxEvents)
        inX     = np.array(inX)
        outY    = np.array(outY)
        weights = np.array(weights)
        lkrM    = np.array(lkrM)
        krM     = np.array(krM)
        totGen  = np.array(totGen)
        
        inX, outY, weights, lkrM, krM = shuffle(inX, outY, weights, lkrM, krM, random_state = 1999)
        print("Number training events :", inX.shape[0], len(inX))
        if not os.path.exists(dataPathFolder):
            os.makedirs(dataPathFolder)
        np.save(dataPathFolder+ "/flat_inX.npy", inX)
        np.save(dataPathFolder+ "/flat_outY.npy", outY)
        np.save(dataPathFolder+ "/flat_weights.npy", weights)
        np.save(dataPathFolder+ "/flat_lkrM.npy", lkrM)
        np.save(dataPathFolder+ "/flat_krM.npy", krM)
        np.save(dataPathFolder+ "/flat_totGen.npy", totGen)
        print("Control plots")
        checkFeatures(inX, dataPathFolder)

    
    inX     = np.load(dataPathFolder+ "/flat_inX.npy")
    outY    = np.load(dataPathFolder+ "/flat_outY.npy")
    weights = np.load(dataPathFolder+ "/flat_weights.npy")
    lkrM    = np.load(dataPathFolder+ "/flat_lkrM.npy")
    krM     = np.load(dataPathFolder+ "/flat_krM.npy")
    totGen  = np.load(dataPathFolder+ "/flat_totGen.npy")
    print("\nMaximum of totGen:\t", totGen.max())
    print("Minimum of totGen:\t", totGen.min())

    
    if ((maxEvents is not None) & (createNewData == False)):
        ratio   = maxEvents/len(inX)
        print("maxev", maxEvents)
        #print("inX  ", inX)
        print("ratio",ratio)
        inX     = inX[:maxEvents]
        outY    = outY[:maxEvents]
        weights = weights[:maxEvents]
        lkrM    = lkrM[:maxEvents]
        krM     = krM[:maxEvents]
        totGen  = totGen[:int(round(ratio*(len(totGen)-1)))]

    
    print("scaling data")
    featureNames = getFeatureNames()
    inXs = multiScale(featureNames, inX)
    checkFeatures(inXs, dataPathFolder, name="scaledPlot")

    print('\n\nShapes of all the data at my disposal:\nInput  \t',inX.shape,'\nOutput\t', outY.shape,'\nWeights\t', weights.shape,'\nLoose M\t', lkrM.shape,'\nFull M\t', krM.shape, '\ntotGen\t', totGen.shape)
    
    #
    #qt = QuantileTransformer(output_distribution='normal', random_state=0)
    #

    inX_train, inX_test, outY_train, outY_test, weights_train, weights_test, lkrM_train, lkrM_test, krM_train, krM_test = train_test_split(inXs, outY, weights, lkrM, krM, test_size = testFraction, random_state = 1998)

    print ("\tData splitted succesfully")
    print("Number train+valid events :", inX_train.shape[0], len(inX_train))
    print("Number test events        :", inX_test.shape[0], len(inX_test))
    print("Number of features        :", inX_train.shape[1])
    
    if not os.path.exists(dataPathFolder+ "/testing"):
        os.makedirs(dataPathFolder+ "/testing")
    
    if createNewData:
        np.save(dataPathFolder+ "/testing/flat_inX"    + "_test.npy", inX_test)
        np.save(dataPathFolder+ "/testing/flat_outY"   + "_test.npy", outY_test)
        np.save(dataPathFolder+ "/testing/flat_weights"+ "_test.npy", weights_test)
        np.save(dataPathFolder+ "/testing/flat_lkrM"   + "_test.npy", lkrM_test)
        np.save(dataPathFolder+ "/testing/flat_krM"    + "_test.npy", krM_test)

    return inX_train, inX_test, outY_train, outY_test, weights_train, weights_test,lkrM_train, lkrM_test, krM_train, krM_test, totGen

def doTrainingAndEvaluation(modelName, outFolder, dataPathFolder = "/nfs/dust/cms/user/celottog/mttNN/npyData"):
    print("Calling doTrainingAndEvaluation ...")
    inX_train, inX_test, outY_train, outY_test, weights_train, weights_test, lkrM_train, lkrM_test, krM_train, krM_test, totGen = loadData(
                                    dataPathFolder = dataPathFolder, 
                                    testFraction = testFraction_,
                                    maxEvents = maxEvents_)
    # Create the output folder if it does not exist
    if not os.path.exists(outFolder):
        os.makedirs(outFolder)
    if not os.path.exists(outFolder+ "/model"):
        os.makedirs(outFolder+ "/model")
    with open(outFolder+"/model/Info.txt", "w") as f:
        print ("LoadData from "+dataPathFolder+"/flat_[..].npy")
        print ("\t nFiles       = "+str(nFiles_), file=f)
        print ("\t maxEvents    = "+str(maxEvents_), file=f)
        print ("\t testFraction = "+str(testFraction_), file=f)
        print ("\t valid split  = "+str(validation_split_), file=f)
        print ("\t Epochs       = "+str(epochs_), file=f)
        print ("\t learningRate = "+str(learningRate_), file=f)  
        print ("\t batchSize    = "+str(batchSize_), file=f)
        print ("\t nDense       = "+str(nDense_), file=f)
        print ("\t nNodes       = "+str(nNodes_), file=f)
        print ("\t regRate      = "+str(regRate_), file=f)
        print ("\t activation   = "+str(activation_), file=f)
        #print ("\t patiencelR   = "+str(patiencelR_), file=f)
        print ("\t patienceeS   = "+str(patienceeS_), file=f)
        print ("\t dropout      = "+str(dropout_), file=f)
        print ("\t expTau       = "+str(exp_tau_), file=f)


    print("Check element number 89 of inX_test", inX_test[89,0],"\n")      


    #feature_names, inX_train, inX_test = helpers.getReducedFeatureNamesAndInputs(inX_train, inX_test, tokeep=tokeep)
    featureNames = getFeatureNames()
    print('Features:', featureNames)



# *******************************
# *                             *
# *        Weight part          *
# *                             *
# *******************************
    if not os.path.exists(outFolder+ "/model"):
        os.makedirs(outFolder+ "/model")

    weights_train, weights_train_original = getWeightsTrain(outY_train, weights_train, outFolder, exp_tau = exp_tau_)

# *******************************
# *                             *
# *        Neural Network       *
# *                             *
# *******************************

    print ("getting model")
    model = getMRegModelFlat(regRate = regRate_, activation = activation_, dropout = dropout_, nDense = nDense_,
                                      nNodes = nNodes_, inputDim = inX_train.shape[1], outputActivation = outputActivation_)


    optimizer = keras.optimizers.Adam(   learning_rate = learningRate_,
                                            beta_1=0.9, beta_2=0.999,   # memory lifetime of the first and second moment
                                            epsilon=1e-07,              # regularization constant to avoid divergences
                                            #weight_decay=None,
                                            #use_ema=False, bema_momentum=0.99, ema_overwrite_frequency=None,
                                            name="Adam"
                                            )

    print ("compiling model")
    #model.compile(optimizer=optimizer, loss = keras.losses.MeanSquaredError(), metrics=['mean_absolute_error','mean_absolute_percentage_error'])
    model.compile(optimizer=optimizer, loss = keras.losses.MeanSquaredError())


    callbacks=[]
    #reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', patience=patiencelR_, factor=reduceLR_factor)
    earlyStop = keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=patienceeS_, verbose = 1, restore_best_weights=True)
    #modelCheckpoint = keras.callbacks.ModelCheckpoint(outFolder + '/model_best.h5', monitor='val_loss', save_best_only=True)
    callbacks.append(earlyStop)
    #callbacks.append(modelCheckpoint)
    
    divisor = int ((1-validation_split_)*len(inX_train))
    outY_valid      = outY_train[divisor:]
    inX_valid       = inX_train[divisor:]
    weights_valid   = weights_train[divisor:]
    inX_train       = inX_train[:divisor]
    outY_train      = outY_train[:divisor]
    weights_train   = weights_train[:divisor]
    print("Number train events :", inX_train.shape[0], len(inX_train))
    print("Number valid events :", inX_valid.shape[0], len(inX_valid))
 
    #jsTV = JSdist(outY_train[:len(outY_valid)-1,0], outY_valid[:len(outY_valid)-1,0])
    #print ("JS between KinR training and validation", str(jsTV[0])[:6], str(jsTV[1])[:6])       

    np.save(dataPathFolder+"/testing/flat_inX"    + "_train.npy", inX_train)
    np.save(dataPathFolder+"/testing/flat_outY"   + "_train.npy", outY_train)
    np.save(dataPathFolder+"/testing/flat_inX"    + "_valid.npy", inX_valid)
    np.save(dataPathFolder+"/testing/flat_outY"   + "_valid.npy", outY_valid)
        

    print ("fitting model")
    fit = model.fit(
            x = inX_train,
            y = outY_train,
            batch_size = batchSize_,
            epochs = epochs_,
            verbose = 1,
            callbacks = callbacks,
            #validation_split = validation_split_,                   # The validation data is selected from the last samples in the x and y data provided, before shuffling. 
            validation_data = (inX_valid, outY_valid, weights_valid),
            #validation_data = (inX_valid, outY_valid),#, weights_valid),
            validation_batch_size = validBatchSize_,
            shuffle = False,
            sample_weight = weights_train
            )

    saveModel = not doEvaluate
    if saveModel:
        model.save(outFolder+"/model/"+modelName+".h5")
        keras.backend.clear_session()
        keras.backend.set_learning_phase(0)
        model = keras.models.load_model(outFolder+"/model/"+modelName+".h5")
        print('inputs: ', [input.op.name for input in model.inputs])
        print('outputs: ', [output.op.name for output in model.outputs])


# Real prediction based on training
    y_predicted = model.predict(inX_test)
    y_predicted_train = model.predict(inX_train)
    y_predicted_valid = model.predict(inX_valid)
    
    np.save(dataPathFolder+"/testing/flat_regY"   + "_test.npy", y_predicted)
    np.save(dataPathFolder+"/testing/flat_regY"   + "_train.npy", y_predicted_train)
    np.save(dataPathFolder+"/testing/flat_regY"   + "_valid.npy", y_predicted_valid)

    doPlotLoss(fit = fit, outFolder=outFolder+"/model")
    

    corr = pearsonr(outY_test.reshape(outY_test.shape[0]), y_predicted.reshape(y_predicted.shape[0]))
    print("correlation:",corr[0])
    
    kl = KLdiv(y_predicted[:,0], outY_test[:,0])
    print ("KL", kl)
    js = JSdist(y_predicted[:,0], outY_test[:,0])
    print ("JS", str(js[0])[:6], str(js[1])[:6])
    with open(outFolder+"/model/Info.txt", "a+") as f:
        f.write("JS\t"+ str(js[0])[:6]+"\n"+"KL\t"+ str(kl)[:6]+"\ncorrelation:\t"+str(corr[0])+"\n")
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write("\n")
        #print(model.summary(), file=f)

        
    totGen_train = totGen[ :int((1-testFraction_)*len(totGen))]
    totGen_test  = totGen[ :int(testFraction_*len(totGen))]
    
    doEvaluationPlots(outY_train, y_predicted_train, weights_train_original, lkrM_train[:divisor], krM_train[:divisor], outFolder = outFolder+"/train", totGen = totGen_train, write=False)
    print("Plots for testing")
    doEvaluationPlots(outY_test, y_predicted, weights_test, lkrM_test, krM_test, outFolder = outFolder+"/test", totGen = totGen_test, write = True)
    doPlotShap(featureNames, model, inX_test, outFolder = outFolder)




    
        #frozen_graph = freeze_session(tf.compat.v1.keras.backend.get_session(), output_names=[out.op.name for out in model.outputs])
        #tf.compat.v1.train.write_graph(frozen_graph, outFolder+ modelName+'.pbtxt', as_text=True)
        #tf.compat.v1.train.write_graph(frozen_graph, outFolder+ modelName+'.pb', as_text=False)
        #print ("Saved model to",outFolder+'/'+year+'/'+modelName+'.pbtxt/.pb/.h5')



def justEvaluate(dataPathFolder, modelDir, modelName, outFolder):
    print("Calling justEvaluate ...")
    inX_train, inX_test, outY_train, outY_test, weights_train, weights_test, lkrM_train, lkrM_test, krM_train, krM_test, totGen = loadData(
                                    dataPathFolder = dataPathFolder,
                                    testFraction = testFraction_,
                                    maxEvents = maxEvents_)


    model = keras.models.load_model(modelDir+"/"+modelName+".h5")
    y_predicted = model.predict(inX_test)

    doEvaluationPlots(outY_test, y_predicted, weights_test, lkrM_test, krM_test, outFolder = outFolder+"/test", totGen=totGen[:int(testFraction_*len(totGen))], write=True)

def main():
    doEvaluate=False
    print("********************************************\n*					   *\n*        Main function started             *\n*					   *\n********************************************")
    
    if (doEvaluate):
        justEvaluate(dataPathFolder = "/nfs/dust/cms/user/celottog/mttNN/npyData/"+inputName, 
        	         modelDir = "/nfs/dust/cms/user/celottog/mttNN/outputs/"+additionalName_+"/model", modelName = "mttRegModel", outFolder="/nfs/dust/cms/user/celottog/mttNN/outputs/"+additionalName_)
    
    else:
       doTrainingAndEvaluation(dataPathFolder = "/nfs/dust/cms/user/celottog/mttNN/npyData/"+inputName, 
				modelName = "mttRegModel", outFolder="/nfs/dust/cms/user/celottog/mttNN/outputs/"+additionalName_)
if __name__ == "__main__":
	main()
