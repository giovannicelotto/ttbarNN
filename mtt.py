import os
import numpy as np
from sklearn.model_selection import train_test_split
from utils.helpers import *
from utils.models import *
from utils.plot import *
import matplotlib.pyplot as plt
from array import array
import tensorflow as tf
import shap
from scipy.stats.stats import pearsonr
from scipy.spatial.distance import jensenshannon
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'        # TensorFlow will only display error messages and suppress all other messages including these warnings.

maxEvents_          = 100000
epochs_             = 80      # 5000
learningRate_       = 0.001      # 10^-3
batchSize_          = 200     # 25000
dropout_            = 0.      # 0.4
nDense_             = 1
nNodes_             = 12
regRate_            = 0.01    
activation_         = 'selu'
outputActivation_   = 'linear'  
#patiencelR_         = 30
patienceeS_         = 50
testFraction_       = 0.3
doEvaluate          = False
reduceLR_factor     = 0.

def loadData(dataPathFolder , year, additionalName, testFraction, withBTag, pTEtaPhiMode, maxEvents):
    '''
    Load data from saved numpy arrays or create them if not available (using loadRegressionData)
    '''
    dataPathFolderYear = os.path.join(dataPathFolder, year)
    print ("LoadData for year "+year+"/"+" from "+dataPathFolderYear+"/flat_[..]"+additionalName+".npy")
    print ("\t maxEvents    = "+str(maxEvents))
    print ("\t testFraction = "+str(testFraction))
    print ("\t Epochs       = "+str(epochs_))
    print ("\t learningRate = "+str(learningRate_))  
    print ("\t batchSize    = "+str(batchSize_))
    print ("\t dropout      = "+str(dropout_))
    print ("\t nDense       = "+str(nDense_))
    print ("\t nNodes       = "+str(nNodes_))
    print ("\t regRate      = "+str(regRate_))
    print ("\t activation   = "+str(activation_))
    #print ("\t patiencelR   = "+str(patiencelR_))
    print ("\t patienceeS   = "+str(patienceeS_))


    

    createNewData = False
    if not os.path.exists(dataPathFolderYear+"/flat_inX"+additionalName+".npy"):
        createNewData = True
        print("*** Not found the data in the right directory ***")
        print("***           Producing new data              ***")
    else:
        print("\nData found in the directory :"+dataPathFolderYear)
        createNewData = False

    if createNewData:
        minJets=2
        print ("\nNew data will be loaded with settings: nJets >= "+str(minJets)+"; max Events = "+str(maxEvents))
        inX, outY, weights, lkrM, krM = loadRegressionData("/nfs/dust/cms/user/celottog/TopRhoNetwork/rhoInput/powheg/"+year, "miniTree", maxEvents = 0 if maxEvents==None else maxEvents, withBTag=withBTag, pTEtaPhiMode=pTEtaPhiMode)
        inX=np.array(inX)
        outY=np.array(outY)
        weights=np.array(weights)
        lkrM=np.array(lkrM)
        krM=np.array(krM)
        np.save(dataPathFolderYear+"/flat_inX"+additionalName+".npy", inX)
        np.save(dataPathFolderYear+"/flat_outY"+additionalName+".npy", outY)
        np.save(dataPathFolderYear+"/flat_weights"+additionalName+".npy", weights)
        np.save(dataPathFolderYear+"/flat_lkrM"+additionalName+".npy", lkrM)
        np.save(dataPathFolderYear+"/flat_krM"+additionalName+".npy", krM)

    print("*** inX, outY, weights, lkrM, krM loading")
    inX = np.load(dataPathFolderYear+"/flat_inX"+additionalName+".npy")
    outY = np.load(dataPathFolderYear+"/flat_outY"+additionalName+".npy")
    weights = np.load(dataPathFolderYear+"/flat_weights"+additionalName+".npy")
    lkrM = np.load(dataPathFolderYear+"/flat_lkrM"+additionalName+".npy")
    krM = np.load(dataPathFolderYear+"/flat_krM"+additionalName+".npy")
    print("*** inX, outY, weights, lkrM, krM loaded")

    if maxEvents is not None:
        inX = inX[:maxEvents]
        outY = outY[:maxEvents]
        weights = weights[:maxEvents]
        lkrM = lkrM[:maxEvents]
        krM = krM[:maxEvents]

    print('\n\nShapes of all the data at my disposal:\nInput  \t',inX.shape,'\nOutput\t', outY.shape,'\nWeights\t', weights.shape,'\nLoose M\t', lkrM.shape,'\nFull M\t', krM.shape)

    inX_train, inX_test, outY_train, outY_test, weights_train, weights_test, lkrM_train, lkrM_test, krM_train, krM_test = train_test_split(inX, outY, weights, lkrM, krM, test_size = testFraction, random_state=1995)
    
    print ("\tData splitted succesfully")
    print("Number training events :", inX_train.shape[0], len(inX_train))
    print("Number of features     :", inX_train.shape[1])
    print("loadData ended returning inX_train and so on...")
    if createNewData:
        np.save(dataPathFolderYear+"/testing/flat_inX"    + additionalName+"test.npy", inX_test)
        np.save(dataPathFolderYear+"/testing/flat_outY"   + additionalName+"test.npy", outY_test)
        np.save(dataPathFolderYear+"/testing/flat_weights"+ additionalName+"test.npy", weights_test)
        np.save(dataPathFolderYear+"/testing/flat_lkrM"   + additionalName+"test.npy", lkrM_test)
        np.save(dataPathFolderYear+"/testing/flat_krM"    + additionalName+"test.npy", krM_test)
    return inX_train, inX_test, outY_train, outY_test, weights_train, weights_test,lkrM_train, lkrM_test, krM_train, krM_test

def doTrainingAndEvaluation(dataPathFolder, year, additionalName, tokeep, outFolder, modelName = "rhoRegModel_preUL04_moreInputs_Pt30_madgraph_cutSel"):
    dataPathFolderYear = os.path.join(dataPathFolder, year)   
    inX_train, inX_test, outY_train, outY_test, weights_train, weights_test, lkrM_train, lkrM_test, krM_train, krM_test = loadData(
                                    dataPathFolder = dataPathFolder, year = year,
                                    additionalName = additionalName, testFraction = testFraction_,
                                    withBTag = True, pTEtaPhiMode=True,
                                    maxEvents = maxEvents_)


    print(" Check element number 890 of inX_tets", inX_test[89,0],"\n")      
    print ("Input events with \t",inX_train.shape[1], "features")
# Reduce to most useful features (to be kept)
    #feature_names, inX_train, inX_test = helpers.getReducedFeatureNamesAndInputs(inX_train, inX_test, tokeep=tokeep)
    featureNames = getFeatureNames()
    print('Features:', featureNames)

# *******************************
# *                             *
# *        Weight part          *
# *                             *
# *******************************

    weightBins = array('d', [340, 345, 350, 355, 360, 370, 380,  420, 440, 460, 480, 500, 550, 600, 700, 800, 1000, 1500])
    wegihtNBin = len(weightBins)-1
    weightHisto = ROOT.TH1F("weightHisto","weightHisto",wegihtNBin, weightBins)

    print("Filling histo with m_tt")
    for m, weight in zip(outY_train, weights_train):
        weightHisto.Fill(m,abs(weight))
    weightHisto = NormalizeBinWidth1d(weightHisto)
    
    canvas = ROOT.TCanvas()
    weightHisto.Draw("hist")
    weightHisto.SetTitle("Normalized and weighted m_{tt} distibutions. Training")
    weightHisto.SetYTitle('Normalized Counts')
    canvas.SaveAs(outFolder+ "/weights/mttOriginalWeight.pdf")

    weightHisto.Scale(1./weightHisto.Integral())
    maximumBin = weightHisto.GetMaximumBin()
    maximumBinContent = weightHisto.GetBinContent(maximumBin)


    for binX in range(1,weightHisto.GetNbinsX()+1):
        c = weightHisto.GetBinContent(binX)
#        if c>0:
#            weightHisto.SetBinContent(binX, maximumBinContent/(maximumBinContent+c*9))

        if ((binX >= maximumBin) & (c>0)):
            weightHisto.SetBinContent(binX, maximumBinContent/(maximumBinContent+4*c))
        elif (c>0):
            weightHisto.SetBinContent(binX, maximumBinContent/(maximumBinContent+c))
        else:
            weightHisto.SetBinContent(binX, 1.)
    weightHisto.SetBinContent(0, weightHisto.GetBinContent(1))
    weightHisto.SetBinContent(weightHisto.GetNbinsX()+1, weightHisto.GetBinContent(weightHisto.GetNbinsX()))
    weightHisto.SetTitle("Weights distributions")
    canvas.SetLogy(1)
    canvas.SaveAs(outFolder+"/weights/weights.pdf")
    weights_train_original = weights_train
    weights_train_=[]
    for w,m in zip(weights_train, outY_train):
        weightBin = weightHisto.FindBin(m)
        addW = weightHisto.GetBinContent(weightBin)	# weights for importance
        weights_train_.append(abs(w)*addW)           	# weights of the training are the product of the original weights and the weights used to give more importance to regions with few events
    weights_train = np.array(weights_train_)		# final weights_train is np array

    weights_train = 1./np.mean(weights_train)*weights_train	# normalized by the mean


# *******************************
# *                             *
# *        Neural Network       *
# *                             *
# *******************************

# hyperparameters
    learningRate = learningRate_
    batchSize = batchSize_
    dropout = dropout_
    nDense = nDense_
    nNodes = nNodes_
    regRate = regRate_
    activation = activation_
    outputActivation = outputActivation_

    print ("getting model")
    model = getMRegModelFlat(regRate = regRate, activation = activation, dropout = dropout, nDense = nDense,
                                      nNodes = nNodes, inputDim = inX_train.shape[1], outputActivation = outputActivation)


    optimizer = tf.keras.optimizers.Adam(   lr = learningRate,
                                            beta_1=0.9, beta_2=0.999,   # memory lifetime of the first and second moment
                                            epsilon=1e-07,              # regularization constant to avoid divergences
                                            #weight_decay=None,
                                            #use_ema=False, bema_momentum=0.99, ema_overwrite_frequency=None,
                                            name="Adam"
                                            )

    print ("compiling model")
    model.compile(optimizer=optimizer, loss = tf.keras.losses.MeanSquaredError(), metrics=['mean_absolute_error','mean_absolute_percentage_error'])


    callbacks=[]
    #reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', patience=patiencelR_, factor=reduceLR_factor)
    # ADAM already optimizes the LR in each direction
# This callback monitors the validation loss and stops training if the validation loss does not improve for 300 epochs
    earlyStop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=patienceeS_, restore_best_weights=True)
# This callback saves the weights of the best performing model during training, based on the validation loss. The saved model is stored in the specified output folder with the name "model_best.h5"
    modelCheckpoint = tf.keras.callbacks.ModelCheckpoint(outFolder + '/model_best.h5', monitor='val_loss', save_best_only=True)


    callbacks.append(earlyStop)
    callbacks.append(modelCheckpoint)

    print ("fitting model")
    fit = model.fit(
            inX_train,
            outY_train,
            sample_weight = weights_train,
            validation_split = 0.25,
            batch_size = batchSize,
            epochs = epochs_,
            shuffle = False,
            callbacks = callbacks,
            verbose = 1)

    saveModel = not doEvaluate
    if saveModel:
        model.save(outFolder+"/"+modelName+".h5")
        tf.keras.backend.clear_session()
        tf.keras.backend.set_learning_phase(0)
        model = tf.keras.models.load_model(outFolder+"/"+modelName+".h5")
        print('inputs: ', [input.op.name for input in model.inputs])
        print('outputs: ', [output.op.name for output in model.outputs])


# Real prediction based on training
    y_predicted = model.predict(inX_test)
    np.save(dataPathFolderYear+"/testing/flat_regY"   + additionalName+"test.npy", y_predicted)
    y_predicted_train = model.predict(inX_train)

    if not os.path.exists(outFolder+"/"+year):
        os.makedirs(outFolder+"/"+year)

    #  "mean_absolute_error"
    plt.figure(0)
    plt.plot(fit.history['mean_absolute_error'][1:])
    plt.plot(fit.history['val_mean_absolute_error'][1:])
    plt.title('model mean_absolute_error')
    plt.ylabel('mean_absolute_error')
    plt.xlabel('epoch')
    # plt.yscale('log')
    plt.ylim(ymax = min(fit.history['mean_absolute_error'])*1.4, ymin = min(fit.history['mean_absolute_error'])*0.9)
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(outFolder+"/"+year+"/mean_absolute_error.pdf")
    #  "mean_absolute_percentage_error"
    plt.figure(1)
    plt.plot(fit.history['mean_absolute_percentage_error'][1:])
    plt.plot(fit.history['val_mean_absolute_percentage_error'][1:])
    plt.title('model mean_absolute_percentage_error')
    plt.ylabel('mean_absolute_percentage_error')
    plt.xlabel('epoch')
    # plt.yscale('log')
    plt.ylim(ymax = min(fit.history['mean_absolute_percentage_error'])*1.4, ymin = min(fit.history['mean_absolute_percentage_error'])*0.9)
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(outFolder+"/"+year+"/mean_absolute_percentage_error.pdf")
    # "Loss"
    plt.figure(2)
    plt.plot(fit.history['loss'][1:])
    plt.plot(fit.history['val_loss'][1:])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    # plt.yscale('log')
    plt.ylim(ymax = max(min(fit.history['loss']), min(fit.history['val_loss']))*1.4, ymin = min(min(fit.history['loss']),min(fit.history['val_loss']))*0.9)
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(outFolder+"/"+year+"/loss.pdf")

    corr = pearsonr(outY_test.reshape(outY_test.shape[0]), y_predicted.reshape(y_predicted.shape[0]))
    print("correlation:",corr[0])
# statistica distance: measures how one probab distr P is different from a second Q. In our case predicted output and expected one
    #kl = compute_kl_divergence(y_predicted, outY_test, n_bins=100)
    #print ("KL", kl)
# another way to say how similar are two distributions
    #js = compute_js_divergence(y_predicted, outY_test, n_bins=100)
    #print ("JS", js)
    doEvaluationPlots(outY_train, y_predicted_train, weights_train_original, lkrM_train, krM_train, year = year, outFolder = outFolder+"/train")
    print("Plots for testing")
    doEvaluationPlots(outY_test, y_predicted, weights_test, lkrM_test, krM_test, year = year, outFolder = outFolder+"/test")


    #print('Before defining class \'rho\'')
    #class_names=["mtt"]
    #print('After defining class \'rho\'')
    max_display = inX_test.shape[1]
# Commented till here
    for explainer, name  in [(shap.GradientExplainer(model,inX_test[:1000]),"GradientExplainer"),]:
        shap.initjs()
        print("... {0}: explainer.shap_values(X)".format(name))
        shap_values = explainer.shap_values(inX_test[:1000])
        print("... shap.summary_plot")
        plt.clf()
        shap.summary_plot(	shap_values, inX_test[:1000], plot_type="bar",
			        feature_names=featureNames,
     #       			class_names=class_names,
            			max_display=max_display, plot_size=[15.0,0.4*max_display+1.5], show=False)
        plt.savefig(outFolder+"/"+year+"/"+"shap_summary_{0}.pdf".format(name))


    
        #frozen_graph = freeze_session(tf.compat.v1.keras.backend.get_session(), output_names=[out.op.name for out in model.outputs])
        #tf.compat.v1.train.write_graph(frozen_graph, outFolder+ modelName+'.pbtxt', as_text=True)
        #tf.compat.v1.train.write_graph(frozen_graph, outFolder+ modelName+'.pb', as_text=False)
        #print ("Saved model to",outFolder+'/'+year+'/'+modelName+'.pbtxt/.pb/.h5')
def justEvaluate(dataPathFolder, additionalName, year, tokeep, modelDir, modelName, outFolder):
    inX_train, inX_test, outY_train, outY_test, weights_train, weights_test, lkrM_train, lkrM_test, krM_train, krM_test = loadData(
                                    dataPathFolder = dataPathFolder, year = year,
                                    additionalName = additionalName, testFraction = testFraction_,
                                    withBTag = True, pTEtaPhiMode=True,
                                    maxEvents = maxEvents_)


    featureNames = getFeatureNames()

    model = tf.keras.models.load_model(modelDir+"/"+modelName+".h5")
    y_predicted = model.predict(inX_test)

    doEvaluationPlots(outY_test, y_predicted, weights_test, lkrM_test, krM_test, year = year, outFolder = outFolder+"/test")

def main():
    print("********************************************\n*					   *\n*        Main function started             *\n*					   *\n********************************************")
    keep = None 
    
    if (doEvaluate):
        print("Calling justEvaluate ...")
        justEvaluate(dataPathFolder="/nfs/dust/cms/user/celottog/mttNN/npyData", additionalName="_ttbar", year="2016", tokeep = keep,
        	         modelDir = "/nfs/dust/cms/user/celottog/mttNN/outputs", modelName = "mttRegModel", outFolder="/nfs/dust/cms/user/celottog/mttNN/outputs")
    
    else:
       print("Calling doTrainingAndEvaluation ...")
       doTrainingAndEvaluation(dataPathFolder = "/nfs/dust/cms/user/celottog/mttNN/npyData", additionalName = "_ttbar", year ="2016", tokeep = keep,
				modelName = "mttRegModel", outFolder="/nfs/dust/cms/user/celottog/mttNN/outputs")
if __name__ == "__main__":
	main()
