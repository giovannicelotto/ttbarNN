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
from sklearn.utils import shuffle
from scipy.stats import gaussian_kde
from scipy.stats import entropy
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'        # TensorFlow will only display error messages and suppress all other messages including these warnings.

nFiles_             = 3
maxEvents_          = 100001
testFraction_       = 0.3
validation_split_   = 0.3
epochs_             = 100       # 5000
learningRate_       = 0.01     # 10^-3
batchSize_          = 128        
nDense_             = 2   
nNodes_             = 64   
regRate_            = 0.01
activation_         = 'relu'
outputActivation_   = 'linear'
patienceeS_         = 15
#patiencelR_         = 30
reduceLR_factor     = 0.
dropout_            = 0.     
doEvaluate          = False
additionalName_     = "_complex"
exp_tau             = 150


def KLdiv(p, q):
    xs_ = np.linspace(340, 800, 50)
    kde_p = gaussian_kde(p)(xs_) 
    kde_q = gaussian_kde(q)(xs_) 
    kl_divergence = entropy(kde_p, kde_q)
    return kl_divergence

def JSdist(p, q):
    xs_ = np.linspace(340, 800, 50)
    kde_p = gaussian_kde(p)(xs_) 
    kde_q = gaussian_kde(q)(xs_) 
    
    m = (kde_p + kde_q) / 2

    return [np.sqrt(entropy(kde_p, m)/2 + entropy(kde_q, m)/2), jensenshannon(kde_p, kde_q)]


def loadData(dataPathFolder , additionalName, testFraction, withBTag, pTEtaPhiMode, maxEvents):
    '''
    Load data from saved numpy arrays or create them if not available (using loadRegressionData)
    '''
    
    print ("LoadData from "+dataPathFolder+"/flat_[..]"+additionalName+".npy")
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
    print ("\t expTau       = "+str(exp_tau))


    createNewData = False
    if not os.path.exists(dataPathFolder+"/"+additionalName+ "/flat_inX"+additionalName+".npy"):
        createNewData = True
        print("*** Not found the data in the directory "+dataPathFolder+"/"+additionalName+ "/flat_inX"+additionalName+".npy")
        print("***           Producing new data              ***")
    else:
        print("\nData found in the directory :" + dataPathFolder)
        createNewData = False

    if createNewData:
        minJets=2
        print ("\nNew data will be loaded with settings: nJets >= "+str(minJets)+"; max Events = "+str(maxEvents))
        inX, outY, weights, lkrM, krM = loadRegressionData("/nfs/dust/cms/user/celottog/ttbarSignalFromDilepton", "miniTree", nFiles=nFiles_,  withBTag=withBTag, pTEtaPhiMode=pTEtaPhiMode)
        inX     = np.array(inX)
        outY    = np.array(outY)
        weights = np.array(weights)
        lkrM    = np.array(lkrM)
        krM     = np.array(krM)
        #pt = PowerTransformer(method='box-cox', standardize=True)
        #quantile_transformer = QuantileTransformer(output_distribution='normal', random_state=0)
        #inX[:,0] = np.ndarray.flatten(pt.fit_transform(inX[:,0].reshape(-1, 1)))
        #inX[:,1] = np.ndarray.flatten(pt.fit_transform(inX[:,1].reshape(-1, 1)))
        #inX[:,2] = np.ndarray.flatten(quantile_transformer.fit_transform(inX[:,2].reshape(-1, 1)))

        inX, outY, weights, lkrM, krM = shuffle(inX, outY, weights, lkrM, krM, random_state = 1999)
        print("Number training events before [:maxEvents]:", inX.shape[0], len(inX))
        if not os.path.exists(dataPathFolder+"/"+additionalName):
            os.makedirs(dataPathFolder+"/"+additionalName)
        np.save(dataPathFolder+"/"+additionalName+ "/flat_inX"+additionalName+".npy", inX)
        np.save(dataPathFolder+"/"+additionalName+ "/flat_outY"+additionalName+".npy", outY)
        np.save(dataPathFolder+"/"+additionalName+ "/flat_weights"+additionalName+".npy", weights)
        np.save(dataPathFolder+"/"+additionalName+ "/flat_lkrM"+additionalName+".npy", lkrM)
        np.save(dataPathFolder+"/"+additionalName+ "/flat_krM"+additionalName+".npy", krM)

    print("*** inX, outY, weights, lkrM, krM loading")
    inX     = np.load(dataPathFolder+"/"+additionalName+ "/flat_inX"+additionalName+".npy")
    outY    = np.load(dataPathFolder+"/"+additionalName+ "/flat_outY"+additionalName+".npy")
    weights = np.load(dataPathFolder+"/"+additionalName+ "/flat_weights"+additionalName+".npy")
    lkrM    = np.load(dataPathFolder+"/"+additionalName+ "/flat_lkrM"+additionalName+".npy")
    krM     = np.load(dataPathFolder+"/"+additionalName+ "/flat_krM"+additionalName+".npy")
    print("*** inX, outY, weights, lkrM, krM loaded")

    if maxEvents is not None:
        inX     = inX[:maxEvents]
        outY    = outY[:maxEvents]
        weights = weights[:maxEvents]
        lkrM    = lkrM[:maxEvents]
        krM     = krM[:maxEvents]

    print('\n\nShapes of all the data at my disposal:\nInput  \t',inX.shape,'\nOutput\t', outY.shape,'\nWeights\t', weights.shape,'\nLoose M\t', lkrM.shape,'\nFull M\t', krM.shape)
    
    #pt = PowerTransformer(method='box-cox', standardize=True)
    #qt = QuantileTransformer(output_distribution='normal', random_state=0)
    #inX[:,] = qt.fit_transform(inX)

    inX_train, inX_test, outY_train, outY_test, weights_train, weights_test, lkrM_train, lkrM_test, krM_train, krM_test = train_test_split(inX, outY, weights, lkrM, krM, test_size = testFraction, random_state = 1999)

    print ("\tData splitted succesfully")
    print("Number training events :", inX_train.shape[0], len(inX_train))
    print("Number of features     :", inX_train.shape[1])
    print("loadData ended returning inX_train and so on...")
    if not os.path.exists(dataPathFolder+"/"+additionalName+ "/testing"):
        os.makedirs(dataPathFolder+"/"+additionalName+ "/testing")
    
    if createNewData:
        np.save(dataPathFolder+"/"+additionalName+ "/testing/flat_inX"    + additionalName+"_test.npy", inX_test)
        np.save(dataPathFolder+"/"+additionalName+ "/testing/flat_outY"   + additionalName+"_test.npy", outY_test)
        np.save(dataPathFolder+"/"+additionalName+ "/testing/flat_weights"+ additionalName+"_test.npy", weights_test)
        np.save(dataPathFolder+"/"+additionalName+ "/testing/flat_lkrM"   + additionalName+"_test.npy", lkrM_test)
        np.save(dataPathFolder+"/"+additionalName+ "/testing/flat_krM"    + additionalName+"_test.npy", krM_test)

    return inX_train, inX_test, outY_train, outY_test, weights_train, weights_test,lkrM_train, lkrM_test, krM_train, krM_test

def doTrainingAndEvaluation(modelName, additionalName, outFolder, dataPathFolder = "/nfs/dust/cms/user/celottog/mttNN/npyData"):
    
    inX_train, inX_test, outY_train, outY_test, weights_train, weights_test, lkrM_train, lkrM_test, krM_train, krM_test = loadData(
                                    dataPathFolder = dataPathFolder, 
                                    additionalName = additionalName, testFraction = testFraction_,
                                    withBTag = True, pTEtaPhiMode=True,
                                    maxEvents = maxEvents_)
    # Create the output folder if it does not exist
    if not os.path.exists(outFolder):
        os.makedirs(outFolder)

    print(" Check element number 89 of inX_test", inX_test[89,0],"\n")      


    #feature_names, inX_train, inX_test = helpers.getReducedFeatureNamesAndInputs(inX_train, inX_test, tokeep=tokeep)
    featureNames = getFeatureNames()
    print('Features:', featureNames)



# *******************************
# *                             *
# *        Weight part          *
# *                             *
# *******************************

    weightBins = array('d', [340, 342.5, 345, 347.5, 350, 355, 360, 370, 380, 390, 400, 410, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600 ]) # 
    wegihtNBin = len(weightBins)-1
    weightHisto = ROOT.TH1F("weightHisto","weightHisto",wegihtNBin, weightBins)

    print("Filling histo with m_tt")
    for m, weight in zip(outY_train, weights_train):
        weightHisto.Fill(m,abs(weight))
    weightHisto = NormalizeBinWidth1d(weightHisto)

    
    '''xs_ = np.linspace(340, 800, 1000)
    kde = gaussian_kde(outY_train[:,0])(xs_)        
    kde = kde/np.max(kde)                           # set the max od the kde to 1
    xsmax = xs_[np.argmax(kde)]
    myexp = np.piecewise(xs_, [xs_<xsmax, xs_>=xsmax], [1, np.exp(-(xs_-xsmax)[xs_>=xsmax]/200)])
    #kde_prime = np.abs(kde[2:]-kde[:len(kde)-2]/(2*(xs_[1]-xs_[0])))
    #kde_sec = np.abs(kde[2:]-2*kde[1:len(kde)-1]+kde[:len(kde)-2]/(xs_[1]-xs_[0])**2)

    fig, ax = plt.subplots(1, 1)
    #ax.plot(xs_, myexp)
    ax.plot(xs_, (1/kde)*myexp)
    #ax.plot(xs_, (1/kde)*myexp)
    #ax.plot(xs_[1:len(xs_)-1], kde_prime)
    #ax.plot(xs_[1:len(xs_)-1], kde_sec)
    #ax.plot(xs_, 0.004*0.004*1/(0.004+kde), linewidth=0.5, marker='o', markersize=1)
    fig.savefig(outFolder+"/model/kde.pdf")
    plt.cla()'''
    if not os.path.exists(outFolder+ "/model"):
        os.makedirs(outFolder+ "/model")
    canvas = ROOT.TCanvas()
    weightHisto.Draw("hist")
    weightHisto.SetTitle("Normalized and weighted m_{tt} distibutions. Training")
    weightHisto.SetYTitle('Normalized Counts')
    canvas.SaveAs(outFolder + "/model/mttOriginalWeight.pdf")

    weightHisto.Scale(1./weightHisto.Integral())
    maximumBin = weightHisto.GetMaximumBin()
    maximumBinContent = weightHisto.GetBinContent(maximumBin)

    firstMidpoint = (weightHisto.GetXaxis().GetBinLowEdge(1)+weightHisto.GetXaxis().GetBinUpEdge(1))/2
    #print("firstMidpoint: ", firstMidpoint)
    for binX in range(1,weightHisto.GetNbinsX()+1):
        midpoint = (weightHisto.GetXaxis().GetBinLowEdge(binX) + weightHisto.GetXaxis().GetBinUpEdge(binX))/2
        #print("Current midpoint:\t", midpoint)
        c = weightHisto.GetBinContent(binX)
        if (c>0):
            weightHisto.SetBinContent(binX, maximumBinContent/(c)*np.exp(-(midpoint-firstMidpoint)/exp_tau))#
        #if ((binX >= maximumBin+10) & (c>0)):
        #    weightHisto.SetBinContent(binX, 1)
        #elif (c>0):
        #    weightHisto.SetBinContent(binX, maximumBinContent/c)
        else:
            weightHisto.SetBinContent(binX, 1.)     
    weightHisto.SetBinContent(0, weightHisto.GetBinContent(1))
    weightHisto.SetBinContent(weightHisto.GetNbinsX()+1, weightHisto.GetBinContent(weightHisto.GetNbinsX()))
    weightHisto.SetTitle("Weights distributions")
    canvas.SetLogy(1)
    canvas.SaveAs(outFolder+"/model/weights.pdf")
    weights_train_original = weights_train
    weights_train_=[]
    for w,m in zip(weights_train, outY_train):
        weightBin = weightHisto.FindBin(m)
        addW = weightHisto.GetBinContent(weightBin)	    # weights for importance
        weights_train_.append(abs(w)*addW)           	# weights of the training are the product of the original weights and the weights used to give more importance to regions with few events
    weights_train = np.array(weights_train_)		    # final weights_train is np array

    weights_train = 1./np.mean(weights_train)*weights_train



# *******************************
# *                             *
# *        Neural Network       *
# *                             *
# *******************************

    print ("getting model")
    model = getMRegModelFlat(regRate = regRate_, activation = activation_, dropout = dropout_, nDense = nDense_,
                                      nNodes = nNodes_, inputDim = inX_train.shape[1], outputActivation = outputActivation_)


    optimizer = tf.keras.optimizers.Adam(   lr = learningRate_,
                                            beta_1=0.9, beta_2=0.999,   # memory lifetime of the first and second moment
                                            epsilon=1e-07,              # regularization constant to avoid divergences
                                            #weight_decay=None,
                                            #use_ema=False, bema_momentum=0.99, ema_overwrite_frequency=None,
                                            name="Adam"
                                            )

    print ("compiling model")
    #model.compile(optimizer=optimizer, loss = tf.keras.losses.MeanSquaredError(), metrics=['mean_absolute_error','mean_absolute_percentage_error'])
    model.compile(optimizer=optimizer, loss = tf.keras.losses.MeanSquaredError())


    callbacks=[]
    #reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', patience=patiencelR_, factor=reduceLR_factor)
    earlyStop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=patienceeS_, verbose = 1, restore_best_weights=True)
    #modelCheckpoint = tf.keras.callbacks.ModelCheckpoint(outFolder + '/model_best.h5', monitor='val_loss', save_best_only=True)
    callbacks.append(earlyStop)
    #callbacks.append(modelCheckpoint)
    
    divisor = int ((1-validation_split_)*len(inX_train))
    outY_valid      = outY_train[divisor:]
    inX_valid       = inX_train[divisor:]
    weights_valid   = weights_train[divisor:]
    inX_train       = inX_train[:divisor]
    outY_train      = outY_train[:divisor]
    weights_train   = weights_train[:divisor]

    jsTV = JSdist(inX_train[:len(outY_valid)-1,0], inX_valid[:len(outY_valid)-1,0])
    print ("JS between LooseR training and validation", str(jsTV[0])[:6], str(jsTV[1])[:6])    
    jsTV = JSdist(inX_train[:len(outY_valid)-1,1], inX_valid[:len(outY_valid)-1,1])
    print ("JS between KinR training and validation", str(jsTV[0])[:6], str(jsTV[1])[:6])       

    np.save(dataPathFolder+"/"+additionalName+"/testing/flat_inX"    + additionalName+"_train.npy", inX_train)
    np.save(dataPathFolder+"/"+additionalName+"/testing/flat_outY"   + additionalName+"_train.npy", outY_train)
    np.save(dataPathFolder+"/"+additionalName+"/testing/flat_inX"    + additionalName+"_valid.npy", inX_valid)
    np.save(dataPathFolder+"/"+additionalName+"/testing/flat_outY"   + additionalName+"_valid.npy", outY_valid)
        

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
            shuffle = False,
            sample_weight = weights_train
            )

    saveModel = not doEvaluate
    if saveModel:
        model.save(outFolder+"/model/"+modelName+".h5")
        tf.keras.backend.clear_session()
        tf.keras.backend.set_learning_phase(0)
        model = tf.keras.models.load_model(outFolder+"/model/"+modelName+".h5")
        print('inputs: ', [input.op.name for input in model.inputs])
        print('outputs: ', [output.op.name for output in model.outputs])


# Real prediction based on training
    y_predicted = model.predict(inX_test)
    y_predicted_train = model.predict(inX_train)
    y_predicted_valid = model.predict(inX_valid)
    
    np.save(dataPathFolder+"/"+additionalName+"/testing/flat_regY"   + additionalName+"_test.npy", y_predicted)
    np.save(dataPathFolder+"/"+additionalName+"/testing/flat_regY"   + additionalName+"_train.npy", y_predicted_train)
    np.save(dataPathFolder+"/"+additionalName+"/testing/flat_regY"   + additionalName+"_valid.npy", y_predicted_valid)

    doPlotLoss(fit = fit, outFolder=outFolder+"/model")
    

    corr = pearsonr(outY_test.reshape(outY_test.shape[0]), y_predicted.reshape(y_predicted.shape[0]))
    print("correlation:",corr[0])
    
    kl = KLdiv(y_predicted[:,0], outY_test[:,0])
    print ("KL", kl)
    js = JSdist(y_predicted[:,0], outY_test[:,0])
    print ("JS", str(js[0])[:6], str(js[1])[:6])
    with open(outFolder+"/model/Info.txt", "w") as f:
        f.write("JS\t"+ str(js[0])[:6]+"\n"+"KL\t"+ str(kl)[:6])
        print(model.summary(), file=f)

        

    doEvaluationPlots(outY_train, y_predicted_train, weights_train_original, lkrM_train[:divisor], krM_train[:divisor], outFolder = outFolder+"/train")
    print("Plots for testing")
    doEvaluationPlots(outY_test, y_predicted, weights_test, lkrM_test, krM_test, outFolder = outFolder+"/test")


    max_display = inX_test.shape[1]
    for explainer, name  in [(shap.GradientExplainer(model,inX_test[:1000]),"GradientExplainer"),]:
        shap.initjs()
        print("... {0}: explainer.shap_values(X)".format(name))
        shap_values = explainer.shap_values(inX_test[:1000])
        print("... shap.summary_plot")
        plt.clf()
        shap.summary_plot(	shap_values, inX_test[:1000], plot_type="bar",
			            feature_names=featureNames,
            			max_display=max_display, plot_size=[15.0,0.4*max_display+1.5], show=False)
        plt.savefig(outFolder+"/model/"+"shap_summary_{0}.pdf".format(name))


    
        #frozen_graph = freeze_session(tf.compat.v1.keras.backend.get_session(), output_names=[out.op.name for out in model.outputs])
        #tf.compat.v1.train.write_graph(frozen_graph, outFolder+ modelName+'.pbtxt', as_text=True)
        #tf.compat.v1.train.write_graph(frozen_graph, outFolder+ modelName+'.pb', as_text=False)
        #print ("Saved model to",outFolder+'/'+year+'/'+modelName+'.pbtxt/.pb/.h5')



def justEvaluate(dataPathFolder, additionalName, modelDir, modelName, outFolder):
    inX_train, inX_test, outY_train, outY_test, weights_train, weights_test, lkrM_train, lkrM_test, krM_train, krM_test = loadData(
                                    dataPathFolder = dataPathFolder,
                                    additionalName = additionalName, testFraction = testFraction_,
                                    withBTag = True, pTEtaPhiMode=True,
                                    maxEvents = maxEvents_)


    model = tf.keras.models.load_model(modelDir+"/"+modelName+".h5")
    y_predicted = model.predict(inX_test)

    doEvaluationPlots(outY_test, y_predicted, weights_test, lkrM_test, krM_test, outFolder = outFolder+"/test")

def main():
    print("********************************************\n*					   *\n*        Main function started             *\n*					   *\n********************************************")
    keep = None 
    
    if (doEvaluate):
        print("Calling justEvaluate ...")
        justEvaluate(dataPathFolder="/nfs/dust/cms/user/celottog/mttNN/npyData", additionalName = additionalName_, 
        	         modelDir = "/nfs/dust/cms/user/celottog/mttNN/outputs/"+additionalName_+"/model", modelName = "mttRegModel", outFolder="/nfs/dust/cms/user/celottog/mttNN/outputs/"+additionalName_)
    
    else:
       print("Calling doTrainingAndEvaluation ...")
       doTrainingAndEvaluation(dataPathFolder = "/nfs/dust/cms/user/celottog/mttNN/npyData", additionalName = additionalName_, 
				modelName = "mttRegModel", outFolder="/nfs/dust/cms/user/celottog/mttNN/outputs/"+additionalName_)
if __name__ == "__main__":
	main()
