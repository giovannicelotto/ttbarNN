import os
import numpy as np
from sklearn.model_selection import train_test_split
from utils.helpers import *
from utils.models import *
from utils.plot import *
from utils.stat import *
from stat import *
from npyData.checkFeatures import checkFeatures
import matplotlib.pyplot as plt
from array import array
from tensorflow import keras
import shap
from scipy.stats.stats import pearsonr
from scipy.spatial.distance import jensenshannon
from sklearn.utils import shuffle
from scipy.stats import gaussian_kde
from scipy.stats import entropy
from utils.helpersForPrediction import *


def loadData(dataPathFolder , testFraction, maxEvents, minbjets, nFiles_, output=True, scale = 'multi'):
    '''
    Load data from saved numpy arrays or create them if not available (using loadRegressionData)
    '''
    
    print ("LoadData from " + dataPathFolder+"/flat_[..].npy")
    print ("\t maxEvents    = "+str(maxEvents))
    print ("\t testFraction = "+str(testFraction))



    createNewData = False
    if not os.path.exists(dataPathFolder+ "/flat_inX.npy"):
        createNewData = True
        print("*** Not found the data in the directory "+dataPathFolder+ "/flat_inX.npy")
    else:
        print("\nData found in the directory :" + dataPathFolder)
        createNewData = False

    if createNewData:
        inX, outY, weights, lkrM, krM, totGen = loadRegressionData("/nfs/dust/cms/user/celottog/ttbarSignalFromDilepton/first10files/", "miniTree", nFiles=nFiles_, minbjets=minbjets, maxEvents=maxEvents)
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
    inX, outY, weights, lkrM, krM = shuffle(inX, outY, weights, lkrM, krM, random_state = 42)
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

    
    print("Scaling data")
    featureNames = getFeatureNames()
    if (scale == 'standard'):
        inXs = standardScale(featureNames, inX)
    elif (scale == 'multi'):
        inXs = multiScale(featureNames, inX)
    if (output):
        checkFeatures(inXs, dataPathFolder, name="scaledPlot")

    print('\n\nShapes of all the data at my disposal:\nInput  \t',inX.shape,'\nOutput\t', outY.shape,'\nWeights\t', weights.shape,'\nLoose M\t', lkrM.shape,'\nFull M\t', krM.shape, '\ntotGen\t', totGen.shape)
    
# Separation training and testing with a seed for random searches
    inX_train, inX_test, outY_train, outY_test, weights_train, weights_test, lkrM_train, lkrM_test, krM_train, krM_test = train_test_split(inXs, outY, weights, lkrM, krM, test_size = testFraction, random_state = 1998)
    print("Element 90 test", outY_test[90])
    print ("Data splitted succesfully")
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



def loadDataForPrediction(dataPathFolder, minbjets, output=True, scale = 'multi'):
    '''
    Load data from saved numpy arrays or create them if not available (using loadRegressionDataForPrediction)
    '''
    
    print ("LoadData from "+dataPathFolder+"/inX.npy")

    createNewData = False
    if not os.path.exists(dataPathFolder+ "/inX.npy"):
        createNewData = True
        print("*** Not found the data in the directory "+dataPathFolder+ "/inX.npy")
    else:
        print("\nData found in the directory :" + dataPathFolder)
        createNewData = False

    if createNewData:
        inX, weights, lkrM, krM = loadRegressionDataForPrediction(dataPathFolder, "miniTree", minbjets=minbjets)
        inX     = np.array(inX)
        weights = np.array(weights)
        lkrM    = np.array(lkrM)
        krM     = np.array(krM)
        
        inX, weights, lkrM, krM = shuffle(inX, weights, lkrM, krM, random_state = 1999)
        print("Number training events :", inX.shape[0], len(inX))
        if not os.path.exists(dataPathFolder):
            os.makedirs(dataPathFolder)
        np.save(dataPathFolder+ "/inX.npy", inX)
        np.save(dataPathFolder+ "/weights.npy", weights)
        np.save(dataPathFolder+ "/lkrM.npy", lkrM)
        np.save(dataPathFolder+ "/krM.npy", krM)
        print("Control plots")
        checkFeatures(inX, dataPathFolder)

    
    inX     = np.load(dataPathFolder+ "/inX.npy")
    weights = np.load(dataPathFolder+ "/weights.npy")
    lkrM    = np.load(dataPathFolder+ "/lkrM.npy")
    krM     = np.load(dataPathFolder+ "/krM.npy")

      
    print("Scaling data")
    featureNames = getFeatureNames()
    if (scale == 'standard'):
        inXs = standardScale(featureNames, inX)
    elif (scale == 'multi'):
        inXs = multiScale(featureNames, inX)
    if (output):
        checkFeatures(inXs, dataPathFolder, name="scaledPlot")

    print('\n\nShapes of all the data at my disposal:\nInput  \t',inX.shape,'\nWeights\t', weights.shape,'\nLoose M\t', lkrM.shape,'\nFull M\t', krM.shape)
    

    return inX, weights, lkrM, krM