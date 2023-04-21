import os
import numpy as np
from sklearn.model_selection import train_test_split
from utils.helpers import *
from utils.models import *
from utils.plot import *
from utils.stat import *
from stat import *
from npyData.checkFeatures import checkFeatures
from sklearn.utils import shuffle


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
        
        path = "/nfs/dust/cms/user/celottog/ttbarSignalFromDilepton/first10files/"
        inX, outY, weights, lkrM, krM, totGen, mask = loadRegressionData(path, "miniTree", nFiles=nFiles_, minbjets=minbjets, maxEvents=maxEvents)

        inX     = np.array(inX)
        outY    = np.array(outY)
        weights = np.array(weights)
        lkrM    = np.array(lkrM)
        krM     = np.array(krM)
        totGen  = np.array(totGen)
        mask    = np.array(mask) # mask permette di passare da totGen a outY.
        print("\neventIn shape:\t", np.array(inX).shape)
        print("eventOut shape:\t",  np.array(outY).shape)
        print("totGen shape:\t",  np.array(totGen).shape)
        print("loose shape:\t", np.array(lkrM).shape)
        print("krm shape:\t",  np.array(krM).shape)
        print("mask shape:\t",  np.array(mask).shape)
        
        assert(len(inX)==len(outY)), "same lenght"
        assert (len(outY)==len(weights)), "same lenght"
        assert (len(weights)==len(lkrM)), "same length"
        assert (len(mask)==len(krM)), "same length"
        assert(len(totGen)==len(lkrM)), "same lenght"
        
        print("Shuffling before saving data...")
        assert (outY[mask, 0] == totGen[mask]).all(), "Mask does not match the previous mask"
        inX, outY, weights, lkrM, krM, totGen, mask = shuffle(inX, outY, weights, lkrM, krM, totGen, mask, random_state = 1999)
        print("Check after shuffle data...")
        assert (outY[mask, 0] == totGen[mask]).all(), "Mask does not match the previous mask"
        
        print("Number of events passing cuts {}/{} = {}".format( len(mask[mask==True]), len(mask), len(mask[mask==True])/len(mask)))
        if not os.path.exists(dataPathFolder):
            os.makedirs(dataPathFolder)
            print("Creating ", dataPathFolder)
        print("Saving files...")
        np.save(dataPathFolder+ "/flat_inX.npy", inX)
        np.save(dataPathFolder+ "/flat_outY.npy", outY)
        np.save(dataPathFolder+ "/flat_weights.npy", weights)
        np.save(dataPathFolder+ "/flat_lkrM.npy", lkrM)
        np.save(dataPathFolder+ "/flat_krM.npy", krM)
        np.save(dataPathFolder+ "/flat_totGen.npy", totGen)
        np.save(dataPathFolder+ "/flat_mask.npy", mask)
        print("Control plots...")
        checkFeatures(inX[mask,:], dataPathFolder)
    print("Loading files...")
    inX     = np.load(dataPathFolder+ "/flat_inX.npy")
    outY    = np.load(dataPathFolder+ "/flat_outY.npy")
    weights = np.load(dataPathFolder+ "/flat_weights.npy")
    lkrM    = np.load(dataPathFolder+ "/flat_lkrM.npy")
    krM     = np.load(dataPathFolder+ "/flat_krM.npy")
    totGen  = np.load(dataPathFolder+ "/flat_totGen.npy")
    mask  = np.load(dataPathFolder+ "/flat_mask.npy")
    print("Shuffling after loading...")
    inX, outY, weights, lkrM, krM, totGen, mask = shuffle(inX, outY, weights, lkrM, krM, totGen, mask, random_state = 42)
    print("\nMaximum of totGen:\t", totGen.max())
    print("Minimum of totGen:\t", totGen.min())

    # to be defined better!
    if ((maxEvents is not None) & (createNewData == False)):
        ratio   = maxEvents/len(inX)
        print("maxev", maxEvents)
        #print("inX  ", inX)
        print("ratio",ratio)
        inX     = inX[:maxEvents]
        outY    = outY[:maxEvents]
        weights = weights[:maxEvents]
        lkrM    = lkrM[:int(round(ratio*(len(totGen)-1)))]
        krM     = krM[:int(round(ratio*(len(totGen)-1)))]
        totGen  = totGen[:int(round(ratio*(len(totGen)-1)))]

    
    print("Scaling data...")
    featureNames = getFeatureNames()
    if (scale == 'standard'):
        inXs = standardScale(featureNames, inX[mask,:])
    elif (scale == 'multi'):
        inXs = multiScale(featureNames, inX[mask,:], dataPathFolder = dataPathFolder)
    if (output):
        checkFeatures(inXs, dataPathFolder, name="scaledPlot")
    inX[mask,:] = inXs
    
# Separation training and testing with a seed for random searches
    print("Splitting training and testing...")
    inX_train, inX_test, outY_train, outY_test, weights_train, weights_test, lkrM_train, lkrM_test, krM_train, krM_test, totGen_train, totGen_test, mask_train, mask_test = train_test_split(inX, outY, weights, lkrM, krM, totGen, mask, test_size = testFraction, random_state = 1998)
    assert (outY_train[mask_train, 0] == totGen_train[mask_train]).all(), "Mask does not match after splitting training and testing"
    print("Element 90 test", outY_test[90])
    print ("Data splitted succesfully")
    print("Number train+valid events (masked) :", inX_train[mask_train, :].shape[0])
    print("Number test events        (masked) :", len(inX_test[mask_test, :]))
    print("Number of features        :", inX_train.shape[1])
    
    if not os.path.exists(dataPathFolder+ "/testing"):
        print("Creating "+dataPathFolder+ "/testing")
        os.makedirs(dataPathFolder+ "/testing")
    print("Saving data for training and testing")
    if createNewData:
        np.save(dataPathFolder+ "/testing/flat_inX"    + "_test.npy", inX_test)
        np.save(dataPathFolder+ "/testing/flat_outY"   + "_test.npy", outY_test)
        np.save(dataPathFolder+ "/testing/flat_weights"+ "_test.npy", weights_test)
        np.save(dataPathFolder+ "/testing/flat_lkrM"   + "_test.npy", lkrM_test)
        np.save(dataPathFolder+ "/testing/flat_krM"    + "_test.npy", krM_test)

    return inX_train, inX_test, outY_train, outY_test, weights_train, weights_test,lkrM_train, lkrM_test, krM_train, krM_test, totGen_train, totGen_test, mask_train, mask_test



































