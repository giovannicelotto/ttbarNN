import os
import numpy as np
from sklearn.model_selection import train_test_split
from utils.helpers import loadRegressionData, getFeatureNames
from utils.models import *
from utils.plot import *
from utils.stat import multiScale, standardScale, scaleNonAnalytical
from stat import *
from npyData.checkFeatures import checkFeatures
from sklearn.utils import shuffle
from tabulate import tabulate
from sklearn.impute import KNNImputer
import time
import pickle
def weightedStd(values, weights):
    """
    Return the weighted average and standard deviation.
    values, weights -- NumPy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return np.sqrt(variance)

def loadData(npyDataFolder , testFraction, maxEvents, minbjets, nFiles, outFolder, doubleNN=False, output=True, scale = 'standard'):
    '''
    Load data from saved numpy arrays or create them if not available (using loadRegressionData)
    '''
    
    print ("LoadData from " + npyDataFolder+"/[..].npy")
    
    createNewData = False
    if not os.path.exists(npyDataFolder+ "/inX.npy"):
        createNewData = True
        print("*** Not found the data in the directory "+npyDataFolder+ "/inX.npy")
    else:
        print("\nData found in the directory :" + npyDataFolder)
        createNewData = False
    
    if createNewData:
        
        path = "/nfs/dust/cms/user/celottog/CMSSW_10_6_32/src/TopAnalysis/Configuration/analysis/diLeptonic/miniTree_2018/Nominal/emu"
        inX, outY, weights, lkrM, krM, totGen, mask = loadRegressionData(path, "miniTree", nFiles=nFiles, minbjets=minbjets, maxEvents=maxEvents)

        inX     = np.array(inX)
        outY    = np.array(outY)
        weights = np.array(weights)
        lkrM    = np.array(lkrM)
        krM     = np.array(krM)
        totGen  = np.array(totGen)
        mask    = np.array(mask) # mask permette di passare da totGen a outY.
        print("\neventIn shape:\t", np.array(inX).shape)
        print("eventOut shape:\t",  np.array(outY).shape)
        print("totGen shape:\t",    np.array(totGen).shape)
        print("loose shape:\t",     np.array(lkrM).shape)
        print("krm shape:\t",       np.array(krM).shape)
        print("mask shape:\t",      np.array(mask).shape)
        
        assert  (len(inX)==len(outY)),      "same lenght %d %d" %(len(inX), len(outY))
        assert  (len(outY)==len(weights)),  "same lenght %d %d" %(len(outY), len(weights))
        assert  (len(weights)==len(lkrM)),  "same length %d %d" %(len(weights), len(krM))
        assert  (len(mask)==len(krM)),      "same length %d %d" %(len(mask), len(krM))
        assert  (len(totGen)==len(lkrM)),   "same lenght %d %d" %(len(totGen), len(mask))
        
        print("Shuffling before saving data...")
        assert (outY[mask, 0] == totGen[mask]).all(), "Mask does not match the previous mask. New assert 2505"
        inX, outY, weights, lkrM, krM, totGen, mask = shuffle(inX, outY, weights, lkrM, krM, totGen, mask, random_state = 1999)
        print("Check after shuffle data...")
        assert (outY[mask, 0] == totGen[mask]).all(), "Mask does not match the previous mask"
        
        
        if not os.path.exists(npyDataFolder):
            os.makedirs(npyDataFolder)
            print("Creating ", npyDataFolder)
        print("Saving files...")
        np.save(npyDataFolder+ "/inX.npy", inX)
        np.save(npyDataFolder+ "/outY.npy", outY)
        np.save(npyDataFolder+ "/weights.npy", weights)
        np.save(npyDataFolder+ "/lkrM.npy", lkrM)
        np.save(npyDataFolder+ "/krM.npy", krM)
        np.save(npyDataFolder+ "/totGen.npy", totGen)
        np.save(npyDataFolder+ "/mask.npy", mask)
        
    print("2. Loading files...")
    inX     = np.load(npyDataFolder+ "/inX.npy")
    outY    = np.load(npyDataFolder+ "/outY.npy")
    weights = np.load(npyDataFolder+ "/weights.npy")
    lkrM    = np.load(npyDataFolder+ "/lkrM.npy")
    krM     = np.load(npyDataFolder+ "/krM.npy")
    totGen  = np.load(npyDataFolder+ "/totGen.npy")
    mask    = np.load(npyDataFolder+ "/mask.npy")
    print(" Number of events passing cuts   {}/{} = {}".format( len(mask[mask==True]), len(mask), len(mask[mask==True])/len(mask)))
    print(" Number of events w/ Loose & Kin {}/{} = {}".format( (inX[:,0]>-998).sum(), len(inX), len(inX[inX[:,0]>-998])/len(inX)))
    print(" Number of recovered events      {}/{} = {}".format( (inX[:,0]<-4998).sum(), len(inX), (inX[:,0]<-4998).sum()*1./len(inX)))
    print(" Number of events + recovered    {}/{} = {}".format( ((inX[:,0]>-998) | (inX[:,0]<-4998)).sum(), len(inX), ((inX[:,0]>-998) | (inX[:,0]<-4998)).sum()/len(inX)))
    
    nan_values = np.isnan(inX)
    inf_values = np.isinf(inX)

    # Check if any NaN values exist in the array
    if np.any(nan_values):
        print("There are NaN values in the array.")
    else:
        print("There are no NaN values in the array.")
    if np.any(inf_values):
        print("There are inf values in the array.")
    else:
        print("There are no inf values in the array.")
    

    
# Print and plot checks
    dnnMask = (inX[:,0]>-998)  # exclude nan (kinReco or loose fail) and out of acceptance
    if (createNewData):
        print(" Control plots of all the data...")
        checkFeatures(inX[dnnMask,:], npyDataFolder)

    featureNames = getFeatureNames()
    data = []
    for i in range(len(featureNames)):
        data.append([featureNames[i], inX[dnnMask, i].min(), inX[:,i].max()])
    
    print(tabulate(data, headers=['Feature Name', 'Min', 'Max'], tablefmt='grid'))
    data = [['Weights', weights.min(), weights.max(), np.abs(weights).min()]]
    print(tabulate(data, headers=['Feature Name', 'Min', 'Max', 'Abs Min'], tablefmt='grid'))


    print("3. Shuffling after loading...")
    inX, outY, weights, lkrM, krM, totGen, mask, dnnMask = shuffle(inX, outY, weights, lkrM, krM, totGen, mask, dnnMask, random_state = 42)
    print("\n   Maximum of totGen:\t", totGen.max())
    print("   Minimum of totGen:\t", totGen.min())

    # to be defined better!
    '''if ((maxEvents is not None) & (createNewData == False)):
        ratio   = maxEvents/len(inX)
        print("maxev", maxEvents)
        #print("inX  ", inX)
        print("ratio",ratio)
        inX     = inX[:maxEvents]
        outY    = outY[:maxEvents]
        weights = weights[:maxEvents]
        lkrM    = lkrM[:int(round(ratio*(len(totGen)-1)))]
        krM     = krM[:int(round(ratio*(len(totGen)-1)))]
        totGen  = totGen[:int(round(ratio*(len(totGen)-1)))]'''

    
    
# Separation training and testing with a seed for random searches
    print("3. Splitting training and testing...")
    inX_train, inX_test, outY_train, outY_test, weights_train, weights_test, lkrM_train, lkrM_test, krM_train, krM_test, totGen_train, totGen_test, mask_train, mask_test = train_test_split(inX, outY, weights, lkrM, krM, totGen, mask, test_size = testFraction, random_state = 1998) #, 
    dnnMask_train = (inX_train[:,0]>-998)
    dnn2Mask_train = inX_train[:,0]<-4998
    dnn2Mask_test = inX_test[:,0]<-4998
    dnnMask_test = (inX_test[:,0]>-998)
    #assert (outY_train[mask_train, 0] == totGen_train[mask_train]).all(), "Mask does not match after splitting training and testing"
    print(" Element 90 test", outY_test[90])
    print(" Data splitted succesfully")
    print(" Number train+valid events (masked) :", inX_train[inX_train[:,0]>-998, :].shape[0])
    print(" Number test events        (masked) :", len(inX_test[inX_test[:,0]>-998, :]))
    print(" Number of features        :", inX_train.shape[1])
    

    print("4. Scaling training data...")
    featureNames = getFeatureNames()
    if (scale == 'standard'):
        inXs_train = standardScale(featureNames, inX_train[dnnMask_train,:], outFolder+'/scalers.pkl')
    elif (scale == 'multi'):
        inXs_train = multiScale(featureNames, inX_train[dnnMask_train,:], outFolder+'/scalers.pkl')
    if (output):
        checkFeatures(inXs_train[:,:], npyDataFolder, name="scaledPlot")
    inX_train[dnnMask_train,:] = inXs_train


    print("   Loading scalers and scaling the testing")

    dnnMask_test = (inX_test[:,0]>-998)
    with open(outFolder+"/scalers.pkl", 'rb') as file:
        scalers = pickle.load(file)

        if (scalers['type']=='multi'):
            maxer = scalers['maxer']
            powerer = scalers['powerer']
            scaler = scalers['scaler']
            maxable = scalers['maxable']
            powerable = scalers['powerable']
            scalable = scalers['scalable']

            inXs_test = inX_test[dnnMask_test, :]

            inXs_test[:, maxable]   = maxer.transform( inXs_test  [:, maxable])
            inXs_test[:, powerable] = powerer.transform( inXs_test[:, powerable])
            inXs_test[:, scalable]  = scaler.transform( inXs_test [:, scalable])
            inX_test[dnnMask_test, :] = inXs_test

        if (scalers['type']=='standard'):
            scaler = scalers['scaler']
            scalable = scalers['scalable']

            inXs_test = inX_test[dnnMask_test, :]

            inXs_test[:, scalable]  = scaler.transform( inXs_test [:, scalable])
            inX_test[dnnMask_test, :] = inXs_test

    # Second scaling
    if (doubleNN):
        dnn2Mask_train = inX_train[:,0]<-4998
        dnn2Mask_test = inX_test[:,0]<-4998
        inX_train[dnn2Mask_train, 15:], inX_test[dnn2Mask_test, 15:] = scaleNonAnalytical(getFeatureNames()[15:], inX_train, inX_test,  npyDataFolder, outFolder)



    if not os.path.exists(npyDataFolder+ "/testing"):
        print(" Creating "+npyDataFolder+ "/testing")
        os.makedirs(npyDataFolder+ "/testing")
    print(" Saving data for training and testing")
    if createNewData:
        np.save(npyDataFolder+ "/testing/inX"    + "_test.npy", inX_test)
        np.save(npyDataFolder+ "/testing/totGen" + "_test.npy", totGen_test)
        np.save(npyDataFolder+ "/testing/outY"   + "_test.npy", outY_test)
        np.save(npyDataFolder+ "/testing/weights"+ "_test.npy", weights_test)
        np.save(npyDataFolder+ "/testing/lkrM"   + "_test.npy", lkrM_test)
        np.save(npyDataFolder+ "/testing/krM"    + "_test.npy", krM_test)

    return inX_train, inX_test, outY_train, outY_test, weights_train, weights_test,lkrM_train, lkrM_test, krM_train, krM_test, totGen_train, totGen_test, mask_train, mask_test



































