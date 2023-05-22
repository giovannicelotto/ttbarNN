import os
import numpy as np
from utils.helpers4P import *
from utils.stat import *    # for scaling
from stat import *
from npyData.checkFeatures import checkFeatures
from sklearn.utils import shuffle
from utils.helpers import getFeatureNames
import pickle

def loadData4P(dataPathFolder, minbjets, scalerPath, scaler2Path, output=True, nFiles = None):
    print("loadData called")
    '''
    Load Run2 data from saved numpy arrays or create them if not available (using loadRegressionData4P)
    dataPathFolder = folder where npy data are searched. If not found, create npydata from the folder above it where from root files. npy files are saved in datapathFolder
    minbjets = min number bjets required in the creation of data
    scalerPath = path where to take the scalers (saved during the training of NN)
    output = decide id producing a plot of scaled quantities.
    nFiles = not used here leave it None
    
    '''
# check for already existing data    
    print ("LoadData from "+dataPathFolder+"/inX.npy")
    createNewData = False
    if not os.path.exists(dataPathFolder+ "/inX.npy"):
        createNewData = True
        print("*** Not found the data in the directory "+dataPathFolder+ "/inX.npy")
    else:
        print("\nData found in the directory :" + dataPathFolder)
        createNewData = False
# otherwise create tehm
    if createNewData:
        inX, weights, lkrM, krM = loadRegressionData4P(path = dataPathFolder+"/..", treeName = "miniTree", minbjets=minbjets, nFiles = nFiles)
        inX     = np.array(inX)
        weights = np.array(weights)
        lkrM    = np.array(lkrM)
        krM     = np.array(krM)
# inX loose and full vanno maskati. hanno tutti la stessa luinghezza=event passano tagli comuni. se un evento non passa un taglio particolare in quell'array e meso a -999        
        assert len(inX)==len(lkrM)==len(krM)==len(weights), "Check lengths -- data4P"
        inX, weights, lkrM, krM = shuffle(inX, weights, lkrM, krM, random_state = 1999)
        #print(inX)
        #print(weights)
        dnnMask = (inX[:,0]>-998)
        
        
        if not os.path.exists(dataPathFolder):
            os.makedirs(dataPathFolder)
        np.save(dataPathFolder+ "/inX.npy", inX)
        np.save(dataPathFolder+ "/weights.npy", weights)
        np.save(dataPathFolder+ "/lkrM.npy", lkrM)
        np.save(dataPathFolder+ "/krM.npy", krM)
        print("Control plots")
        checkFeatures(inX[dnnMask,:], dataPathFolder)

# Load data and scale them
    inX     = np.load(dataPathFolder+ "/inX.npy")
    weights = np.load(dataPathFolder+ "/weights.npy")
    lkrM    = np.load(dataPathFolder+ "/lkrM.npy")
    krM     = np.load(dataPathFolder+ "/krM.npy")
    
    dnnMask  = (inX[:,0]>-998)
    dnn2Mask = (inX[:,0]<-4998)
    print("Number events passing DNN1 requiremenet:", dnnMask.sum())
    print("Number events passing DNN2 requiremenet:", dnn2Mask.sum())
    print("Number events passing Loose requiremenet:", len(lkrM[lkrM>-998]))
    print("Number events passing Full requiremenet:", len(krM[krM>-998]))

    
    print("Loading the scalers...")
    # First scaling
    with open(scalerPath, 'rb') as file:
        scalers = pickle.load(file)

        if (scalers['type']=='multi'):
            print("Multi scalers opened")
            maxer = scalers['maxer']
            powerer = scalers['powerer']
            scaler = scalers['scaler']
            maxable = scalers['maxable']
            powerable = scalers['powerable']
            scalable = scalers['scalable']

            inXs = inX[dnnMask, :]
            inXs[:, maxable]   = maxer.transform( inXs  [:, maxable])
            inXs[:, powerable] = powerer.transform( inXs[:, powerable])
            inXs[:, scalable]  = scaler.transform( inXs [:, scalable])
            inX[dnnMask, :] = inXs

        if (scalers['type']=='standard'):
            scaler = scalers['scaler']
            scalable = scalers['scalable']

            inXs = inX[dnnMask, :]
            inXs[:, scalable]  = scaler.transform( inXs [:, scalable])
            inX[dnnMask, :] = inXs
        if (output):
            checkFeatures(inXs, dataPathFolder, name="scaledPlot")
    
            


    # Second scaling:
    with open(scaler2Path, 'rb') as file:
        scalers = pickle.load(file)

        if (scalers['type']=='multi'):
            print("Multi scalers opened")
            maxer = scalers['maxer']
            powerer = scalers['powerer']
            scaler = scalers['scaler']
            maxable = scalers['maxable']
            powerable = scalers['powerable']
            scalable = scalers['scalable']

            inXs = inX[dnn2Mask, 14:]
            inXs[:, maxable]   = maxer.transform( inXs  [:, maxable])
            inXs[:, powerable] = powerer.transform( inXs[:, powerable])
            inXs[:, scalable]  = scaler.transform( inXs [:, scalable])
            inX[dnn2Mask, 14:] = inXs

        if (scalers['type']=='standard'):
            scaler = scalers['scaler']
            scalable = scalers['scalable']

            inXs = inX[dnn2Mask, 14:]
            inXs[:, scalable]  = scaler.transform( inXs [:, scalable])
            inX[dnn2Mask, 14:] = inXs
    
    if (output):
        checkFeatures(inXs[:,:], dataPathFolder, name="SscaledPlot", featureNames = getFeatureNames()[14:])
    

    
    return inX, weights, lkrM, krM

'''def checkBkg(path):
    print ("Checking if bkg npyData already exist in"+path+"/inX.npy")
    if not os.path.exists(path+ "/inX.npy"):
        print("Bkg not found in "+path+ "/inX.npy")
        return False
    else:
        print("\nBkg found in the directory :" + path)
        return True'''