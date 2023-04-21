import os
import numpy as np
from utils.helpers4P import *
from utils.stat import *    # for scaling
from stat import *
from npyData.checkFeatures import checkFeatures
from sklearn.utils import shuffle
from utils.helpers import getFeatureNames
import pickle

def loadData4P(dataPathFolder, minbjets, scalerPath, output=True, nFiles = None, signal=True):
    print("loadData called")
    '''
    Load data from saved numpy arrays or create them if not available (using loadRegressionData4P)
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
        inX, weights, lkrM, krM = loadRegressionData4P(path = dataPathFolder+"/..", treeName = "miniTree", minbjets=minbjets, nFiles = nFiles, signal = signal)
        inX     = np.array(inX)
        weights = np.array(weights)
        lkrM    = np.array(lkrM)
        krM     = np.array(krM)
# inX loose and full vanno maskati. hanno tutti la stessa luinghezza=event passano tagli comuni. se un evento non passa un taglio particolare in quell'array e meso a -999        
        assert len(inX)==len(lkrM)==len(krM)==len(weights), "Check lengths -- data4P"
        inX, weights, lkrM, krM = shuffle(inX, weights, lkrM, krM, random_state = 1999)
        print(inX)
        m = (inX[:,0]>-998)
        print("Number events passing DNN requiremenet:", len(inX[m,0]))
        print("Number events passing Loose requiremenet:", len(lkrM[lkrM>-998]))
        print("Number events passing Full requiremenet:", len(krM[krM>-998]))
        if not os.path.exists(dataPathFolder):
            os.makedirs(dataPathFolder)
        np.save(dataPathFolder+ "/inX.npy", inX)
        np.save(dataPathFolder+ "/weights.npy", weights)
        np.save(dataPathFolder+ "/lkrM.npy", lkrM)
        np.save(dataPathFolder+ "/krM.npy", krM)
        print("Control plots")
        checkFeatures(inX[m,:], dataPathFolder)

# Load data and scale them
    inX     = np.load(dataPathFolder+ "/inX.npy")
    weights = np.load(dataPathFolder+ "/weights.npy")
    lkrM    = np.load(dataPathFolder+ "/lkrM.npy")
    krM     = np.load(dataPathFolder+ "/krM.npy")
    m = (inX[:,0]>-998)

    
    print("Loading the scalers...")
    with open(scalerPath, 'rb') as file:
        scalers = pickle.load(file)
        maxer = scalers['maxer']
        powerer = scalers['powerer']
        scaler = scalers['scaler']
        maxable = scalers['maxable']
        powerable = scalers['powerable']
        scalable = scalers['scalable']
        #boxable = scalers['boxable']
        #keepable = scalers['keepable']

    print("Scaling data")
    inXs = inX[m, :]

    inXs[:, maxable]   = maxer.transform( inXs  [:, maxable])
    inXs[:, powerable] = powerer.transform( inXs[:, powerable])
    inXs[:, scalable]  = scaler.transform( inXs [:, scalable])
    inX[m, :] = inXs

    
    if (output):
        checkFeatures(inXs, dataPathFolder, name="scaledPlot")
    inX[inX[:,0]>-998,:] = inXs

    print('\n\nShapes of all the data passing criteria Step 3-7. Each array has values -998 where loose, full, both do not have solutions:\nInput  \t',inX.shape,'\nWeights\t', weights.shape,'\nLoose M\t', lkrM.shape,'\nFull M\t', krM.shape)
    
    return inX, weights, lkrM, krM

def checkBkg(path):
    print ("Checking if bkg npyData already exist in"+path+"/inX.npy")
    if not os.path.exists(path+ "/inX.npy"):
        print("Bkg not found in "+path+ "/inX.npy")
        return False
    else:
        print("\nBkg found in the directory :" + path)
        return True