# use createbkg to create inX file from nonused root files
# will also create a norm file which you need

import os
import matplotlib.pyplot as plt
import matplotlib
from utils.data4P import loadData4P, checkBkg
import glob
from tensorflow import keras
from utils.plot import covarianceMatrix, diffCrossSec, invariantMass, getRecoBin, getDiffBin
import numpy as np
import pickle
import time
import tensorflow as tf
import logging
import sys
folder_path = '/nfs/dust/cms/user/celottog/realData'
sys.path.append(folder_path)
from utilsForEval import *
logging.getLogger("tensorflow").setLevel(logging.WARNING)

matplotlib.use('agg')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'        # TensorFlow will only display error messages and suppress all other messages including these warnings.


   
def main():
    print("********************************************\n*					   *\n*        Main function started             *\n*					   *\n********************************************")
    
    nFiles_             = None
    signalPath          = "/nfs/dust/cms/user/celottog/realData/data/npyData"                         # folder where load signal events
    bkgPath             = "/nfs/dust/cms/user/celottog/realData/bkg/npyData"                            # folder of bkg file
    outFolder           = "/nfs/dust/cms/user/celottog/realData/outputs"                                # output
    modelDir            = "/nfs/dust/cms/user/celottog/mttNN/outputs/10*None_3*[46_38_111]2404afterBayes/model"        # dir of the model trained
    scalerPath          = "/nfs/dust/cms/user/celottog/mttNN/npyData/10*None/scalers.pkl"                # dir of the scalers
    MCPath              = "/nfs/dust/cms/user/celottog/realData/signal"
    if not os.path.exists(outFolder+ "/counts"):
        os.makedirs(outFolder+ "/counts")
        print("Created folder " + outFolder+"/counts")
    if (True):
        # Load the model
        print("Loading model...")
        
        model = keras.models.load_model(modelDir+"/mttRegModel.h5")
        print("model loaded...")
        print("Model summary\n", model.summary())
        
        # Create real npyData and scale them using the scalers stored in scalerPath
        inX, weights, lkrM, krM =  loadData4P(dataPathFolder = signalPath,   minbjets = 1, output=True, scalerPath=scalerPath, nFiles = nFiles_)
        m = inX[:,0]>-998
        # Predict values
        print("Predicting values ...")
        
        yPredicted = model.predict(inX[m,:])
        bins = getDiffBin()
        dataCounts = np.histogram(yPredicted[:,0], bins = bins, weights=weights[m])[0]
        print("Signal counts before normalization and bkg subtraction", dataCounts)
        
        

        lumi = 59.7*1000.
        xsections = {
        #diboson
            'wwtoall':              118.7,
            'wztoall':              47.13,
            'zztoall':              16.523,
        #other channels
            'ttbarbg_fromDilepton': 831.76*0.10706,
            'ttbarbg_fromLjets':    831.76*0.44113,
            'ttbarbg_fromHadronic': 831.76*0.45441,
        #ttbar+boson
            'ttbarWjetstoqq':       0.4062,
            'ttbarWjetstolnu':      0.2043,
            'ttbarZtollnunu':       0.2529,
            'ttbarZtoqq':           0.5297,
        # single TOP
            'singletop_s_leptonic': 10.32,
            'singletop_t':          136.02,
            'singletop_tw':         35.85*0.54559,
            'singleantitop_t':      80.95,
            'singleantitop_tw':     35.85*0.54559,
        # Wjets and Zjets
            'wtolnu':               61526.7,
            'dy1050':               22635.1,
            'dy50inf':              6225.4
        }
        bkgList = xsections.keys()
        bkgNorm = {}
        bkgErrs = {}
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


        for process in bkgList:
            print("Opening ", process)
            
            bkgInX      = np.load(bkgPath+'/'+process+'_inX.npy')
            bkgWeights  = np.load(bkgPath+'/'+process+'_weights.npy')
            
            print("Scaling data")
            mbkg = (bkgInX[:,0]>-998)
            bkgInXscaled = bkgInX[mbkg, :]

            bkgInXscaled[:, maxable]   = maxer.transform( bkgInXscaled  [:, maxable])
            bkgInXscaled[:, powerable] = powerer.transform( bkgInXscaled[:, powerable])
            bkgInXscaled[:, scalable]  = scaler.transform( bkgInXscaled [:, scalable])
            bkgInX[mbkg, :] = bkgInXscaled
            
            bkgPredicted = model.predict(bkgInX[mbkg,:])[:,0]
            if (bkgPredicted.shape != bkgWeights[bkgWeights>-998].shape):
                print("shapes of bkg predicted and weights are not equal.\nWeights:\n")
                print(bkgWeights)
                print("\ninX\n", bkgInX)
            bkgCounts   = np.histogram(bkgPredicted, bins = bins, weights = bkgWeights[bkgWeights>-998])[0]
            print(bkgCounts)
            bkgNormTemp     = np.load(bkgPath + '/' + process + '_norm.npy')
            bkgCounts = bkgCounts * lumi * xsections[process]/bkgNormTemp
            bkgErr = np.sqrt(bkgCounts) * lumi * xsections[process]/bkgNormTemp
            print("Efficiency:\t", bkgCounts.sum()/bkgNormTemp)
            bkgNorm[process] = bkgCounts
            bkgErrs[process] = bkgErr
        print(bkgNorm)


    #  *************************       SIGNAL FROM MONTE CARLO        ****************************************
        print("Opening ttbar")
        MCInX      = np.load(MCPath+"/MC_inX.npy")
        MCWeights  = np.load(MCPath+'/MC_weights.npy')
        

        print("Scaling data")
        mMC = (MCInX[:,0]>-998)
        MCInXscaled = MCInX[mMC, :]

        MCInXscaled[:, maxable]   = maxer.transform( MCInXscaled  [:, maxable])
        MCInXscaled[:, powerable] = powerer.transform( MCInXscaled[:, powerable])
        MCInXscaled[:, scalable]  = scaler.transform( MCInXscaled [:, scalable])
        MCInX[mMC, :] = MCInXscaled
        
        MCPredicted = model.predict(MCInX[mMC,:])[:,0]
        if (MCPredicted.shape != MCWeights[MCWeights>-998].shape):
            print("shapes of MC predicted and weights are not equal.\nWeights:\n")
            print(MCWeights)
            print("\ninX\n", MCInX)
        MCCounts   = np.histogram(MCPredicted, bins = bins, weights = MCWeights[MCWeights>-998])[0]
        print(MCCounts)
        MCNormTemp     = np.load(MCPath + '/MC_norm.npy')

        MCCounts = MCCounts * lumi * xsections['ttbarbg_fromDilepton']/MCNormTemp
        MCErr = np.sqrt(MCCounts) * lumi * xsections['ttbarbg_fromDilepton']/MCNormTemp
        print("Efficiency:\t", MCCounts.sum()/MCNormTemp)

        np.save("/nfs/dust/cms/user/celottog/realData/outputs/counts/bkgNorm.npy", bkgNorm)
        np.save("/nfs/dust/cms/user/celottog/realData/outputs/counts/MCCounts.npy", MCCounts)
        np.save("/nfs/dust/cms/user/celottog/realData/outputs/counts/dataCounts.npy", dataCounts)
    
    bkgNorm     = np.load("/nfs/dust/cms/user/celottog/realData/outputs/counts/bkgNorm.npy", allow_pickle=True).item()
    MCCounts    = np.load("/nfs/dust/cms/user/celottog/realData/outputs/counts/MCCounts.npy")        
    dataCounts  = np.load("/nfs/dust/cms/user/celottog/realData/outputs/counts/dataCounts.npy")    

    bins = getDiffBin()
    overlapDistribution(bkgNorm, bins, MCCounts, dataCounts, outFolder)
    


#   Compute subtraction of bkg and propagate errors
    dataCountsSubtracted = dataCounts
    errCountsSubtracted = np.sqrt(dataCounts)
    for i in xsections.keys():
        dataCountsSubtracted = dataCountsSubtracted - bkgNorm[i]
        errCountsSubtracted  = np.sqrt(errCountsSubtracted**2 + bkgErrs[i]**2)
    
    errCountsSubtracted = np.diag(errCountsSubtracted**2)
    
    plotUnfolded(dataCountsSubtracted, errCountsSubtracted, bins, modelDir, outFolder)

    '''looseM  = np.load(modelDir+"/looseM.npy")
    kinM    = np.load(modelDir+"/kinM.npy")


    print("\nCondition Number of the DNN matrix:  "+str(np.linalg.cond(dnnM))+"\n")
    invariantMass(y_predicted, lkrM, krM, totGen=None, outFolder=outFolder)
    diffCrossSec(y_predicted, lkrM, krM, totGen=None, dnnMatrix=dnnM, kinMatrix=kinM, looseMatrix=looseM, outFolder=outFolder, write=False)'''
    
    # Produce plots
    

if __name__ == "__main__":
	main()