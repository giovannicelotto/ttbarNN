import os
import matplotlib.pyplot as plt
import matplotlib
from utils.data4P import loadData4P, checkBkg
import glob
from tensorflow import keras
from utils.plot import covarianceMatrix, diffCrossSec, invariantMass, getRecoBin
import numpy as np


matplotlib.use('agg')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'        # TensorFlow will only display error messages and suppress all other messages including these warnings.

   
def main():
    print("********************************************\n*					   *\n*        Main function started             *\n*					   *\n********************************************")
    nFiles_             = None
    maxEvents_          = None
    dataPathFolder      = "/nfs/dust/cms/user/celottog/realData/signal/npyData"                         # folder where load signal events
    outFolder           = "/nfs/dust/cms/user/celottog/realData/outputs"                                # output
    modelDir            = "/nfs/dust/cms/user/celottog/mttNN/outputs/3*None_2*[16_32]1904/model"        # dir of the model trained
    scalerPath          = "/nfs/dust/cms/user/celottog/mttNN/npyData/3*None/scalers.pkl"                # dir of the scalers
    bkgPath             = "/nfs/dust/cms/user/celottog/realData/bkg/npyData"                            # folder of bkg file

    if not os.path.exists(outFolder+ "/model"):
        os.makedirs(outFolder+ "/model")
        print("Created folder " + outFolder+"/model")

    # Load the model
    print("Loading model...")
    model = keras.models.load_model(modelDir+"/mttRegModel.h5")


    # Create npyData and scale them using the scalers stored in scalerPath
    inX, weights, lkrM, krM =  loadData4P(dataPathFolder = dataPathFolder,   minbjets = 1, output=True, scalerPath=scalerPath, nFiles = nFiles_)
    m = inX[:,0]>-998
    # Predict values
    print("Predicting values ...")
    yPredicted = model.predict(inX[m,:])
    bins = getRecoBin()
    signalCounts = np.histogram(yPredicted, bins = bins)
    print("Signal counts before normalization and bkg subtraction", signalCounts[0])
    
# load bkg
    bkgList = ['wwtoall', 'wztoall', 'zztoall']
    bkg = {}
    lumi = 59.7/1000.
    xsections = {
        'wwtoall': 118.7,
        'wztoall': 47.13,
        'zztoall': 16.523
    }
    bkgNorm = {}

    for process in range(len(bkgList)):
        print("Opening ", bkgList[process])
        bkgInX      = np.load(bkgPath+'/'+bkgList[process]+'_inX.npy')
        bkgWeights  = np.load(bkgPath+'/'+bkgList[process]+'_weights.npy')
        bkgPredicted = model.predict(bkgInX[bkgInX[:,0]>-998,:])
        bkgCounts   = np.histogram(bkgPredicted, bins = bins, weights = bkgWeights[m])
        print(bkgCounts[0])
        bkgNormTemp     = np.load(bkgPath + '/' + bkgList[process] + '_norm.npy')
        bkgCounts = bkgCounts * lumi * xsections[process]/bkgNormTemp
        print(bkgCounts[0])
        bkgNorm[process] = bkgNormTemp



    #bkgInX, bkgWeightsb, bkgLkrM, bkgKrM =  loadData4P(dataPathFolder = bkgPath,   minbjets = 1, output=True, scalerPath=scalerPath, nFiles = None)
    #mb = bkgInX[:,0]>-998
    #bkgPredicted = model.predict(bkgInX[mb,:])
    # scale factor
    #bkgCounts = np.histogram(bkgPredicted, bins = bins)
    #print("Bkg counts before normalization", bkgCounts)

# /nfs/dust/cms/user/celottog/realData/bkg/emu_ttbarZtoqq_0.root
    # subtract background
    # write a function that opens a bkg file
    #bkgX, bkgWeights, bkgLkrM, bkgKrM =  loadData4P(dataPathFolder = dataPathFolder,   minbjets = 1, scalerPath = scalerPath, output=True, scale=scale, nFiles = nFiles_)
    # compute get the loose, full and dnn mtt numpy get the entries and normalization
    # histogram the data and normalize the counts by the factor of luminosity
    # returns an array with entries for each bin to be subtracted to the signal (also the signal willbe normalized )
    # 
    #count luminosity of total signal

    # Load matrix and Unfold
    '''print("Loading response matrices...")
    dnnM    = np.load(modelDir+"/dnnM.npy")
    looseM  = np.load(modelDir+"/looseM.npy")
    kinM    = np.load(modelDir+"/kinM.npy")


    print("\nCondition Number of the DNN matrix:  "+str(np.linalg.cond(dnnM))+"\n")
    invariantMass(y_predicted, lkrM, krM, totGen=None, outFolder=outFolder)
    diffCrossSec(y_predicted, lkrM, krM, totGen=None, dnnMatrix=dnnM, kinMatrix=kinM, looseMatrix=looseM, outFolder=outFolder, write=False)'''
    
    # Produce plots
    

if __name__ == "__main__":
	main()