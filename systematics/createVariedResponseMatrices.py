import sys
sys.path.append('/nfs/dust/cms/user/celottog/mttNN')
from utils.plot import getRecoBin
from utils.stat import scalerMC
import numpy as np
from tensorflow import keras
import pandas as pd
from getListOfWeights import getListOfWeights, getSystNames
import os


def getResponse(y_predicted, totGen, weights):
    recoBin = getRecoBin()
    nRecoBin = len(recoBin)-1
    matrix = np.empty((nRecoBin, nRecoBin))
    for i in range(nRecoBin):
        maskGen = (totGen >= recoBin[i]) & (totGen < recoBin[i+1])
       
        normalization = np.sum(weights[maskGen])
        for j in range(nRecoBin):    
            maskRecoGen = (y_predicted >= recoBin[j]) & (y_predicted < recoBin[j+1]) & maskGen
            entries = np.sum(weights[maskRecoGen])
            if normalization == 0:
                matrix[j, i] = 0
            else:
                matrix[j, i] = entries/normalization
    return matrix


def createVariedResponseMatrices(computePrediction):
    modelDir = "/nfs/dust/cms/user/celottog/mttNN/outputs/15*None_[32_64_48]DoubleNN_W-2_2100_fineBin/model"
    inputDir = "/nfs/dust/cms/user/celottog/mttNN/systematics"
    outDir   = "/nfs/dust/cms/user/celottog/mttNN/systematics/responseMatrices"
    modelName = "mttRegModel"
    inX = np.load(inputDir+"/inX.npy")[:, :]
    inX    = scalerMC(modelDir = modelDir, MCInX = inX )
    totGen = np.load("/nfs/dust/cms/user/celottog/realData/signal/MC_totGen.npy")[:]

    assert len(inX)==len(totGen), "Length are not matching %d vs %d" %(len(inX), len(totGen))

    model       = keras.models.load_model(modelDir+"/"+modelName+".h5")
    simpleModel = keras.models.load_model(modelDir+"/"+modelName+"_simple.h5")
    y_predicted = np.ones(len(inX))*-999
    dnnMask  = inX[:,0]>-998
    dnn2Mask = inX[:,0]<-4998
    if not os.path.exists(outDir+ "/y_predicted.npy"):
        computePrediction=True
    if computePrediction:
        y_predicted[dnnMask] = model.predict(inX[dnnMask, :])[:,0]
        y_predicted[dnn2Mask] = simpleModel.predict(inX[dnn2Mask, 15:])[:,0]
        np.save(outDir+"/y_predicted.npy", y_predicted)
        


    print("y_predicted", y_predicted)


    #for variations
    df = pd.read_pickle('/nfs/dust/cms/user/celottog/mttNN/systematics/df.pkl')
    listOfWeights = getListOfWeights(df)
    listOfSyst    = getSystNames()

    for id in range(len(listOfWeights)):
        w = listOfWeights[id]
        systName = listOfSyst[id]
        m = getResponse(y_predicted, totGen, w)
        np.save(outDir+"/"+str(systName)+".npy", m)

  


    return

if __name__ =='__main__':
    if len(sys.argv) > 1:
        computPrediction = bool(int(sys.argv[1]))
    else:
        computPrediction = False
    createVariedResponseMatrices(computPrediction)