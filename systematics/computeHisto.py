import numpy as np
import sys
from getVariations import getVarNames
from getListOfWeights import getSystNames
folder_path = '/nfs/dust/cms/user/celottog/realData'
sys.path.append(folder_path)
folder_path = '/nfs/dust/cms/user/celottog/mttNN'
sys.path.append(folder_path)
from utils.helpers import getFeatureNamesOriginalNumbered
from plotInput import getRange
import pickle

def computeHisto(inX, listOfWeights):
    x1, x2, nbin = getRange()
    featureNames = getFeatureNamesOriginalNumbered()
    systPerCent = {}

    with open("/nfs/dust/cms/user/celottog/mttNN/systematics/dictionaryPhaseFactors.pkl", "rb") as file:
        spaceFactors = pickle.load(file)

    for featureN in range(0,73):
            print(featureNames[featureN],"...\n")
            uf = inX[:,featureN] < x1[featureN]
            of = inX[:,featureN] > x2[featureN]
            inX[uf,featureN] = x1[featureN]+0.0001
            inX[of,featureN] = x2[featureN]-0.0001
            countsNoVar = np.histogram(inX[:,featureN], weights=listOfWeights[0], bins=nbin[featureN], range=(x1[featureN], x2[featureN]))[0]
            countsNoVar = countsNoVar * spaceFactors['NOMINAL_NORENORMALIZATION']/spaceFactors['NOMINAL']
            #print("Counts no Var:", countsNoVar, "\n\n")
            systPerFeature = np.zeros(nbin[featureN])
            assert len(systPerFeature)==len(countsNoVar), "Feature nbin do not match"
            for var in range(0,len(listOfWeights)):
                #print(getSystNames()[var])
                countsVar = np.histogram(inX[:,featureN], weights=listOfWeights[var], bins=nbin[featureN], range=(x1[featureN], x2[featureN]))[0]
                for key in spaceFactors:
                    #print(key) 
                    if key.lower() in getSystNames()[var]:
                        #print("Found", key)
                        countsVar = countsVar*spaceFactors['NOMINAL_NORENORMALIZATION']/spaceFactors[key]

                print(getSystNames()[var], (countsNoVar-countsVar)/countsNoVar, "\n\n")
                systPerFeature = np.sqrt(systPerFeature**2+((countsNoVar-countsVar)/countsNoVar)**2)
                #print(getSystNames()[var], (countsNoVar-countsVar)/countsNoVar)
            print(systPerFeature)
            systPerCent[featureNames[featureN]] = systPerFeature
    print(systPerCent)
    with open("/nfs/dust/cms/user/celottog/mttNN/systematics/dictionarySystematics.pickle", "wb") as file:
        pickle.dump(systPerCent, file)

    return

if __name__ == "__main__":
    computeHisto()