import numpy as np
import sys
from getVariations import getVarNames
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
    #for key in spaceFactors:
    #    print(key)
    
    for featureN in range(0,len(featureNames)):
            print(featureNames[featureN],"...\n")
            uf = (inX[:,featureN] < x1[featureN]) & (inX[:,featureN] > -998)
            of = inX[:,featureN] > x2[featureN]
            inX[uf,featureN] = x1[featureN]+0.0001
            inX[of,featureN] = x2[featureN]-0.0001
            countsNoVar = np.histogram(inX[:,featureN], weights=listOfWeights['weights'], bins=nbin[featureN], range=(x1[featureN], x2[featureN]))[0]
            countsNoVar = countsNoVar * spaceFactors['NOMINAL_NORENORMALIZATION']/spaceFactors['NOMINAL']
            
            
            systPerFeature = np.zeros(nbin[featureN])
            assert len(systPerFeature)==len(countsNoVar), "Feature nbin do not match"
            for var in listOfWeights.keys():
                countsVar = np.histogram(inX[:,featureN], weights=listOfWeights[var], bins=nbin[featureN], range=(x1[featureN], x2[featureN]))[0]
                # Keep cross section fixed
                flag = False
                for key in spaceFactors:
                    if key.lower() == var.lower()[4:]:
                        print("Found %s"%var)
                        countsVar = countsVar*spaceFactors['NOMINAL_NORENORMALIZATION']/spaceFactors[key]
                        flag=True
                #if var == 'weights':
                #    countsVar = countsVar*spaceFactors['NOMINAL_NORENORMALIZATION']/spaceFactors['NOMINAL']
                #    assert not flag
                #    flag=True
                if flag == False:
                    print("Not Found")
                    countsVar = countsVar * spaceFactors['NOMINAL_NORENORMALIZATION']/spaceFactors['NOMINAL']     
                systPerFeature = np.sqrt(systPerFeature**2+((countsNoVar-countsVar)/countsNoVar)**2)
                if len(countsNoVar)<4:
                    print(var, (countsNoVar-countsVar)/countsNoVar)
            #print(systPerFeature)
            systPerCent[featureNames[featureN]] = systPerFeature
    print(systPerCent)
    with open("/nfs/dust/cms/user/celottog/mttNN/systematics/dictionarySystematics.pickle", "wb") as file:
        pickle.dump(systPerCent, file)

    return

if __name__ == "__main__":
    computeHisto()