import glob
from getVariations import getVariations, getVarNames
from getListOfWeights import getListOfWeights
from computeHisto import computeHisto
from phaseSpaceChange import phaseSpaceChange
import pandas as pd
import numpy as np
import sys

# Open root files
# From root files opened extract ratio bewteen phase space applying the systematics and in the nominal case
# From root files save in a df the weights of the different variations
# load inX file saved using createSignal and moved in the folder systematics
# Extract the multiplications between scale factors and variations
# Compute histogram and get the percentage of difference. That is the error.
def main(createDF, createSpaceFactors):
    rootPath  = '/nfs/dust/cms/user/celottog/CMSSW_10_6_32/src/TopAnalysis/Configuration/analysis/diLeptonic/miniTree_2018/Nominal/emu'
    fileNames = glob.glob(rootPath+'/emu_ttbarsignalplustau_*.root')		# List of files.root in the directory
    fileNames = fileNames[15:25]
    if (createDF):
        df = getVariations(fileNames, 'miniTree')
        df.to_pickle('/nfs/dust/cms/user/celottog/mttNN/systematics/df.pkl')
    if (createSpaceFactors):
        phaseSpaceChange(fileNames)
    print("Opening df...")
    df = pd.read_pickle('/nfs/dust/cms/user/celottog/mttNN/systematics/df.pkl')
    print("Opened df...")
    #print(df.keys())
    #print(df['var_pdf'])
    
    inX = np.load("/nfs/dust/cms/user/celottog/mttNN/systematics/inX.npy")
    assert len(df)==len(inX), "\nLength of inX = %d\nLength of df  = %d" %(len(inX), len(df))
    df = df[inX[:,0]>-998]
    inX = inX[inX[:,0]>-998, :]
    

    listOfWeights = getListOfWeights(df)
    computeHisto(inX, listOfWeights)
    




    return
if __name__ == "__main__":
    print(sys.argv )
    if len(sys.argv) > 1:
        createDF = bool(int(sys.argv[1]))
        createSpaceFactors = bool(int(sys.argv[2]))
    else:
        createDF = False
        createSpaceFactors = False
    main(createDF, createSpaceFactors)

