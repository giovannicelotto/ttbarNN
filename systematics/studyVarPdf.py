import ROOT
import glob
import numpy as np
from progress.bar import IncrementalBar

def main(treeName):
    rootPath  = '/nfs/dust/cms/user/celottog/CMSSW_10_6_32/src/TopAnalysis/Configuration/analysis/diLeptonic/miniTree_2018/Nominal/emu'
    fileNames = glob.glob(rootPath+'/emu_ttbarsignalplustau*.root')		# List of files.root in the directory
    fileNames = fileNames[18:19]

    for fileName in fileNames:
        f = ROOT.TFile.Open(fileName)
        tree = f.Get(treeName)
        print(tree.GetEntries())
        print("Read TTree: {} (Entries: {})".format(treeName, tree.GetEntries()))
# Setting branches in the tree
        weight = np.array([0], dtype='d')
        tree.SetBranchAddress("weight", weight)
        var_PDF = np.array([0]*103, dtype='f')
        tree.SetBranchAddress("var_PDF", var_PDF)
        maxEntries = tree.GetEntries() 
        bigNumber = 20000
        maxForBar = int(maxEntries/bigNumber)

        bar = IncrementalBar('Processing', max=maxForBar, suffix='%(percent).1f%% - %(eta)ds')
        print("Start loop")
        for i in range(0, tree.GetEntries()):

            tree.GetEntry(i)

            print("i = ",i, var_PDF)
            tree.SetBranchStatus("*",1)
        

    return


if __name__ == "__main__":
    treeName = 'miniTree'
    main(treeName)




