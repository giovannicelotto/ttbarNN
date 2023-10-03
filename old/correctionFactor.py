import glob
import ROOT
import numpy as np
from utils.plot import GetRecoBin
import time
from progress.bar import IncrementalBar

def main():
    start_time = time.time()
    recoBin = GetRecoBin()
    fileNames = [   "/nfs/dust/cms/user/celottog/ttbarSignalFromDilepton/ee_ttbarsignalplustau_fromDilepton_16.root",
                   "/nfs/dust/cms/user/celottog/ttbarSignalFromDilepton/mumu_ttbarsignalplustau_fromDilepton_17.root",
                  "/nfs/dust/cms/user/celottog/ttbarSignalFromDilepton/emu_ttbarsignalplustau_fromDilepton_18.root",
                    "/nfs/dust/cms/user/celottog/ttbarSignalFromDilepton/ee_ttbarsignalplustau_fromDilepton_0.root",
                    "/nfs/dust/cms/user/celottog/ttbarSignalFromDilepton/emu_ttbarsignalplustau_fromDilepton_0.root",
                    "/nfs/dust/cms/user/celottog/ttbarSignalFromDilepton/mumu_ttbarsignalplustau_fromDilepton_0.root",
                    "/nfs/dust/cms/user/celottog/ttbarSignalFromDilepton/ee_ttbarsignalplustau_fromDilepton_1.root",
                    "/nfs/dust/cms/user/celottog/ttbarSignalFromDilepton/emu_ttbarsignalplustau_fromDilepton_1.root",
                    "/nfs/dust/cms/user/celottog/ttbarSignalFromDilepton/mumu_ttbarsignalplustau_fromDilepton_1.root",
                    "/nfs/dust/cms/user/celottog/ttbarSignalFromDilepton/ee_ttbarsignalplustau_fromDilepton_2.root",
                    "/nfs/dust/cms/user/celottog/ttbarSignalFromDilepton/emu_ttbarsignalplustau_fromDilepton_2.root",
                    "/nfs/dust/cms/user/celottog/ttbarSignalFromDilepton/mumu_ttbarsignalplustau_fromDilepton_2.root",
                    ]
    #path = "/nfs/dust/cms/user/celottog/ttbarSignalFromDilepton/"
    #fileNames = glob.glob(path+'/*.root')
    #maxEvents = 100000
    ttbarMCut = []
    ttbarMNoCut = []
    top			= ROOT.TLorentzVector(0.,0.,0.,0.)
    antitop		= ROOT.TLorentzVector(0.,0.,0.,0.)
    ttbar		= ROOT.TLorentzVector(0.,0.,0.,0.)
    for fileName in fileNames:
        print(fileNames.index(fileName)+1,"/",len(fileNames))
        f = ROOT.TFile.Open(fileName)
    
        treeName = 'miniTree'
        tree = f.Get(treeName)

        gen_top_pt =  np.array([0], dtype='f')
        tree.SetBranchAddress("gen_top_pt", gen_top_pt)
        gen_top_eta =  np.array([0], dtype='f')
        tree.SetBranchAddress("gen_top_eta", gen_top_eta)
        gen_top_phi =  np.array([0], dtype='f')
        tree.SetBranchAddress("gen_top_phi", gen_top_phi)
        gen_top_m =  np.array([0], dtype='f')
        tree.SetBranchAddress("gen_top_m", gen_top_m)
        gen_antitop_pt =  np.array([0], dtype='f')
        tree.SetBranchAddress("gen_antitop_pt", gen_antitop_pt)
        gen_antitop_eta =  np.array([0], dtype='f')
        tree.SetBranchAddress("gen_antitop_eta", gen_antitop_eta)
        gen_antitop_phi =  np.array([0], dtype='f')
        tree.SetBranchAddress("gen_antitop_phi", gen_antitop_phi)
        gen_antitop_m =  np.array([0], dtype='f')
        tree.SetBranchAddress("gen_antitop_m", gen_antitop_m)

         

        bigNumber = 20000
        maxForBar = int(tree.GetEntries()/bigNumber)
        bar = IncrementalBar('Processing', max=maxForBar, suffix='%(percent).1f%% - %(eta)ds')
        for i in range(tree.GetEntries()):
            if(i%bigNumber==0):
                bar.next()


            tree.GetEntry(i)
            top.SetPtEtaPhiM(gen_top_pt[0],gen_top_eta[0],gen_top_phi[0],gen_top_m[0])
            antitop.SetPtEtaPhiM(gen_antitop_pt[0],gen_antitop_eta[0],gen_antitop_phi[0],gen_antitop_m[0])
            ttbar = top+antitop
            ttbarMCut.append(ttbar.M())
        
        #print(treeName)
        #print("Entries      :", tree.GetEntries())
        #print("All ttbar gen:", len(ttbarM))
        #print(recoCounts)






        treeName = 'miniTree_allGen'
        tree = f.Get(treeName)


        gen_top_pt =  np.array([0], dtype='f')
        tree.SetBranchAddress("gen_top_pt", gen_top_pt)
        gen_top_eta =  np.array([0], dtype='f')
        tree.SetBranchAddress("gen_top_eta", gen_top_eta)
        gen_top_phi =  np.array([0], dtype='f')
        tree.SetBranchAddress("gen_top_phi", gen_top_phi)
        gen_top_m =  np.array([0], dtype='f')
        tree.SetBranchAddress("gen_top_m", gen_top_m)
        gen_antitop_pt =  np.array([0], dtype='f')
        tree.SetBranchAddress("gen_antitop_pt", gen_antitop_pt)
        gen_antitop_eta =  np.array([0], dtype='f')
        tree.SetBranchAddress("gen_antitop_eta", gen_antitop_eta)
        gen_antitop_phi =  np.array([0], dtype='f')
        tree.SetBranchAddress("gen_antitop_phi", gen_antitop_phi)
        gen_antitop_m =  np.array([0], dtype='f')
        tree.SetBranchAddress("gen_antitop_m", gen_antitop_m)


        bigNumber = 20000
        maxForBar = int(tree.GetEntries()/bigNumber)
        bar = IncrementalBar('Processing', max=maxForBar, suffix='%(percent).1f%% - %(eta)ds')
        for i in range(tree.GetEntries()):
            if(i%bigNumber==0):
                bar.next()


            tree.GetEntry(i)
            top.SetPtEtaPhiM(gen_top_pt[0],gen_top_eta[0],gen_top_phi[0],gen_top_m[0])
            antitop.SetPtEtaPhiM(gen_antitop_pt[0],gen_antitop_eta[0],gen_antitop_phi[0],gen_antitop_m[0])
            ttbar = top+antitop
            ttbarMNoCut.append(ttbar.M())

        
        #print(treeName)
        #print("Entries      :", tree.GetEntries())
        #print("All ttbar gen:", len(ttbarM))
        
        #print(genCounts)
        
    ttbarMCut = np.array(ttbarMCut)
    ttbarMNoCut = np.array(ttbarMNoCut)     
    recoCounts, edges = np.histogram(ttbarMCut, bins=recoBin)
    genCounts, edges = np.histogram(ttbarMNoCut, bins=recoBin)
    print(recoCounts.sum())   
    print("\n\nRatio\n", recoCounts/genCounts)


    end_time = time.time()
    total_time = end_time - start_time
    print(f"Execution time: {total_time} seconds")
if __name__ == "__main__":
	main()
