import glob
import sys
sys.path.append('/nfs/dust/cms/user/celottog/mttNN/utils')
import ROOT
import numpy as np
from progress.bar import IncrementalBar
import math
from helpers import rotation, append4vector



	

def createRealData(path, treeName, minbjets):
	'''
	Open trees
	Set Branches to variables
	Define Lorentz Vector and compute observables of interest from measured ones
	Returns a list of lists
	'''
# List that will be converted to numpy array to be used in the NN
	eventIn=[]		# Input
	kinRecoOut=[]	# kinRecoM (to be compared with NN output)
	lKinRecoOut=[]	# loosekinRecoM (to be compared with NN output)
	weights=[]		# weights to be used in the NN


	f = ROOT.TFile.Open(path)
	tree = f.Get(treeName)

	channelID = None
	isEMuChannel = False
	if ("emu_") in path:
		isEMuChannel = True
		channelID = 0
	elif ("mumu_") in path:
		channelID = -1
	elif ("ee_") in path:
		channelID = 1
	assert channelID == 0,"Not using emu files!"

	print("Read TTree: {} (Entries: {})".format(treeName, tree.GetEntries()))
# Setting branches in the tree
	if(True):
		tree.SetBranchStatus("var_*",0)             # Ignoring all the var variables in the tree

		passStep3 = np.array([0], dtype=bool)
		tree.SetBranchAddress("passStep3", passStep3)

		hasKinRecoSolution = np.array([0], dtype=bool)
		tree.SetBranchAddress("hasKinRecoSolution", hasKinRecoSolution)

		hasLooseKinRecoSolution = np.array([0], dtype=bool)
		tree.SetBranchAddress("hasLooseKinRecoSolution", hasLooseKinRecoSolution)

		njets = np.array([0]*20, dtype='uint')
		tree.SetBranchAddress("n_jets", njets)
		jetPt = np.array([0]*20, dtype='f')
		tree.SetBranchAddress("jets_pt", jetPt)
		jetEta = np.array([0]*20, dtype='f')
		tree.SetBranchAddress("jets_eta", jetEta)
		jetPhi = np.array([0]*20, dtype='f')
		tree.SetBranchAddress("jets_phi", jetPhi)
		jetM = np.array([0]*20, dtype='f')
		tree.SetBranchAddress("jets_m", jetM)
		weight = np.array([0], dtype='d')
		tree.SetBranchAddress("weight", weight)
		leptonSF = np.array([0], dtype='f')
		tree.SetBranchAddress("leptonSF", leptonSF)
		btagSF = np.array([0], dtype='f')
		tree.SetBranchAddress("btagSF", btagSF)
		pileupSF = np.array([0], dtype='f')
		tree.SetBranchAddress("pileupSF", pileupSF)
		prefiringWeight = np.array([0], dtype='f')
		tree.SetBranchAddress("l1PrefiringWeight", prefiringWeight)

	# Measured leptons pt, eta, phi, m
		lepton1_pt =  np.array([0], dtype='f')
		tree.SetBranchAddress("lepton1_pt", lepton1_pt)
		lepton1_eta =  np.array([0], dtype='f')
		tree.SetBranchAddress("lepton1_eta", lepton1_eta)
		lepton1_phi =  np.array([0], dtype='f')
		tree.SetBranchAddress("lepton1_phi", lepton1_phi)
		lepton1_m =  np.array([0], dtype='f')
		tree.SetBranchAddress("lepton1_m", lepton1_m)
		lepton2_pt =  np.array([0], dtype='f')
		tree.SetBranchAddress("lepton2_pt", lepton2_pt)
		lepton2_eta =  np.array([0], dtype='f')
		tree.SetBranchAddress("lepton2_eta", lepton2_eta)
		lepton2_phi =  np.array([0], dtype='f')
		tree.SetBranchAddress("lepton2_phi", lepton2_phi)
		lepton2_m =  np.array([0], dtype='f')
		tree.SetBranchAddress("lepton2_m", lepton2_m)
	# Measured met
		met_pt =  np.array([0], dtype='f')
		tree.SetBranchAddress("met_pt", met_pt)
		met_phi =  np.array([0], dtype='f')
		tree.SetBranchAddress("met_phi", met_phi)

	# kinReco top and antitop
		kinReco_top_pt =  np.array([0], dtype='f')
		tree.SetBranchAddress("kinReco_top_pt", kinReco_top_pt)
		kinReco_top_eta =  np.array([0], dtype='f')
		tree.SetBranchAddress("kinReco_top_eta", kinReco_top_eta)
		kinReco_top_phi =  np.array([0], dtype='f')
		tree.SetBranchAddress("kinReco_top_phi", kinReco_top_phi)
		kinReco_top_m =  np.array([0], dtype='f')
		tree.SetBranchAddress("kinReco_top_m", kinReco_top_m)

		kinReco_antitop_pt =  np.array([0], dtype='f')
		tree.SetBranchAddress("kinReco_antitop_pt", kinReco_antitop_pt)
		kinReco_antitop_eta =  np.array([0], dtype='f')
		tree.SetBranchAddress("kinReco_antitop_eta", kinReco_antitop_eta)
		kinReco_antitop_phi =  np.array([0], dtype='f')
		tree.SetBranchAddress("kinReco_antitop_phi", kinReco_antitop_phi)
		kinReco_antitop_m =  np.array([0], dtype='f')
		tree.SetBranchAddress("kinReco_antitop_m", kinReco_antitop_m)
	# Loose kinreco ttbar
		looseKinReco_ttbar_pt =  np.array([0], dtype='f')
		tree.SetBranchAddress("looseKinReco_ttbar_pt", looseKinReco_ttbar_pt)
		looseKinReco_ttbar_eta =  np.array([0], dtype='f')
		tree.SetBranchAddress("looseKinReco_ttbar_eta", looseKinReco_ttbar_eta)
		looseKinReco_ttbar_phi =  np.array([0], dtype='f')
		tree.SetBranchAddress("looseKinReco_ttbar_phi", looseKinReco_ttbar_phi)
		looseKinReco_ttbar_m =  np.array([0], dtype='f')
		tree.SetBranchAddress("looseKinReco_ttbar_m", looseKinReco_ttbar_m)
	# Jets
		jetBTagged = np.array([0]*20, dtype='f')
		tree.SetBranchAddress("jets_btag", (jetBTagged))
		jetBTagScore = np.array([0]*20, dtype='f')
		tree.SetBranchAddress("jets_btag_score", (jetBTagScore))
		jetTopMatched = np.array([0]*20, dtype='f')
		tree.SetBranchAddress("jets_topMatched", (jetTopMatched))
		jetAntiTopMatched = np.array([0]*20, dtype='f')
		tree.SetBranchAddress("jets_antitopMatched", (jetAntiTopMatched))


	maxEntries = tree.GetEntries() 

	bigNumber = 20000
	maxForBar = int(maxEntries/bigNumber)

# Tlorentz vectors

	lep1		= ROOT.TLorentzVector(0.,0.,0.,0.)
	lep2		= ROOT.TLorentzVector(0.,0.,0.,0.)
	met			= ROOT.TLorentzVector(0.,0.,0.,0.)
	kr_top		= ROOT.TLorentzVector(0.,0.,0.,0.)
	kr_antitop	= ROOT.TLorentzVector(0.,0.,0.,0.)
	lkr_ttbar	= ROOT.TLorentzVector(0.,0.,0.,0.)
	kr_nonbjet	= ROOT.TLorentzVector(0.,0.,0.,0.)
	lkr_nonbjet	= ROOT.TLorentzVector(0.,0.,0.,0.)
	kr_ttbar	= ROOT.TLorentzVector(0.,0.,0.,0.)
	bjettemp	= ROOT.TLorentzVector(0.,0.,0.,0.)
	dilepton	= ROOT.TLorentzVector(0.,0.,0.,0.)
	zero		= ROOT.TLorentzVector(0., 0., 0., 0.)
	extraJet 	= ROOT.TLorentzVector(0.,0.,0.,0.)


	bar = IncrementalBar('Processing', max=maxForBar, suffix='%(percent).1f%% - %(eta)ds')
	for i in range(maxEntries):
		lep1.SetPtEtaPhiM(0.,0.,0.,0.)
		lep2.SetPtEtaPhiM(0.,0.,0.,0.)
		met.SetPtEtaPhiM(0.,0.,0.,0.)
		kr_top.SetPtEtaPhiM(0.,0.,0.,0.)
		kr_antitop.SetPtEtaPhiM(0.,0.,0.,0.)
		lkr_ttbar.SetPtEtaPhiM(0.,0.,0.,0.)
		kr_nonbjet.SetPtEtaPhiM(0.,0.,0.,0.)
		kr_ttbar.SetPtEtaPhiM(0.,0.,0.,0.)
		lkr_nonbjet.SetPtEtaPhiM(0.,0.,0.,0.)
		bjettemp.SetPtEtaPhiM(0.,0.,0.,0.)
		dilepton.SetPtEtaPhiM(0.,0.,0.,0.)
		extraJet.SetPtEtaPhiM(0., 0., 0., 0.)
		ht = 0
		zero.SetPtEtaPhiM(0., 0., 0., 0.)
		
		evFeatures=[]
		tree.GetEntry(i)
		totWeight=1.

		if(i%bigNumber==0):
			bar.next()

		pass3= bool(passStep3[0])
		haskrs = bool(hasKinRecoSolution[0])
		haslkrs = bool(hasLooseKinRecoSolution[0])
		totWeight=weight[0]*btagSF[0]*leptonSF[0]*pileupSF[0]*prefiringWeight[0]
# Pass3: correct reconstruction of leptons
		if (pass3==True):
			lep1.SetPtEtaPhiM(lepton1_pt[0],lepton1_eta[0],lepton1_phi[0],lepton1_m[0])
			lep2.SetPtEtaPhiM(lepton2_pt[0],lepton2_eta[0],lepton2_phi[0],lepton2_m[0])	
		else:
			lep1.SetPtEtaPhiM(0,0,0,0)
			lep2.SetPtEtaPhiM(0,0,0,0)
		numJets = njets[0]
		dilepton=lep1+lep2

# Mll cut in the Z window
		passMLLCut = False
		if isEMuChannel:
			passMLLCut = True
		elif not (dilepton.M()>76. and dilepton.M()<106.):
			passMLLCut = True
		else:
			passMLLCut = False
		assert passMLLCut, "passMLLcut not passed! Not using only emu file"
# Cut in the MET (to suppress Zjets)
		passMETCut = False
		if isEMuChannel:
			passMETCut = True
		elif (met_pt[0]>40.):
			passMETCut = True
		else:
			passMETCut = False
		assert passMETCut, "passMET not passed! Not using only emu file"

# kinReco
		kr_top.SetPtEtaPhiM(kinReco_top_pt[0],kinReco_top_eta[0],kinReco_top_phi[0],kinReco_top_m[0])
		kr_antitop.SetPtEtaPhiM(kinReco_antitop_pt[0],kinReco_antitop_eta[0],kinReco_antitop_phi[0],kinReco_antitop_m[0])
		kr_ttbar= kr_antitop + kr_top
# LooseKinReco
		lkr_ttbar.SetPtEtaPhiM(looseKinReco_ttbar_pt[0],looseKinReco_ttbar_eta[0],looseKinReco_ttbar_phi[0],looseKinReco_ttbar_m[0])
		

		# conditions fro building an input events (training or testing)
		#if (pass3 & (numJets>=2) & passMETCut & (dilepton.M()>20.) & passMLLCut & (ttbar.M()>0) & (kr_ttbar.M()<1500) & (lkr_ttbar.M() < 1500)  ):# & (kr_ttbar.M()>1) & (lkr_ttbar.M() > 1)  & 
		if ( (numJets>=2) & (passMETCut) & (passMLLCut) & (pass3) & (weight[0]>0)):
			
			# jets identification
			jets = []
			bjets = []
			btag = []
			bscore =[]
			for idx in range(numJets):
				jet4=ROOT.TLorentzVector(0.,0.,0.,0.)
				jet4.SetPtEtaPhiM(jetPt[idx],jetEta[idx],jetPhi[idx],jetM[idx])
				jets.append(jet4)
				ht = ht+jet4.Pt()
				bTagged=int(bool(jetBTagged[idx]))
				btag.append(bTagged)
				bscore.append(jetBTagScore[idx])
				

			nonbjets = [jets[j] for j in range(len(btag)) if btag[j] == 0]
			bjets    = [jets[j] for j in range(len(btag)) if btag[j] == 1]
			if len(bjets) < minbjets:  # another cut
				continue
			else:
				pass

			if (haslkrs):
				lKinRecoOut.append(lkr_ttbar.M())
				assert (math.isclose(lkr_ttbar.M(), looseKinReco_ttbar_m[0])), "Loose mass value from the tree not the same from the root computation"
			else:
				lKinRecoOut.append(-999)

			if (haskrs):
				kinRecoOut.append(kr_ttbar.M())
			else:
				kinRecoOut.append(-999)
			
			if ((haslkrs) & (looseKinReco_ttbar_m[0] < 5000) & (haskrs) & (kr_ttbar.M()<5000)):
				pass
			else:
				weights.append(-999)
				eventIn.append([-999] * 72)
				continue
				

			nbjets = len(bjets)
			sortedJets = bjets + nonbjets
			met.SetPtEtaPhiM(met_pt[0],0.,met_phi[0],0.)
			rotation(sortedJets, lep1, lep2, met)	
			assert math.isclose(lep1.Phi(), 0, abs_tol = 1e-09), "Rotation didn't work"
			assert (lep1.Eta()>0), "Rotation didn't work"
			assert (lep2.Phi()>0), "Rotation didn't work"




			evFeatures.append(looseKinReco_ttbar_pt[0])
			evFeatures.append(looseKinReco_ttbar_eta[0])
			evFeatures.append(looseKinReco_ttbar_phi[0])
			evFeatures.append(looseKinReco_ttbar_m[0])
			append4vector(evFeatures, kr_ttbar)


			evFeatures.append(kinReco_top_pt[0])
			evFeatures.append(kinReco_top_eta[0])
			evFeatures.append(kinReco_top_phi[0])
			evFeatures.append(kinReco_antitop_pt[0])
			evFeatures.append(kinReco_antitop_eta[0])
			evFeatures.append(kinReco_antitop_phi[0])


			append4vector(evFeatures, dilepton)
			append4vector(evFeatures, dilepton+sortedJets[0]+sortedJets[1])
			append4vector(evFeatures, dilepton+sortedJets[0]+sortedJets[1]+met)

			evFeatures.append(numJets)									 		# 6
			evFeatures.append(nbjets)
			
			for jetTemp in sortedJets[2:]:
				extraJet = extraJet + jetTemp

			append4vector(evFeatures, extraJet)
			append4vector(evFeatures, dilepton+sortedJets[0]+sortedJets[1]+met+extraJet)
		
			mlb_array = []
			leptons = [lep1, lep2]
			# Find first lb min
			for j in range(len(sortedJets)):
				mlb_array.append((lep1+sortedJets[j]).M())
				mlb_array.append((lep2+sortedJets[j]).M())
			sorted_indices = np.argsort(mlb_array)
			append4vector(evFeatures, leptons[sorted_indices[0]%2] + sortedJets[sorted_indices[0]//2])
			append4vector(evFeatures, leptons[sorted_indices[1]%2] + sortedJets[sorted_indices[1]//2])
		
			#evFeatures.append(mlb_min)	

			append4vector(evFeatures, sortedJets[0])	
			evFeatures.append(btag[0])
			bscore = [bscore[j] if bscore[j]>0 else 0 for j in range(len(bscore))]
			evFeatures.append(bscore[0])	

			append4vector(evFeatures, sortedJets[1])
			evFeatures.append(btag[1])	
			evFeatures.append(bscore[1])		

			#append4vector(evFeatures, lep1)
			evFeatures.append(lep1.Pt())
			evFeatures.append(lep1.Eta())
			evFeatures.append(lep1.M())
			append4vector(evFeatures, lep2)
			evFeatures.append(met.Pt())		
			evFeatures.append(met.Phi())
			evFeatures.append(ht)
			
			dR_lep1_lep2 = lep1.DeltaR(lep2)
			dR_lepton1_jet1 = lep1.DeltaR(sortedJets[0])
			dR_lepton1_jet2 = lep1.DeltaR(sortedJets[1])
			dR_lepton2_jet1 = lep2.DeltaR(sortedJets[0])
			dR_lepton2_jet2 = lep2.DeltaR(sortedJets[1])
			dR_jet1_jet2 = sortedJets[0].DeltaR(sortedJets[1])	
			
			evFeatures.append(dR_lep1_lep2)
			evFeatures.append(dR_lepton1_jet1)
			evFeatures.append(dR_lepton1_jet2)
			evFeatures.append(dR_lepton2_jet1)
			evFeatures.append(dR_lepton2_jet2)
			evFeatures.append(dR_jet1_jet2)


			weights.append(totWeight)
			eventIn.append(evFeatures)
		tree.SetBranchStatus("*",1)

	return eventIn, weights, lKinRecoOut, kinRecoOut


def loadRegressionData4P(path, treeName, minbjets, nFiles):
    '''
    Pass the list of input output features, weights, and output of the invariant mass from LKR and FKR everything in list of list
    '''
    print("Searching root files in ", path)	
    fileNames = glob.glob(path+'/*.root')		# List of files.root in the directory
    nFiles = len(fileNames) if nFiles == None else nFiles
    fileNames =  [i for i in fileNames if "emu" in i][:nFiles]
    print (len(fileNames), " files to be used\n")

    eventIn, weights,lkrM,krM = [],[],[],[]	# Empty lists for input, output, weights, outputs of analytical results
    n=0
    for filename in fileNames:					# Looping over the file names
        n = n+1
        print ("\n",filename,"("+str(n)+"/"+str(len(fileNames))+")")
        i,w,l,k = createRealData(filename, treeName, minbjets=minbjets)
        eventIn+=i
        weights+=w
        lkrM+=l
        krM+=k
    print("\neventIn shape:\t", np.array(eventIn).shape)

    return eventIn, weights, lkrM, krM

