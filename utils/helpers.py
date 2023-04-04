import glob
import ROOT
import numpy as np
from progress.bar import IncrementalBar
import tensorflow as tf
import random


#definisci un append function che aggiunge il nome della variabile se non già presente
def getFeatureNames():
	feature_names =[
				'lkr_ttbar_pt', 'lkr_ttbar_eta', 'lkr_ttbar_phi',
				'lkr_ttbar_m',
				'kr_ttbar_pt', 'kr_ttbar_eta', 'kr_ttbar_phi',
				'kr_ttbar_m',
				'kr_top_pt', 'kr_top_eta', 'kr_top_phi', 'kr_top_m',
				'kr_antitop_pt', 'kr_antitop_eta', 'kr_antitop_phi', 'kr_antitop_m',
				'dilepton_pt', 'dilepton_eta', 'dilepton_phi', 'dilepton_m',
				'llj1j2_pt', 'llj1j2_eta', 'llj1j2_phi', 'llj1j2_m',
				'llj1j2MET_pt','llj1j2MET_eta', 'llj1j2MET_phi', 'llj1j2MET_m',
				'njets', 'nbjets',
				'extraJets_pt', 'extraJets_eta', 'extraJets_phi', 'extraJets_m',
				'lljjMETextraj_pt', 'lljjMETextraj_eta', 'lljjMETextraj_phi', 'lljjMETextraj_m',
				'lb1_pt', 'lb1_eta', 'lb1_phi', 'lb1_m',
				'lb2_pt', 'lb2_eta', 'lb2_phi', 'lb2_m',
				'jet1_pt', 'jet1_eta', 'jet1_phi', 'jet1_m', 'jet1btag', 'j1_bscore',
				'jet2_pt', 'jet2_eta', 'jet2_phi', 'jet2_m', 'jet2btag', 'j2_bscore',
				'lep1_pt', 'lep1_eta', 'lep1_phi', 'lep1_m',
				'lep2_pt', 'lep2_eta', 'lep2_phi', 'lep2_m',
				'met_pt', 'met_phi', 'ht', 
				'dR_l1l2','dR_l1j1', 'dR_l1j2', 'dR_l2j1', 'dR_l2j2', 'dR_j1j2','channelID'
				]
	#feature_names =['lkr_ttbar_M', 'kr_ttbar_M', 'dilepton_M', 'llj1j2_M', 'llj1j2MET_M', 'extraJets_M'] #  'channelID' , 
	return feature_names

def append4vector(evFeatures, a):
	evFeatures.append(a.Pt())
	evFeatures.append(a.Eta())
	evFeatures.append(a.Phi())
	evFeatures.append(a.M())


def rotation(jets, lep1, lep2, met):
	# Sign of Eta of First Jet
	signEta = lep1.Eta()/np.abs(lep1.Eta())
	# Phi first jet
	phiFirst = lep1.Phi()
	# Sign of Phi of Second Jet after the shift in Phi
	newPhiSecond = lep2.Phi() - lep1.Phi()
	newPhiSecond = newPhiSecond - 2*np.pi*(newPhiSecond > np.pi) + 2*np.pi*(newPhiSecond< -np.pi)
	signPhi = (newPhiSecond)/np.abs(newPhiSecond)

	

	for i in range(len(jets)):
		# Sign of Eta
		jets[i].SetPtEtaPhiM(jets[i].Pt(), jets[i].Eta()*signEta, jets[i].Phi(), jets[i].M())
		# Shift in Phi and bring back to [-pi, pi]
		jets[i].SetPhi( jets[i].Phi() - phiFirst)
		jets[i].SetPhi( jets[i].Phi() - 2*np.pi*(jets[i].Phi() > np.pi) + 2*np.pi*(jets[i].Phi() < -np.pi))
		# Sign of Phi
		jets[i].SetPhi(jets[i].Phi()*signPhi)

	lep1.SetPtEtaPhiM(lep1.Pt(), lep1.Eta()*signEta, lep1.Phi(),  lep1.M())	
	lep1.SetPhi( lep1.Phi() - phiFirst)
	lep1.SetPhi( lep1.Phi() - 2*np.pi*(lep1.Phi() > np.pi) + 2*np.pi*(lep1.Phi() < -np.pi))
	lep1.SetPhi(lep1.Phi()*signPhi)

	lep2.SetPtEtaPhiM(lep2.Pt(), lep2.Eta()*signEta, lep2.Phi(),  lep2.M())	
	lep2.SetPhi( lep2.Phi() - phiFirst)
	lep2.SetPhi( lep2.Phi() - 2*np.pi*(lep2.Phi() > np.pi) + 2*np.pi*(lep2.Phi() < -np.pi))
	lep2.SetPhi(lep2.Phi()*signPhi)

	met.SetPhi( met.Phi() - phiFirst)
	met.SetPhi( met.Phi() - 2*np.pi*(met.Phi() > np.pi) + 2*np.pi*(met.Phi() < -np.pi))
	met.SetPhi(met.Phi()*signPhi)




def NormalizeBinWidth1d(h):
	for i in range(1, h.GetNbinsX()+1):
		if h.GetBinContent(i) != 0:
			h.SetBinContent(i, h.GetBinContent(i)*1./h.GetBinWidth(i))
	return h

def loadMDataFlat(path, filename, treeName, minbjets, maxEvents):
	'''
	Open trees
	Set Branches to variables
	Define Lorentz Vector and compute observables of interest from measured ones
	Returns a list of lists
	'''
# List that will be converted to numpy array to be used in the NN
	eventIn=[]		# Input
	eventOut=[]		# Target
	kinRecoOut=[]	# kinRecoM (to be compared with NN output)
	lKinRecoOut=[]	# loosekinRecoM (to be compared with NN output)
	weights=[]		# weights to be used in the NN
	totGen = []
	

	f = ROOT.TFile.Open(path)
	tree = f.Get(treeName)

	channelID = None
	isEMuChannel = False
	if ("emu_") in filename:
		isEMuChannel = True
		channelID = 0
	elif ("mumu_") in filename:
		channelID = -1
	elif ("ee_") in filename:
		channelID = 1

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
	# Generated top, antitop
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


	maxEntries = tree.GetEntries() if maxEvents is None else maxEvents
	    
	bigNumber = 20000
	maxForBar = int(maxEntries/bigNumber)

# Tlorentz vectors
	if (True):
		lep1=ROOT.TLorentzVector(0.,0.,0.,0.)
		lep2=ROOT.TLorentzVector(0.,0.,0.,0.)
		met=ROOT.TLorentzVector(0.,0.,0.,0.)
		kr_top=ROOT.TLorentzVector(0.,0.,0.,0.)
		kr_antitop=ROOT.TLorentzVector(0.,0.,0.,0.)
		lkr_ttbar=ROOT.TLorentzVector(0.,0.,0.,0.)
		kr_nonbjet=ROOT.TLorentzVector(0.,0.,0.,0.)
		lkr_nonbjet=ROOT.TLorentzVector(0.,0.,0.,0.)
		kr_ttbar=ROOT.TLorentzVector(0.,0.,0.,0.)
		bjettemp=ROOT.TLorentzVector(0.,0.,0.,0.)
		top=ROOT.TLorentzVector(0.,0.,0.,0.)
		antitop=ROOT.TLorentzVector(0.,0.,0.,0.)
		ttbar=ROOT.TLorentzVector(0.,0.,0.,0.)
		dilepton=ROOT.TLorentzVector(0.,0.,0.,0.)
		zero=ROOT.TLorentzVector(0., 0., 0., 0.)
		extraJet = ROOT.TLorentzVector(0.,0.,0.,0.)


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
		top.SetPtEtaPhiM(0.,0.,0.,0.)
		antitop.SetPtEtaPhiM(0.,0.,0.,0.)
		ttbar.SetPtEtaPhiM(0.,0.,0.,0.)
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
# Cut in the MET (to suppress Zjets)
		passMETCut = False
		if isEMuChannel:
			passMETCut = True
		elif (met_pt[0]>40.):
			passMETCut = True
		else:
			passMETCut = False
# Generated
		top.SetPtEtaPhiM(gen_top_pt[0],gen_top_eta[0],gen_top_phi[0],gen_top_m[0])
		antitop.SetPtEtaPhiM(gen_antitop_pt[0],gen_antitop_eta[0],gen_antitop_phi[0],gen_antitop_m[0])
		ttbar = top+antitop
		totGen.append(ttbar.M())
# kinReco
		kr_top.SetPtEtaPhiM(kinReco_top_pt[0],kinReco_top_eta[0],kinReco_top_phi[0],kinReco_top_m[0])
		kr_antitop.SetPtEtaPhiM(kinReco_antitop_pt[0],kinReco_antitop_eta[0],kinReco_antitop_phi[0],kinReco_antitop_m[0])
		kr_ttbar= kr_antitop + kr_top
# LooseKinReco
		lkr_ttbar.SetPtEtaPhiM(looseKinReco_ttbar_pt[0],looseKinReco_ttbar_eta[0],looseKinReco_ttbar_phi[0],looseKinReco_ttbar_m[0])
		

		# conditions fro building an input events (training or testing)
		#if (pass3 & (numJets>=2) & passMETCut & (dilepton.M()>20.) & passMLLCut & (ttbar.M()>0) & (kr_ttbar.M()<1500) & (lkr_ttbar.M() < 1500)  ):# & (kr_ttbar.M()>1) & (lkr_ttbar.M() > 1)  & 
		if ( (numJets>=2) & (passMETCut) & passMLLCut & (haslkrs) & (looseKinReco_ttbar_m[0] < 1500) & (haskrs)& (kr_ttbar.M()<1500) ):
			if ((ttbar.M()<0)):
				print("Negative ttbar generated!")
			if not pass3:
				print("Step3 not passed")
			if (dilepton.M()<20.):
				print("dilepton sotto i 20")
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
			nbjets = len(bjets)
			sortedJets = bjets + nonbjets
			met.SetPtEtaPhiM(met_pt[0],0.,met_phi[0],0.)
			if len(bjets)<minbjets:
				continue
			rotation(sortedJets, lep1, lep2, met)	

			if (haslkrs & (lkr_ttbar.M()>0)):
				#evFeatures.append(lkr_ttbar.M())
				#append4vector(evFeatures, lkr_ttbar)
				evFeatures.append(looseKinReco_ttbar_pt[0])
				evFeatures.append(looseKinReco_ttbar_eta[0])
				evFeatures.append(looseKinReco_ttbar_phi[0])
				evFeatures.append(looseKinReco_ttbar_m[0])
			else:
				#evFeatures.append(0)				
				append4vector(evFeatures, zero)
			if (haskrs & (kr_ttbar.M()>0)):
				#evFeatures.append(kr_ttbar.M())
				append4vector(evFeatures, kr_ttbar)
				evFeatures.append(kinReco_antitop_pt[0])
				evFeatures.append(kinReco_antitop_eta[0])
				evFeatures.append(kinReco_antitop_phi[0])
				evFeatures.append(kinReco_antitop_m[0])
				append4vector(evFeatures, kr_antitop)
			else:
				#evFeatures.append(0)
				append4vector(evFeatures, zero)
				append4vector(evFeatures, zero)
				append4vector(evFeatures, zero)

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

			append4vector(evFeatures, lep1)
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
			evFeatures.append(channelID)

			
			lKinRecoOut.append(lkr_ttbar.M())   # 2	Output of the LooseKinReco
			kinRecoOut.append(kr_ttbar.M())     # 
			weights.append(totWeight)
			eventIn.append(evFeatures)
			eventOut.append([ttbar.M()])
		tree.SetBranchStatus("*",1)

	return eventIn, eventOut, weights, lKinRecoOut, kinRecoOut, totGen


def loadRegressionData(path, treeName,nFiles, minbjets, maxEvents):
    '''
    Pass the list of input output features, weights, and output of the invariant mass from LKR and FKR everything in list of list
    '''
    print("loadRegressionData called \n\n")
    print("Searching root files in ", path)	
    fileNames = glob.glob(path+'/*.root')		# List of files.root in the directory
    fileNames = [i for i in fileNames if "ee" in i][:nFiles] + [i for i in fileNames if "mumu" in i][:nFiles] + [i for i in fileNames if "emu" in i][:nFiles]
    #fileNames = [i for i in fileNames if "ee" in i][:nFiles]
    print (len(fileNames), " files to be used\n")
    #random.shuffle(fileNames)

    eventIn, eventOut, weights,lkrM,krM, totGen = [],[],[],[],[], []	# Empty lists for input, output, weights, outputs of analytical results
    n=0
    for filename in fileNames:					# Looping over the file names
        n = n+1
        print ("\n",filename,"("+str(n)+"/"+str(len(fileNames))+")")
        i,o,w,l,k, t = loadMDataFlat(filename, filename, treeName, minbjets=minbjets, maxEvents=maxEvents)
        eventIn+=i
        eventOut+=o
        weights+=w
        lkrM+=l
        krM+=k
        totGen+=t
    print("eventIn shape:\t", np.array(eventIn).shape)
    print("totGen shape:\t",  np.array(totGen).shape)
    return eventIn, eventOut, weights, lkrM, krM, totGen


"""def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.compat.v1.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.compat.v1.global_variables()]
        graphdef_inf = tf.compat.v1.graph_util.remove_training_nodes(graph.as_graph_def())
        if clear_devices:
            for node in graphdef_inf.node:
                node.device = ""
        frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
            session, graphdef_inf, output_names, freeze_var_names)
        return frozen_graph"""