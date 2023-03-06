import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import ROOT
import glob
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_WARNINGS'] = 'off'
from keras import backend as K
import tensorflow as tf
import keras.backend as K
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if(len(physical_devices)>0):
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

import uproot
import numpy as np
from utils import style
from sklearn.model_selection import train_test_split
from numba import jit

feature_names_all=[
    "jet1Pt","jet1Eta","jet1Phi","jet1M","jet1BTag",
    "jet2Pt","jet2Eta","jet2Phi","jet3M","jet2BTag",
    #"jet3Pt","jet3Eta","jet3Phi","jet3M","jet3BTag",
    "ht","nBJets","mlbmin",
    "l1j1DR","l2j1DR","l1j2DR","l2j2DR",
#"l1j3DR","l2j3DR",
    "j1j2DR",
#"j1j3DR","l2j3DR",
    "dilepPt","dilepEta","dilepPhi","dilepM",
    "lep1Pt","lep1Eta","lep1Phi","lep1M",
    "lep2Pt","lep2Eta","lep2Phi","lep2M",
    "metPt","metPhi",
    "lkr_rho","kr_rho",
    "kr_ttbarPt","kr_ttbarEta","kr_ttbarPhi","kr_ttbarM",
    "kr_topPt","kr_topEta","kr_topPhi",
    "kr_atopPt","kr_atopEta","kr_atopPhi",
    "kr_jetPt","kr_jetEta","kr_jetPhi","kr_jetM",
    "llj1Pt","llj1Eta","llj1Phi","llj1M",
    "llj2Pt","llj2Eta","llj2Phi","llj2M",
    #"llj3Pt","llj3Eta","llj3Phi","llj3M",
    "l1j1Pt","l1j1Eta","l1j1Phi","l1j1M",
    "l1j2Pt","l1j2Eta","l1j2Phi","l1j2M",
    #"l1j3Pt","l1j3Eta","l1j3Phi","l1j3M",
    "l2j1Pt","l2j1Eta","l2j1Phi","l2j1M",
    "l2j2Pt","l2j2Eta","l2j2Phi","l2j2M",
    #"l2j3Pt","l2j3Eta","l2j3Phi","l2j3M",
    "njets",
    "lkr_ttbarPt","lkr_ttbarEta","lkr_ttbarPhi","lkr_ttbarM",
    "lkr_jetPt","lkr_jetEta","lkr_jetPhi","lkr_jetM",
    "yearID", "channelID"
]


######################### REGRESSION ##################################################################


def loadRegressionData(path, treeName, nJets=2, maxEvents=0, withBTag = False, pTEtaPhiMode=False):
    '''
    Pass the list of input output features, weights, and output of the invariant mass from LKR and FKR everything in list of list
    '''
    print("loadRegressionData called \n\n")
    pathToSearch = path.replace("FR2","*")
    fileNames = glob.glob(pathToSearch+'*.root')		# List of files.root in the directory
    print (path)
    print (fileNames)
    eventInJet, eventOut, weights,lkrM,krM = [],[],[],[],[]	# Empty lists for input, output, weights, outputs of analytical results
    n=0
    for filename in fileNames:					# Looping over the file names
        n = n+1
        if "FR2" in path:
            filename = filename.replace(path,"")
        print ("\n",filename,"("+str(n)+"/"+str(len(fileNames))+")")
        a,b,c,l,k = loadMDataFlat(filename, filename, treeName, nJets=nJets, maxEvents=maxEvents, withBTag = withBTag, pTEtaPhiMode = pTEtaPhiMode)
        eventInJet+=a
        eventOut+=b
        weights+=c
        lkrM+=l
        krM+=k
    return eventInJet, eventOut, weights,lkrM,krM


def loadMDataFlat(path,filename, treeName, nJets=3, maxEvents=0, withBTag = False, pTEtaPhiMode=False):
	'''
	Open trees
	Set Branches to variables
	Define Lorentz Vector and compute observables of interest from measured ones
	Returns a list of lists
	'''
# List that will be converted to numpy array to be used in the NN
	eventInJet=[]	# Input
	eventOut=[]	# Target
	kinRecoOut=[]	# kinRecorho (to be compared with NN output)
	lKinRecoOut=[]	# loosekinRecorho (to be compared with NN output)
	weights=[]	# weights to be used in the NN
	maxJets=nJets	# max number of jets(?)

	f = ROOT.TFile.Open(path)
	tree = f.Get(treeName)

	channel = None
	channelID = None
	year = None
	yearID = None

	isEMuChannel = False
	if ("emu_") in filename:
		isEMuChannel = True
		channel = "emu"
		channelID = 0
	elif ("mumu_") in filename:
		channel = "mumu"
		channelID = -1
	elif ("ee_") in filename:
		channel = "ee"
		channelID = 1

	if ("/2016/") in path:
		year = "2016"
		yearID = -1
	elif ("/2017/") in path:
		year = "2017"
		yearID = 0
	elif ("/2018/") in path:
		year = "2018"
		yearID = 1

	print (path, filename, year, yearID, channel, channelID)

	print("Read TTree: {} (Entries: {})".format(treeName, tree.GetEntries()))
# Setting branches in the tree
	tree.SetBranchStatus("var_*",0)

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


	lkr_nonbjetPt = np.array([0], dtype='f')
	tree.SetBranchAddress("looseKinReco_nonbjet_pt", lkr_nonbjetPt)
	lkr_nonbjetEta = np.array([0], dtype='f')
	tree.SetBranchAddress("looseKinReco_nonbjet_eta", lkr_nonbjetEta)
	lkr_nonbjetPhi = np.array([0], dtype='f')
	tree.SetBranchAddress("looseKinReco_nonbjet_phi", lkr_nonbjetPhi)
	lkr_nonbjetM = np.array([0], dtype='f')
	tree.SetBranchAddress("looseKinReco_nonbjet_m", lkr_nonbjetM)
	lkr_rho = np.array([0], dtype='f')
	tree.SetBranchAddress("looseKinReco_rho", lkr_rho)

	kr_nonbjetPt = np.array([0], dtype='f')
	tree.SetBranchAddress("kinReco_nonbjet_pt", kr_nonbjetPt)
	kr_nonbjetEta = np.array([0], dtype='f')
	tree.SetBranchAddress("kinReco_nonbjet_eta", kr_nonbjetEta)
	kr_nonbjetPhi = np.array([0], dtype='f')
	tree.SetBranchAddress("kinReco_nonbjet_phi", kr_nonbjetPhi)
	kr_nonbjetM = np.array([0], dtype='f')
	tree.SetBranchAddress("kinReco_nonbjet_m", kr_nonbjetM)
	kr_rho = np.array([0], dtype='f')
	tree.SetBranchAddress("kinReco_rho", kr_rho)

	jetBTagged = np.array([0]*20, dtype='f')
	tree.SetBranchAddress("jets_btag", (jetBTagged))

	genpartonjetPt = np.array([0], dtype='f')
	tree.SetBranchAddress("gen_partonLevelGhostCleaned_additional_jet_pt", genpartonjetPt)

	genpartonjetEta = np.array([0], dtype='f')
	tree.SetBranchAddress("gen_partonLevelGhostCleaned_additional_jet_eta", genpartonjetEta)

	genpartonjetPhi = np.array([0], dtype='f')
	tree.SetBranchAddress("gen_partonLevelGhostCleaned_additional_jet_phi", genpartonjetPhi)

	genpartonjetM = np.array([0], dtype='f')
	tree.SetBranchAddress("gen_partonLevelGhostCleaned_additional_jet_m", genpartonjetM)

	genpartonrho = np.array([0], dtype='f')
	tree.SetBranchAddress("gen_partonLevelGhostCleaned_rho", genpartonrho)

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

	met_pt =  np.array([0], dtype='f')
	tree.SetBranchAddress("met_pt", met_pt)
	met_phi =  np.array([0], dtype='f')
	tree.SetBranchAddress("met_phi", met_phi)

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

	looseKinReco_ttbar_pt =  np.array([0], dtype='f')
	tree.SetBranchAddress("looseKinReco_ttbar_pt", looseKinReco_ttbar_pt)
	looseKinReco_ttbar_eta =  np.array([0], dtype='f')
	tree.SetBranchAddress("looseKinReco_ttbar_eta", looseKinReco_ttbar_eta)
	looseKinReco_ttbar_phi =  np.array([0], dtype='f')
	tree.SetBranchAddress("looseKinReco_ttbar_phi", looseKinReco_ttbar_phi)
	looseKinReco_ttbar_m =  np.array([0], dtype='f')
	tree.SetBranchAddress("looseKinReco_ttbar_m", looseKinReco_ttbar_m)

	from progress.bar import IncrementalBar
# If maxevents is defined only a fraction of events
	if maxEvents!=0:
	    maxEntries = maxEvents
	else:
	    maxEntries = tree.GetEntries()

	maxForBar = int(maxEntries/200000)

	lep1=ROOT.TLorentzVector(0.,0.,0.,0.)
	lep2=ROOT.TLorentzVector(0.,0.,0.,0.)
	jet4=ROOT.TLorentzVector(0.,0.,0.,0.)
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

	bar = IncrementalBar('Processing', max=maxForBar, suffix='%(percent).1f%% - %(eta)ds')
	print("\nLooping in the trees")  # GC
	for i in range(tree.GetEntries()):

		lep1.SetPtEtaPhiM(0.,0.,0.,0.)
		lep2.SetPtEtaPhiM(0.,0.,0.,0.)
		jet4.SetPtEtaPhiM(0.,0.,0.,0.)
		met.SetPtEtaPhiM(0.,0.,0.,0.)
		kr_top.SetPtEtaPhiM(0.,0.,0.,0.)
		kr_antitop.SetPtEtaPhiM(0.,0.,0.,0.)
		lkr_ttbar.SetPtEtaPhiM(0.,0.,0.,0.)
		kr_nonbjet.SetPtEtaPhiM(0.,0.,0.,0.)
		kr_ttbar.SetPtEtaPhiM(0.,0.,0.,0.)
		lkr_nonbjet.SetPtEtaPhiM(0.,0.,0.,0.)
		bjettemp.SetPtEtaPhiM(0.,0.,0.,0.)
		top.SetPtEtaPhiM(0.,0.,0.,0.)
		antitop.SetPtEtaPhiM(0.,0.,0.,0.)
		ttbar.SetPtEtaPhiM(0.,0.,0.,0.)
		dilepton.SetPtEtaPhiM(0.,0.,0.,0.)

		jets_info=[]
		jets_matchInfo=[]
		other_info=[]
		tree.GetEntry(i)
		totWeight=1.

		if(i%200000==0):
		    bar.next()

		if(maxEvents!=0):
			if(i%maxEvents==0 and i!=0):
				break

		pass3= bool(passStep3[0])
		haskrs = bool(hasKinRecoSolution[0])
		haslkrs = bool(hasLooseKinRecoSolution[0])
		totWeight=weight[0]*btagSF[0]*leptonSF[0]*pileupSF[0]*prefiringWeight[0]	# definition of weights
# Pass3: correct reconstruction of leptons
		if (pass3==True):
			lep1.SetPtEtaPhiM(lepton1_pt[0],lepton1_eta[0],lepton1_phi[0],lepton1_m[0])
			lep2.SetPtEtaPhiM(lepton2_pt[0],lepton2_eta[0],lepton2_phi[0],lepton2_m[0])
		else:
			lep1.SetPtEtaPhiM(0,0,0,0)
			lep2.SetPtEtaPhiM(0,0,0,0)
		dilepton=lep1+lep2
# Mll cut in the Z window
		passMLLCut = False
# If emu always passed
		if isEMuChannel:
			passMLLCut = True
		elif not (dilepton.M()>76. and dilepton.M()<106.):
			passMLLCut = True
		else:
			passMLLCut = False
# Cut in the MET (to suppress Zjets)
		passMETCut = False
# if emu always passed

		if isEMuChannel:
			passMETCut = True
		elif (met_pt[0]>40.):
			passMETCut = True
		else:
			passMETCut = False

		if(pass3==True):
			numJets = njets[0]
			countB=0.

			# Top properties definition
			top.SetPtEtaPhiM(gen_top_pt[0],gen_top_eta[0],gen_top_phi[0],gen_top_m[0])
			antitop.SetPtEtaPhiM(gen_antitop_pt[0],gen_antitop_eta[0],gen_antitop_phi[0],gen_antitop_m[0])
			ttbar = top+antitop
# number of jets requirements
			if ((numJets==2) and passMETCut and (dilepton.M()>20.) and passMLLCut):
				if(ttbar.M()>0. and genpartonjetPt>30. and abs(genpartonjetEta)<2.4):
					jets = []
					bjets = []
					nbjets = 0
					ht = 0
					for idx in range(maxJets):
						if(idx<numJets):
							jet4=ROOT.TLorentzVector(0.,0.,0.,0.)
							jet4.SetPtEtaPhiM(jetPt[idx],jetEta[idx],jetPhi[idx],jetM[idx])
							if pTEtaPhiMode:
								jets_info.append(jet4.Pt())
								jets_info.append(jet4.Eta())
								jets_info.append(jet4.Phi())
								jets_info.append(jet4.M())
							else:
								jets_info.append(jet4.Px()) #1
								jets_info.append(jet4.Py())#2
								jets_info.append(jet4.Pz())#3
								jets_info.append(jet4.E())#4
							jets.append(jet4) #1-12
							bTagged=int(bool(jetBTagged[idx]))
							ht = ht+jet4.Pt()
							if withBTag:
								jets_info.append(bTagged)
							if(bTagged):
								bjets.append(jet4)
						else:
							jets_info.append(0.)
							jets_info.append(0.)
							jets_info.append(0.)
							jets_info.append(0.)
							if withBTag:
							    jets_info.append(0.)
					weights.append(totWeight)

					lep1.SetPtEtaPhiM(lepton1_pt[0],lepton1_eta[0],lepton1_phi[0],lepton1_m[0])
					lep2.SetPtEtaPhiM(lepton2_pt[0],lepton2_eta[0],lepton2_phi[0],lepton2_m[0])
					met.SetPtEtaPhiM(met_pt[0],0.,met_phi[0],0.)
					

					dilepton=lep1+lep2

					kr_top.SetPtEtaPhiM(kinReco_top_pt[0],kinReco_top_eta[0],kinReco_top_phi[0],kinReco_top_m[0])
					kr_antitop.SetPtEtaPhiM(kinReco_antitop_pt[0],kinReco_antitop_eta[0],kinReco_antitop_phi[0],kinReco_antitop_m[0])
					kr_ttbar= kr_antitop + kr_top
					lkr_ttbar.SetPtEtaPhiM(looseKinReco_ttbar_pt[0],looseKinReco_ttbar_eta[0],looseKinReco_ttbar_phi[0],looseKinReco_ttbar_m[0])

					if (kr_nonbjetPt[0]>30. and abs(kr_nonbjetEta[0])<2.4):
						kr_nonbjet.SetPtEtaPhiM(kr_nonbjetPt[0],kr_nonbjetEta[0],kr_nonbjetPhi[0],kr_nonbjetM[0])
					if (lkr_nonbjetPt[0]>30. and abs(lkr_nonbjetEta[0])<2.4):
						lkr_nonbjet.SetPtEtaPhiM(lkr_nonbjetPt[0],lkr_nonbjetEta[0],lkr_nonbjetPhi[0],lkr_nonbjetM[0])

					mlb_min = 9999.
					comb = 9999
					comb1 = 9999
					comb2 = 9999
					nBJets = len(bjets)
					nbjets = len(bjets)
					for i in range(nBJets):
					    bjettemp.SetPtEtaPhiM(bjets[i].Pt(),bjets[i].Eta(),bjets[i].Phi(),bjets[i].M())
					    comb1 = (lep1+bjettemp).M()
					    comb2 = (lep2+bjettemp).M()
					    comb = comb1 if comb1<comb2 else comb2
					    if(comb<mlb_min):
					        mlb_min=comb
					if mlb_min > 9990:
					    mlb_min = 0.

					dR_lepton1_jet1 = lep1.DeltaR(jets[0])
					dR_lepton1_jet2 = lep1.DeltaR(jets[1])
					#dR_lepton1_jet3 = lep1.DeltaR(jets[2])
					dR_lepton2_jet1 = lep2.DeltaR(jets[0])
					dR_lepton2_jet2 = lep2.DeltaR(jets[1])
					#dR_lepton2_jet3 = lep2.DeltaR(jets[2])

					dR_jet1_jet2 = jets[0].DeltaR(jets[1])
					#dR_jet1_jet3 = jets[0].DeltaR(jets[2])
					#dR_jet2_jet3 = jets[1].DeltaR(jets[2])

					jets_info.append(ht)  # 16
					jets_info.append(nbjets) #17

					jets_info.append(mlb_min) #13
					# jets_info.append((lep1+jets[0]).M()) #18
					# jets_info.append((lep2+jets[0]).M()) #19
					# jets_info.append((lep1+jets[1]).M()) #20
					# jets_info.append((lep2+jets[1]).M()) #21
					# jets_info.append((lep1+jets[2]).M()) #22
					# jets_info.append((lep2+jets[2]).M()) #23
					jets_info.append(dR_lepton1_jet1) #24
					jets_info.append(dR_lepton1_jet2) #25
					#jets_info.append(dR_lepton1_jet3)#26
					jets_info.append(dR_lepton2_jet1)#27
					jets_info.append(dR_lepton2_jet2)#28
					#jets_info.append(dR_lepton2_jet3)#29
					jets_info.append(dR_jet1_jet2)#30
					#jets_info.append(dR_jet1_jet3)#31
					#jets_info.append(dR_jet2_jet3)#32


					jets_info.append(dilepton.Pt())#33
					jets_info.append(dilepton.Eta())#34
					jets_info.append(dilepton.Phi())#35
					jets_info.append(dilepton.M())#36


					jets_info.append(lep1.Pt()) #37
					jets_info.append(lep1.Eta())#38
					jets_info.append(lep1.Phi())#39
					jets_info.append(lep1.M())#40

					jets_info.append(lep2.Pt())#41
					jets_info.append(lep2.Eta())#42
					jets_info.append(lep2.Phi())#43
					jets_info.append(lep2.M())#44

					jets_info.append(met.Pt())#45
					# jets_info.append(met.Eta())
					jets_info.append(met.Phi())#46
					# jets_info.append(met.M())

					if (haslkrs and lkr_ttbar.M()>0.):
					    jets_info.append(lkr_ttbar.M())#47
					    lKinRecoOut.append(lkr_ttbar.M())#47	Output of the LooseKinReco
					else:
					    jets_info.append(0.)
					    lKinRecoOut.append(0.)
					if (haskrs and kr_ttbar.M()>0.):
					    jets_info.append(kr_ttbar.M())#48
					    kinRecoOut.append(kr_ttbar.M())#48		Output of the kinRecoOut
					else:
					    jets_info.append(0.)
					    kinRecoOut.append(0.)

					if (haskrs and kr_ttbar.M()>0.):
					    # jets_info.append(kr_rho) #51
					    jets_info.append(kr_ttbar.Pt())#49
					    jets_info.append(kr_ttbar.Eta())#50
					    jets_info.append(kr_ttbar.Phi())#51
					    jets_info.append(kr_ttbar.M())#52
					    jets_info.append(kr_top.Pt())#53
					    jets_info.append(kr_top.Eta())#54
					    jets_info.append(kr_top.Phi())#55
					    # jets_info.append(kr_top.M())
					    jets_info.append(kr_antitop.Pt())#56
					    jets_info.append(kr_antitop.Eta())#57
					    jets_info.append(kr_antitop.Phi())#58
					    # jets_info.append(kr_antitop.M())
					    if(kr_nonbjet.M()>0.):
					        jets_info.append(kr_nonbjet.Pt())#59
					        jets_info.append(kr_nonbjet.Eta())#60
					        jets_info.append(kr_nonbjet.Phi())#61
					        jets_info.append(kr_nonbjet.M())#62
					    else:
					        jets_info.append(0.)#
					        jets_info.append(0.)
					        jets_info.append(0.)
					        jets_info.append(0.)
					else:
					    jets_info.append(0.)
					    jets_info.append(0.)
					    jets_info.append(0.)
					    jets_info.append(0.)
					    jets_info.append(0.)
					    jets_info.append(0.)
					    jets_info.append(0.)
					    jets_info.append(0.)
					    jets_info.append(0.)
					    jets_info.append(0.)
					    jets_info.append(0.)
					    jets_info.append(0.)
					    jets_info.append(0.)
					    jets_info.append(0.)

					jets_info.append((dilepton+jets[0]).Pt()) #63
					jets_info.append((dilepton+jets[0]).Eta()) #64
					jets_info.append((dilepton+jets[0]).Phi()) #65
					jets_info.append((dilepton+jets[0]).M()) #66
					jets_info.append((dilepton+jets[1]).Pt()) #67
					jets_info.append((dilepton+jets[1]).Eta()) #68
					jets_info.append((dilepton+jets[1]).Phi()) #69
					jets_info.append((dilepton+jets[1]).M()) #70
					#jets_info.append((dilepton+jets[2]).Pt()) #71
					#jets_info.append((dilepton+jets[2]).Eta()) #72
					#jets_info.append((dilepton+jets[2]).Phi()) #73
					#jets_info.append((dilepton+jets[2]).M()) #74

					jets_info.append((lep1+jets[0]).Pt()) #75
					jets_info.append((lep1+jets[0]).Eta()) #76
					jets_info.append((lep1+jets[0]).Phi()) #77
					jets_info.append((lep1+jets[0]).M()) #78
					jets_info.append((lep1+jets[1]).Pt()) #79
					jets_info.append((lep1+jets[1]).Eta()) #80
					jets_info.append((lep1+jets[1]).Phi()) #81
					jets_info.append((lep1+jets[1]).M()) #82
					#jets_info.append((lep1+jets[2]).Pt()) #83
					#jets_info.append((lep1+jets[2]).Eta()) #84
					#jets_info.append((lep1+jets[2]).Phi()) #85
					#jets_info.append((lep1+jets[2]).M()) #86

					jets_info.append((lep2+jets[0]).Pt()) #87
					jets_info.append((lep2+jets[0]).Eta()) #88
					jets_info.append((lep2+jets[0]).Phi()) #89
					jets_info.append((lep2+jets[0]).M()) #90
					jets_info.append((lep2+jets[1]).Pt()) #91
					jets_info.append((lep2+jets[1]).Eta()) #92
					jets_info.append((lep2+jets[1]).Phi()) #93
					jets_info.append((lep2+jets[1]).M()) #94
					#jets_info.append((lep2+jets[2]).Pt()) #95
					#jets_info.append((lep2+jets[2]).Eta()) #96
					#jets_info.append((lep2+jets[2]).Phi()) #97
					#jets_info.append((lep2+jets[2]).M()) #98

					jets_info.append(numJets) #99

					if (haslkrs and lkr_ttbar.M()>0.):
					    jets_info.append(lkr_ttbar.Pt()) #100
					    jets_info.append(lkr_ttbar.Eta())#101
					    jets_info.append(lkr_ttbar.Phi())#102
					    jets_info.append(lkr_ttbar.M())#103
					    if(lkr_nonbjet.M()>0.):
					        jets_info.append(lkr_nonbjet.Pt())#104
					        jets_info.append(lkr_nonbjet.Eta())#105
					        jets_info.append(lkr_nonbjet.Phi())#106
					        jets_info.append(lkr_nonbjet.M())#107
					    else:
					        jets_info.append(0.)
					        jets_info.append(0.)
					        jets_info.append(0.)
					        jets_info.append(0.)
					else:
					    jets_info.append(0.)
					    jets_info.append(0.)
					    jets_info.append(0.)
					    jets_info.append(0.)
					    jets_info.append(0.)
					    jets_info.append(0.)
					    jets_info.append(0.)
					    jets_info.append(0.)

					jets_info.append(yearID)
					jets_info.append(channelID)

					eventInJet.append(jets_info)	# Input features are all the possible combinations of momenta
					eventOut.append([ttbar.M()])	# target of NN is generated M
	tree.SetBranchStatus("*",1)
	return eventInJet,eventOut,weights, lKinRecoOut, kinRecoOut

######################### TOOLS ##################################################################

#
def getReducedFeatureNames(tokeep=None):
    feature_names = [i for i in feature_names_all]
    for i in range(len(feature_names)):
        feature_names[i]=feature_names[i].replace(feature_names[i],str(i)+"_"+feature_names[i])

    feature_names_new = []

    to_remove = []
    if tokeep == None:
        to_keep = [i for i in range(len(feature_names))] #all
    else:
        to_keep = tokeep

    for i in range(len(feature_names)):
        if i in to_keep:
            feature_names_new.append(feature_names[i])
        else:
            to_remove.append(i)

    feature_names = feature_names_new

    return feature_names

'''
def getReducedFeatureNamesAndInputs(inX_train, inX_test, tokeep=None):
    feature_names = [i for i in feature_names_all]
    for i in range(len(feature_names)):
        feature_names[i]=feature_names[i].replace(feature_names[i],str(i)+"_"+feature_names[i])

    feature_names_new = []

    to_remove = []
    if tokeep == None:
        to_keep = [i for i in range(len(feature_names))] #all
    else:
        to_keep = tokeep

    for i in range(len(feature_names)):
        if i in to_keep:
            feature_names_new.append(feature_names[i])
        else:
            to_remove.append(i)

    feature_names = feature_names_new

    inX_train=np.delete(inX_train,  to_remove,1)
    inX_test=np.delete(inX_test,  to_remove,1)

    return feature_names, inX_train, inX_test


def getReducedFeatureNamesAndInputsWithSecondSet(inX_train, inX_test, tokeep=None, tokeep2=None, regDNNPath = "", removeRho = False):

    model_reg_temp = tf.keras.models.load_model(regDNNPath)

    feature_names = [i for i in feature_names_all]
    feature_names2 = [i for i in feature_names_all]
    feature_names_krRho = [i for i in feature_names_all]
    feature_names_lkrRho = [i for i in feature_names_all]
    for i in range(len(feature_names)):
        feature_names[i]=feature_names[i].replace(feature_names[i],str(i)+"_"+feature_names[i])
        feature_names2[i]=feature_names[i].replace(feature_names[i],str(i)+"_"+feature_names[i])
        feature_names_krRho[i]=feature_names[i].replace(feature_names[i],str(i)+"_"+feature_names[i])
        feature_names_lkrRho[i]=feature_names[i].replace(feature_names[i],str(i)+"_"+feature_names[i])

    feature_names_new = []
    feature_names_new2 = []
    feature_names_new_krRho = []
    feature_names_new_lkrRho = []

    to_remove = []
    if tokeep == None:
        to_keep = [i for i in range(len(feature_names))] #all
    else:
        to_keep = tokeep

    for i in range(len(feature_names)):
        if i in to_keep:
            feature_names_new.append(feature_names[i])
        else:
            to_remove.append(i)
    if removeRho:
        to_remove.append(41)
        to_remove.append(42)

    feature_names = feature_names_new

    inX_train1=np.delete(inX_train,  to_remove,1)
    inX_test1=np.delete(inX_test,  to_remove,1)

    to_remove2 = []
    if tokeep2 == None:
        to_keep2 = [i for i in range(len(feature_names2))] #all
    else:
        to_keep2 = tokeep2

    for i in range(len(feature_names2)):
        if i in to_keep2:
            feature_names_new2.append(feature_names2[i])
        else:
            to_remove2.append(i)

    feature_names2 = feature_names_new2

    to_remove_krRho = []
    tokeep_krRho = [42]
    to_remove_lkrRho = []
    tokeep_lkrRho = [41]
    if tokeep_krRho == None:
        to_keep_krRho = [i for i in range(len(feature_names_krRho))] #all
    else:
        to_keep_krRho = tokeep_krRho
    if tokeep_lkrRho == None:
        to_keep_lkrRho = [i for i in range(len(feature_names_lkrRho))] #all
    else:
        to_keep_lkrRho = tokeep_lkrRho

    for i in range(len(feature_names_krRho)):
        if i in to_keep_krRho:
            feature_names_new_krRho.append(feature_names_krRho[i])
        else:
            to_remove_krRho.append(i)
    for i in range(len(feature_names_lkrRho)):
        if i in to_keep_lkrRho:
            feature_names_new_lkrRho.append(feature_names_lkrRho[i])
        else:
            to_remove_lkrRho.append(i)

    feature_names_krRho = feature_names_new_krRho
    feature_names_lkrRho = feature_names_new_lkrRho

    inX_train2=np.delete(inX_train,  to_remove2,1)
    inX_test2=np.delete(inX_test,  to_remove2,1)
    inX_train_krRho=np.delete(inX_train,  to_remove_krRho, 1)
    inX_test_krRho=np.delete(inX_test,  to_remove_krRho, 1)
    inX_train_lkrRho=np.delete(inX_train,  to_remove_lkrRho, 1)
    inX_test_lkrRho=np.delete(inX_test,  to_remove_lkrRho, 1)

    outRho=[]
    outRho_test=[]

    for x, krRho, lkrRho in zip(inX_train2,inX_train_krRho,inX_train_lkrRho):
        arAv = []
        if krRho[0] >0.:
            arAv.append(np.array(krRho[0]))
        if lkrRho[0] >0.:
            arAv.append(np.array(lkrRho[0]))

        r = model_reg_temp.predict(np.array([x]))
        arAv.append(r[0][0])
        arAv = np.array(arAv)
        arAv.flatten()
        outRho.append( r[0][0])

    for x, krRho, lkrRho in zip(inX_test2,inX_test_krRho,inX_test_lkrRho):
        arAv = []
        if krRho[0] >0.:
            arAv.append(krRho[0])
        if lkrRho[0] >0.:
            arAv.append(lkrRho[0])
        r = model_reg_temp.predict(np.array([x]))
        arAv.append(r[0][0])
        arAv = np.array(arAv)
        arAv.flatten()
        outRho_test.append(r[0][0])

    return feature_names, inX_train1, inX_test1,outRho,outRho_test

'''
def rebin2D(h, ngx, ngy):
    hold = h.Clone()
    hold.SetDirectory(0)
    h2 = h.Clone()
    h2.SetDirectory(0)

    nbinsx = hold.GetXaxis().GetNbins()
    nbinsy = hold.GetYaxis().GetNbins()
    xmin  = hold.GetXaxis().GetXmin()
    xmax  = hold.GetXaxis().GetXmax()
    ymin  = hold.GetYaxis().GetXmin()
    ymax  = hold.GetYaxis().GetXmax()
    nx = int(nbinsx/ngx)
    ny = int(nbinsy/ngy)

    h2.SetBins(nx,xmin,xmax,ny,ymin,ymax)

    for biny in range(1,nbinsy+1):
        for binx in range(nbinsx+1):
            ibin = h2.GetBin(binx,biny)
            h2.SetBinContent(ibin,0)

    for biny in range(1,nbinsy+1):
        by  = hold.GetYaxis().GetBinCenter(biny)
        iy  = h2.GetYaxis().FindBin(by)
        for binx in range(1,nbinsx+1):
            bx = hold.GetXaxis().GetBinCenter(binx)
            ix  = h2.GetXaxis().FindBin(bx)
            bin = hold.GetBin(binx,biny)
            ibin= h2.GetBin(ix,iy)
            cu  = hold.GetBinContent(bin)
            h2.AddBinContent(ibin,cu)
    return h2

def drawAsGraph(h,option="pe0"):
    g = ROOT.TGraphAsymmErrors()
    for b in range(h.GetNbinsX()):
        x = h.GetBinLowEdge(b + 1) + 1. * h.GetBinWidth(b + 1)
        y = h.GetBinContent(b + 1)
        uncLowY = h.GetBinError(b + 1)
        uncHighY = h.GetBinError(b + 1)
        g.SetPoint(b, x, y)
        g.SetPointError(b, 0.0, 0.0, uncLowY, uncHighY)
    g.SetLineColor(h.GetLineColor())
    g.SetMarkerColor(h.GetMarkerColor())
    g.SetMarkerStyle(h.GetMarkerStyle())
    g.SetLineStyle(h.GetLineStyle())
    g.SetMarkerSize(h.GetMarkerSize())
    drawOption = option
    g.Draw(drawOption)

    return g

def doRMSandMean(histo, name, outDir):
    style.style1d()
    s = style.style1d()
    c=ROOT.TCanvas()
    Xnb=18
    Xr1=0.1
    Xr2=0.9
    dXbin=(Xr2-Xr1)/((Xnb));
    titleRMSVsGen_ptTop_full ="; M_{true};RMS"
    titleRespVsGen_ptTop_full ="; m(t#overline{t}) true;RMS"
    titleMeanVsGen_ptTop_full ="; m(t#overline{t}) true;Mean"
    h_RMSVsGen_=ROOT.TH1F()
    h_RMSVsGen_.SetDirectory(0)
    h_RMSVsGen_.SetBins(Xnb,Xr1,Xr2)
    h_RMSVsGen_.SetTitleOffset(2.0)
    h_RMSVsGen_.GetXaxis().SetTitleOffset(1.20)
    h_RMSVsGen_.GetYaxis().SetTitleOffset(1.30)
    h_RMSVsGen_.SetTitle(titleRMSVsGen_ptTop_full);
    h_RMSVsGen_.SetStats(0)
    h_RespVsGen_=ROOT.TH1F()
    h_RespVsGen_.SetDirectory(0)
    h_RespVsGen_.SetBins(Xnb,Xr1,Xr2)
    h_RespVsGen_.SetTitleOffset(2.0)
    h_RespVsGen_.GetXaxis().SetTitleOffset(1.20)
    h_RespVsGen_.GetYaxis().SetTitleOffset(1.30)
    h_RespVsGen_.SetTitle(titleRespVsGen_ptTop_full);
    h_RespVsGen_.SetStats(0)
    h_meanVsGen_=ROOT.TH1F()
    h_meanVsGen_.SetDirectory(0)
    h_meanVsGen_.SetBins(Xnb,Xr1,Xr2)
    h_meanVsGen_.SetTitleOffset(2.0)
    h_meanVsGen_.GetXaxis().SetTitleOffset(1.20)
    h_meanVsGen_.GetYaxis().SetTitleOffset(1.30)
    h_meanVsGen_.SetTitle(titleMeanVsGen_ptTop_full)
    h_meanVsGen_.SetStats(0)
    for i in range(Xnb):
        rms = (histo.ProjectionY("_py",histo.GetXaxis().FindFixBin(Xr1+i*dXbin) ,histo.GetXaxis().FindFixBin(Xr1+(i+1)*dXbin),"")).GetRMS()
        rms_err = (histo.ProjectionY("_py",histo.GetXaxis().FindFixBin(Xr1+i*dXbin) ,histo.GetXaxis().FindFixBin(Xr1+(i+1)*dXbin),"")).GetRMSError()
        mean = (histo.ProjectionY("_py",histo.GetXaxis().FindFixBin(Xr1+i*dXbin) ,histo.GetXaxis().FindFixBin(Xr1+(i+1)*dXbin),"")).GetMean()
        mean_err = (histo.ProjectionY("_py",histo.GetXaxis().FindFixBin(Xr1+i*dXbin) ,histo.GetXaxis().FindFixBin(Xr1+(i+1)*dXbin),"")).GetMeanError()
        respRMS = rms/(1.+mean)
        respRMS_err = np.sqrt((1./(1+mean) * rms_err)**2. + (rms/(1+mean)**2. *mean_err)**2.)
        h_RMSVsGen_.SetBinContent(i+1, rms)
        h_RMSVsGen_.SetBinError(i+1, rms_err)
        h_meanVsGen_.SetBinContent(i+1, mean)
        h_meanVsGen_.SetBinError(i+1, mean_err)

        h_RespVsGen_.SetBinContent(i+1, respRMS)
        h_RespVsGen_.SetBinError(i+1, respRMS_err)

    h_RMSVsGen_.SetStats(0)
    h_RMSVsGen_.Draw()
    c.SaveAs(outDir+name+"_rms.pdf")
    h_RespVsGen_.SetStats(0)
    h_RespVsGen_.Draw()
    c.SaveAs(outDir+name+"_respRMS.pdf")
    c.Clear()
    h_meanVsGen_.SetStats(0)
    h_meanVsGen_.Draw()
    c.SaveAs(outDir+name+"_mean.pdf")
    return h_RMSVsGen_, h_meanVsGen_, h_RespVsGen_

def doPSE(hResp, hGen, name, outDir):
    style.style1d()
    s = style.style1d()
    c = ROOT.TCanvas()
    n = hResp.GetNbinsX()
    hp = ROOT.TH1D("Purity", "", n, 0.0, n)
    hs = ROOT.TH1D("Stabilty", "", n, 0.0, n)
    he = ROOT.TH1D("Efficiency", "", n, 0.0, n)

    for b in range(1,n+1):
        GenInBinAndRecInBin = hResp.GetBinContent(b, b)
        RecInBin = hResp.Integral(b, b, 1, n)
        GenInBinAndRec = hResp.Integral(1, n, b, b)
        GenInBinAll = hGen.GetBinContent(b)
        if RecInBin>0.:
            hp.SetBinContent(b, GenInBinAndRecInBin / RecInBin)
        else:
            hp.SetBinContent(b, 0.)
        if GenInBinAndRec>0.:
            hs.SetBinContent(b, GenInBinAndRecInBin / GenInBinAndRec)
        else:
            hs.SetBinContent(b, 0.)
        if GenInBinAll>0.:
            he.SetBinContent(b, GenInBinAndRec / GenInBinAll)
        else:
            he.SetBinContent(b, 0.)

    leg = ROOT.TLegend(0.15, 0.67, 0.40, 0.85)
    leg.SetTextFont(62)
    hr = ROOT.TH2D("", "", 1, he.GetBinLowEdge(1), he.GetBinLowEdge(n + 1), 1, 0.0, 1.0)
    hr.GetXaxis().SetTitle("Bin")
    hr.SetStats(0)
    hr.Draw()
    hp.SetMarkerColor(4)
    hp.SetMarkerStyle(23)
    leg.AddEntry(hp, "Purity", "p")
    drawAsGraph(hp)
    hs.SetMarkerColor(2)
    hs.SetMarkerStyle(22)
    leg.AddEntry(hs, "Stability", "p")
    drawAsGraph(hs)
    he.SetMarkerColor(8)
    he.SetMarkerStyle(20)
    leg.AddEntry(he, "Efficiency", "p")
    drawAsGraph(he)

    leg.Draw()
    c.SaveAs(outDir+name+"_pse.pdf")
    return hp,hs,he

def uncertaintyBinomial(pass_, all_):
    return (1./all_)*np.sqrt(pass_ - pass_*pass_/all_)


def purityStabilityGraph(h2d, type):
    import ROOT
    nBins = h2d.GetNbinsX()

    graph = ROOT.TGraphErrors(nBins)

    # Calculating each point of graph for each diagonal bin
    for iBin in range(1,nBins+1):
        diag = h2d.GetBinContent(iBin, iBin)
        reco = h2d.Integral(iBin, iBin, 1, -1)+1e-30
        gen = h2d.Integral(1, -1, iBin, iBin)+1e-30

        value = diag/reco if (type == 0 ) else diag/gen
        error = uncertaintyBinomial(diag, reco) if (type == 0) else uncertaintyBinomial(diag, gen)

        bin = h2d.GetXaxis().GetBinCenter(iBin)
        binW = h2d.GetXaxis().GetBinWidth(iBin)

        graph.SetPoint(iBin-1, bin, value)
        graph.SetPointError(iBin-1, binW/2., error)

    return graph



def setGraphStyle(graph, line=-1, lineColor=-1, lineWidth=-1, marker=-1, markerColor=-1, markerSize=-1, fill=-1, fillColor=-1):
    import ROOT
    if(line != -1): graph.SetLineStyle(line)
    if(lineColor != -1): graph.SetLineColor(lineColor)
    if(lineWidth != -1): graph.SetLineWidth(lineWidth)

    if(fill != -1): graph.SetFillStyle(fill)
    if(fillColor != -1): graph.SetFillColor(fillColor)

    if(marker != -1): graph.SetMarkerStyle(marker)
    if(markerColor != -1): graph.SetMarkerColor(markerColor)
    if(markerSize != -1): graph.SetMarkerSize(markerSize)


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
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
    """
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
        return frozen_graph
