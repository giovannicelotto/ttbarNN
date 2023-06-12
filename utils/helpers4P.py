import glob
import sys
sys.path.append('/nfs/dust/cms/user/celottog/mttNN/utils')
import ROOT
import numpy as np
from progress.bar import IncrementalBar
import math
from helpers import rotation, append4vector
import sys
sys.path.append('/nfs/dust/cms/user/celottog/mlb_studies/')
from mlb_distribution import loadBinsCounts, checkProbability



	

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
	bins, counts = loadBinsCounts()	

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
	
	tree.SetBranchStatus("var_*",0)             # Ignoring all the var variables in the tree

	passStep3 = np.array([0], dtype=bool)
	tree.SetBranchAddress("passStep3", passStep3)


# Jets
	jetBTagged = np.array([0]*20, dtype='b')
	tree.SetBranchAddress("jets_btag", (jetBTagged))
	jetBTagScore = np.array([0]*20, dtype='f')
	tree.SetBranchAddress("jets_btag_score", (jetBTagScore))
	jetTopMatched = np.array([0]*20, dtype='b')
	tree.SetBranchAddress("jets_topMatched", (jetTopMatched))
	jetAntiTopMatched = np.array([0]*20, dtype='b')
	tree.SetBranchAddress("jets_antitopMatched", (jetAntiTopMatched))
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
# Weights and SF
	weight = np.array([0], dtype='d')
	tree.SetBranchAddress("weight", weight)
	leptonSF = np.array([0], dtype='f')
	tree.SetBranchAddress("leptonSF", leptonSF)
	triggerSF = np.array([0], dtype='f')
	tree.SetBranchAddress("triggerSF", triggerSF)
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

# kinReco and LooseReco
	hasKinRecoSolution = np.array([0], dtype=bool)
	tree.SetBranchAddress("hasKinRecoSolution", hasKinRecoSolution)
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

	hasLooseKinRecoSolution = np.array([0], dtype=bool)
	tree.SetBranchAddress("hasLooseKinRecoSolution", hasLooseKinRecoSolution)
	looseKinReco_ttbar_pt =  np.array([0], dtype='f')
	tree.SetBranchAddress("looseKinReco_ttbar_pt", looseKinReco_ttbar_pt)
	looseKinReco_ttbar_eta =  np.array([0], dtype='f')
	tree.SetBranchAddress("looseKinReco_ttbar_eta", looseKinReco_ttbar_eta)
	looseKinReco_ttbar_phi =  np.array([0], dtype='f')
	tree.SetBranchAddress("looseKinReco_ttbar_phi", looseKinReco_ttbar_phi)
	looseKinReco_ttbar_m =  np.array([0], dtype='f')
	tree.SetBranchAddress("looseKinReco_ttbar_m", looseKinReco_ttbar_m)


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
	kr_ttbar	= ROOT.TLorentzVector(0.,0.,0.,0.)
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
		kr_ttbar.SetPtEtaPhiM(0.,0.,0.,0.)
		dilepton.SetPtEtaPhiM(0.,0.,0.,0.)
		extraJet.SetPtEtaPhiM(0., 0., 0., 0.)
		zero.SetPtEtaPhiM(0., 0., 0., 0.)
		ht = 0
		totWeight=1.
		
		evFeatures=[]
		tree.GetEntry(i)

		if(i%bigNumber==0):
			bar.next()

		pass3= bool(passStep3[0])
		haskrs = bool(hasKinRecoSolution[0])
		haslkrs = bool(hasLooseKinRecoSolution[0])
		totWeight = weight[0]*btagSF[0]*leptonSF[0]*pileupSF[0]*prefiringWeight[0]*triggerSF[0]
		
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
		if ( (numJets>=2) & (passMETCut) & (passMLLCut) & (pass3) ):
			
			# jets identification
			jets   = []
			bjets  = []
			btag   = []
			bscore = []
			sortedJets = [] 		# used later to decide which jets are the ones coming from top decays
			for idx in range(numJets):
				jet4=ROOT.TLorentzVector(0.,0.,0.,0.)
				jet4.SetPtEtaPhiM(jetPt[idx],jetEta[idx],jetPhi[idx],jetM[idx])
				jets.append(jet4)
				ht = ht+jet4.Pt()
				btag.append(int(bool(jetBTagged[idx])))
				bscore.append(jetBTagScore[idx])

				

			nbjets = sum(btag)
			


			if nbjets < minbjets:  # out of my kin phase space
				continue
			else:
				pass
				#now we are in the phase space
			weights.append(totWeight)
			
			if (haslkrs):
					lKinRecoOut.append(lkr_ttbar.M())
					assert (math.isclose(lkr_ttbar.M(), looseKinReco_ttbar_m[0])), "Loose mass value from the tree not the same from the root computation"
			else:
				lKinRecoOut.append(-999)
			haskrs = True if ((haskrs) & (not math.isnan(kr_ttbar.M()))) else False
			if (haskrs):
				kinRecoOut.append(kr_ttbar.M())
			else:
				kinRecoOut.append(-999)
			

			met.SetPtEtaPhiM(met_pt[0],0.,met_phi[0],0.)
			toBeRotated = jets +[kr_top, kr_antitop, lkr_ttbar]
			rotation(toBeRotated, lep1, lep2, met)
			kr_ttbar = kr_top + kr_antitop
			dilepton = lep1 + lep2
			if (haskrs):
				assert math.isclose( kr_ttbar.M(), kinRecoOut[-1] , abs_tol = 1e-03), "Rotation. IsNan %.1f and %.1f" %( kr_ttbar.M(),  kinRecoOut[-1])

			nonbjets = [jets[j] for j in range(len(btag)) if btag[j] == 0]
			bjets    = [jets[j] for j in range(len(btag)) if btag[j] == 1]
			# new ordering of bscore and btag
			sortedBscore  = [bscore[j] for j in range(numJets) if btag[j] == 1] + [bscore[j] for j in range(numJets) if btag[j] == 0]
			sortedBtag    = [btag[j] for j in range(numJets) if btag[j] == 1] + [btag[j] for j in range(numJets) if btag[j] == 0]

			
			assert math.isclose(lep1.Phi(), 0, abs_tol = 1e-09), "Rotation didn't work"
			assert (lep1.Eta()>0), "Rotation didn't work"
			assert (lep2.Phi()>0), "Rotation didn't work"

			ttPt = kinReco_top_pt[0] + kinReco_antitop_pt[0]
			if (haslkrs & haskrs & (ttPt<13000) & (kr_ttbar.M()<13000)):
				evFeatures.append(lkr_ttbar.Pt())
				evFeatures.append(lkr_ttbar.Eta())
				evFeatures.append(lkr_ttbar.Phi())
				evFeatures.append(lkr_ttbar.M())
				append4vector(evFeatures, kr_ttbar)
				evFeatures.append(kr_top.Pt())		
				evFeatures.append(kr_top.Eta())
				evFeatures.append(kr_top.Phi())
				evFeatures.append(kr_antitop.Pt())
				evFeatures.append(kr_antitop.Eta())
				evFeatures.append(kr_antitop.Phi())
				evFeatures.append(kr_top.DeltaR(kr_antitop))
			else:
				for z in range(15):
					evFeatures.append(-4999)







			append4vector(evFeatures, dilepton)

			assert (numJets>=nbjets), "numJets < numbjets {} vs {}".format((numJets, nbjets))
			evFeatures.append(numJets)
			evFeatures.append(nbjets)
			
			
			leptons = [lep1, lep2]
			maxNotFound = False
			lookForPriorZero = False
			takeFirstJets = False
			if (nbjets>=2):
# If we have more than 2 jets: first find all the combinations using only b-jets
				#print("Try with %s bjets" %nbjets)
				probs = []
				pairs = []

				for idx in range(nbjets): 		# index of the jet1 among the bjets
					for jz in range(nbjets):	# index of the jet2 among the bjets
						if idx == jz: 			# means the same jets associated to both letpons. Set a negative unphysical value
							pairs.append([-1, -1])
							probs.append(-1)
							continue
						pairs.append(([(leptons[0] + bjets[idx]).M(), (leptons[1] + bjets[jz]).M()]))
						probs.append(checkProbability(bins, counts, pairs[-1]))
				probs = np.array(probs)
				pairs = np.array(pairs)
				#print("pairs", pairs)
				#print("probs", probs)
				if probs.max()>0:
					max_index = np.argmax(probs)
					first = int (max_index//nbjets)
					second = int(max_index%nbjets)
					#print("max_index \t%d\nfirst \t%d\nsecond \t%d" %(max_index, first, second))
					sortedJets.append(bjets[first])
					sortedJets.append(bjets[second])
					for jetInd in range(nbjets):
						if jetInd not in [first, second]:
							extraJet = extraJet + bjets[jetInd]
					for jetInd in range(len(nonbjets)):
						extraJet = extraJet + nonbjets[jetInd]
					assert first is not second
					bscore[0], bscore[1]	= sortedBscore[first], sortedBscore[second]
					btag[0], btag[1]	= sortedBtag[first], sortedBtag[second]

					assert sortedBtag[first]==1, "Cross check"
					assert sortedBtag[second]==1, "Cross check"
				elif probs.max()==0: 			# there is the possibility that using only bjets we don't find the real jets
					#print("Max not Found")
					maxNotFound = True		# e.g. btag efficiency (mistagging), bjets from gluon splitting
									# in that case we need to consider also the nonbjets
									# the same applies if we don't have enough bjets
			if ((nbjets==1) | maxNotFound):
				#print("Matrix with njets")
				probs = []
				pairs = []
				priority = []
				for idx in range(numJets):
					for jz in range(numJets):
						if idx == jz:
							pairs.append([-1, -1])
							probs.append(-1)
							priority.append(-1)
							continue
						pairs.append([(leptons[0] + jets[idx]).M(), (leptons[1] + jets[jz]).M()])
						probs.append(checkProbability(bins, counts, pairs[-1]))
						priority.append(btag[idx] + btag[jz])
				probs = np.array(probs)
				pairs = np.array(pairs)
				priority = np.array(priority)
				#print("pairs", pairs)
				#print("probs", probs)
				#print("priority", priority)
				# look for all the combinations with at least one bjet
				if (priority==1).sum()>0:				# if there are cases with priority = 1
					if probs[priority==1].max()>0:
						#print("max with pr=1")
						max_index = -1
						max_value = -1
						for w in range(len(priority)):
							if priority[w] == 1 and probs[w] > max_value:
								max_value = probs[w]
								max_index = w
						
						first = int (max_index//numJets)
						second = int (max_index%numJets)
						#print("max_index \t%d\nfirst \t%d\nsecond \t%d" %(max_index, first, second))
						sortedJets.append(jets[first])
						sortedJets.append(jets[second])
						for jetInd in range(numJets):
							if jetInd not in [first, second]:
								extraJet = extraJet + jets[jetInd]
						assert first is not second
						bscore[0], bscore[1] 	= bscore[first], bscore[second]
						btag[0], btag[1]			= btag[first], btag[second]
						#print("btag ", btag)
						assert btag[0]+ btag[1]  ==1, " instead %d"%(btag[first]+ btag[second])
					else:
						lookForPriorZero = True
				else:

					lookForPriorZero = True
				if (((priority==0).sum()>0) & lookForPriorZero):		# else look for cases with priority equal to 0
					if probs[priority==0].max()>0:
						#print("prior=0")
						#print("da verificare")
						max_index = -1
						max_value = -1
						for w in range(len(priority)):
							if priority[w] == 0 and probs[w] > max_value:
								max_value = probs[w]
								max_index = w
						
						first = int( max_index//numJets)
						second = int (max_index%numJets)
						sortedJets.append(jets[first])
						sortedJets.append(jets[second])
						for jetInd in range(numJets):
							if jetInd not in [first, second]:
								extraJet = extraJet + jets[jetInd]
						assert first is not second
						bscore[0] 	= bscore[first]
						bscore[1] 	= bscore[second]
						btag[0]		= btag[first]
						btag[1]		= btag[second]	
					else:
						takeFirstJets = True
						#assert False, "1. Hai usato tutti i jet e comunque tutte le combo fanno cagare. Sto evento fa schifo. Ti tocca usare il prodotto di mlb"
				elif (lookForPriorZero & ((priority==0).sum()==0)):
					takeFirstJets = True
				if (takeFirstJets):
					pairs = []
					# take the first 2 jets with highest pt and minimize mlb1*mlb2
					mlbDot1 = ((leptons[0] + jets[0]).M() * (leptons[1] + jets[1]).M())
					mlbDot2 = ((leptons[0] + jets[1]).M() * (leptons[1] + jets[0]).M())
					first = 0 if mlbDot1 < mlbDot2 else 1
					second = 1 if first==0 else 0
					sortedJets.append(jets[first])
					sortedJets.append(jets[second])
					for jetInd in range(numJets):
						if jetInd not in [first, second]:
							extraJet = extraJet + jets[jetInd]
					assert first is not second
					bscore[0] 	= bscore[first]
					bscore[1] 	= bscore[second]
					btag[0]		= btag[first]
					btag[1]		= btag[second]	


			append4vector(evFeatures, dilepton + sortedJets[0] + sortedJets[1])
			append4vector(evFeatures, dilepton + sortedJets[0] + sortedJets[1] + met)
			append4vector(evFeatures, extraJet)
			append4vector(evFeatures, dilepton + sortedJets[0] + sortedJets[1] + met + extraJet)
			append4vector(evFeatures, leptons[0] + sortedJets[0])
			append4vector(evFeatures, leptons[1] + sortedJets[1])

			append4vector(evFeatures, sortedJets[0])	
			evFeatures.append(btag[0])
			bscore = [bscore[j] if bscore[j]>0 else 0 for j in range(len(bscore))]
			evFeatures.append(bscore[0])	

			append4vector(evFeatures, sortedJets[1])
			evFeatures.append(btag[1])	
			evFeatures.append(bscore[1])		

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
    fileNames =  [i for i in fileNames if "run" in i][:nFiles]
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

def getXSections():
	xsections = {
        #diboson
            'wwtoall':              118.7,
            'wztoall':              47.13,
            'zztoall':              16.523,
        #other channels
            'ttbarbg_fromDilepton': 831.76*0.10706,
            'ttbarbg_fromLjets':    831.76*0.44113,
            'ttbarbg_fromHadronic': 831.76*0.45441,
        #ttbar+boson
            'ttbarWjetstoqq':       0.4062,
            'ttbarWjetstolnu':      0.2043,
            'ttbarZtollnunu':       0.2529,
            'ttbarZtoqq':           0.5297,
        # single TOP
            'singletop_s_leptonic': 10.32,
            'singletop_t':          136.02,
            'singletop_tw':         35.85*0.54559,
            'singleantitop_t':      80.95,
            'singleantitop_tw':     35.85*0.54559,
        # Wjets and Zjets
            'wtolnu':               61526.7,
            'dy1050':               22635.1,
            'dy50inf':              6225.4
        }	
	return xsections
