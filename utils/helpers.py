import glob
import ROOT
import numpy as np
from progress.bar import IncrementalBar
import tensorflow as tf

#definisci un append function che aggiunge il nome della variabile se non già presente
def getFeatureNames():
	feature_names =['lkr_ttbarpt', 'lkr_ttbareta', 'lkr_ttbarphi', 'lkr_ttbarM',
					'kr_ttbarpt', 'kr_ttbareta', 'kr_ttbarphi', 'kr_ttbarM',
					'dileptonpt', 'dileptoneta', 'dileptonphi', 'dileptonM',
					'llj1j2_pt', 'llj1j2_eta', 'llj1j2_phi', 'llj1j2M',
					'llj1j2METpt','llj1j2METeta', 'llj1j2METphi', 'llj1j2MET_M', 'njets', 'nbjets',
					'extraJetspt', 'extraJetseta', 'extraJetsphi', 'extraJetsM', 'mlb_min',
					'jet1pt', 'jet1eta', 'jet1phi', 'jet1m', 'jet1btag',
					'jet2pt', 'jet2eta', 'jet2phi', 'jet2m', 'jet2btag',
					'lep1pt', 'lep1eta', 'lep1phi', 'lep1m',
					'lep2pt', 'lep2eta', 'lep2phi', 'lep2m',
					'metpt', 'metphi', 'ht',
					'dR_lepton1_jet1', 'dR_lepton1_jet2', 'dR_lepton2_jet1', 'dR_lepton2_jet2', 'dR_jet1_jet2','channelID'
	]
	return feature_names
def append4vector(evFeatures, a):
	evFeatures.append(a.Pt())
	evFeatures.append(a.Eta())
	evFeatures.append(a.Phi())
	evFeatures.append(a.M())



def NormalizeBinWidth1d(h):
	for i in range(1, h.GetNbinsX()+1):
		if h.GetBinContent(i) != 0:
			h.SetBinContent(i, h.GetBinContent(i)*1./h.GetBinWidth(i))
	return h
			
def loadMDataFlat(path, filename, treeName, maxJets, maxEvents, withBTag = False, pTEtaPhiMode=False):
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
	maxJets=maxJets	# max number of jets

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
	# there must be met_eta
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

	
# If maxevents is defined only a fraction of events
	if maxEvents!=0:
		maxEntries = maxEvents
	else:
		maxEntries = tree.GetEntries()
	    
	bigNumber = 20000
	maxForBar = int(maxEntries/bigNumber)

# Tlorentz vectors
	lep1=ROOT.TLorentzVector(0.,0.,0.,0.)
	lep2=ROOT.TLorentzVector(0.,0.,0.,0.)
#GC	jet4=ROOT.TLorentzVector(0.,0.,0.,0.)
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

	bar = IncrementalBar('Processing', max=maxForBar, suffix='%(percent).1f%% - %(eta)ds')
	print("\nLooping in the trees with ", tree.GetEntries()," entries. Only ",maxEntries, " will be used")  # GC
	for i in range(maxEntries):
		lep1.SetPtEtaPhiM(0.,0.,0.,0.)
		lep2.SetPtEtaPhiM(0.,0.,0.,0.)
#GC		jet4.SetPtEtaPhiM(0.,0.,0.,0.)
		met.SetPtEtaPhiM(0.,0.,0.,0.)
		kr_top.SetPtEtaPhiM(0.,0.,0.,0.)
		kr_antitop.SetPtEtaPhiM(0.,0.,0.,0.)
		lkr_ttbar.SetPtEtaPhiM(0.,0.,0.,0.)
		kr_nonbjet.SetPtEtaPhiM(0.,0.,0.,0.)
		kr_ttbar.SetPtEtaPhiM(0.,0.,0.,0.)
		lkr_nonbjet.SetPtEtaPhiM(0.,0.,0.,0.)
		bjettemp.SetPtEtaPhiM(0.,0.,0.,0.)
		dilepton.SetPtEtaPhiM(0.,0.,0.,0.)

		ht = 0
		# Generated vectors
		top.SetPtEtaPhiM(0.,0.,0.,0.)
		antitop.SetPtEtaPhiM(0.,0.,0.,0.)
		ttbar.SetPtEtaPhiM(0.,0.,0.,0.)
		
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
			numJets = njets[0]
		else:
			lep1.SetPtEtaPhiM(0,0,0,0)
			lep2.SetPtEtaPhiM(0,0,0,0)
			numJets = njets[0]
			
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
		if isEMuChannel:
			passMETCut = True
		elif (met_pt[0]>40.):
			passMETCut = True
		else:
			passMETCut = False
			
		top.SetPtEtaPhiM(gen_top_pt[0],gen_top_eta[0],gen_top_phi[0],gen_top_m[0])
		antitop.SetPtEtaPhiM(gen_antitop_pt[0],gen_antitop_eta[0],gen_antitop_phi[0],gen_antitop_m[0])
		ttbar = top+antitop
		
		# Add here requirements on maxjets
		# conditions fro building an input events (training or testing)
		if (pass3 and (numJets>=2) and passMETCut and (dilepton.M()>20.) and passMLLCut and (ttbar.M()>340.) & (kinReco_antitop_m<2000) & (kinReco_antitop_m>1) & (looseKinReco_ttbar_m > 1) & (looseKinReco_ttbar_m < 2000)):
			
			# jets identification
			jets = []
			bjets = []
			btag = []
			for idx in range(min(numJets,maxJets)):
				jet4=ROOT.TLorentzVector(0.,0.,0.,0.)
				jet4.SetPtEtaPhiM(jetPt[idx],jetEta[idx],jetPhi[idx],jetM[idx])
				jets.append(jet4) 
				bTagged=int(bool(jetBTagged[idx]))
				btag.append(bTagged)
				ht = ht+jet4.Pt()
			#	if withBTag:
			#		evFeatures.append(bTagged)								# 5 features with the info on btag
					
				if(bTagged):
					bjets.append(jet4)

			nbjets = len(bjets)
			nonbjets = [jets[j] for j in range(len(btag)) if btag[j] == 0]
			sortedJets = bjets + nonbjets
			weights.append(totWeight)
			kr_top.SetPtEtaPhiM(kinReco_top_pt[0],kinReco_top_eta[0],kinReco_top_phi[0],kinReco_top_m[0])
			kr_antitop.SetPtEtaPhiM(kinReco_antitop_pt[0],kinReco_antitop_eta[0],kinReco_antitop_phi[0],kinReco_antitop_m[0])
			kr_ttbar= kr_antitop + kr_top
			lkr_ttbar.SetPtEtaPhiM(looseKinReco_ttbar_pt[0],looseKinReco_ttbar_eta[0],looseKinReco_ttbar_phi[0],looseKinReco_ttbar_m[0])
				

			if (haslkrs and lkr_ttbar.M()>0.):
				append4vector(evFeatures,lkr_ttbar)
				lKinRecoOut.append(lkr_ttbar.M())   # 2	Output of the LooseKinReco
			else:
				append4vector(evFeatures,zero)
				lKinRecoOut.append(0.)
			if (haskrs and kr_ttbar.M()>0.):
				append4vector(evFeatures, kr_ttbar)
				kinRecoOut.append(kr_ttbar.M())     # 
			else:
				append4vector(evFeatures, zero)
				kinRecoOut.append(0.)

			met.SetPtEtaPhiM(met_pt[0],0.,met_phi[0],0.)
			#allLepton = dilepton + met
			
			append4vector(evFeatures, dilepton)

			if (len(bjets) >= 2):
				append4vector(evFeatures, dilepton+bjets[0]+bjets[1])
				append4vector(evFeatures, dilepton+bjets[0]+bjets[1]+met)
				
			elif (len(bjets) == 1):
				
				append4vector(evFeatures, dilepton+ bjets[0]+ nonbjets[0])
				append4vector(evFeatures,dilepton+bjets[0]+ nonbjets[0]+met)

			elif (len(bjets) == 0):
				append4vector(evFeatures, dilepton+ jets[0] + jets[1])
				append4vector(evFeatures, dilepton+ jets[0] + jets[1] + met)
				
			evFeatures.append(numJets)									 		# 6
			evFeatures.append(nbjets)
			
			extraJet = ROOT.TLorentzVector(0.,0.,0.,0.)
			if(numJets>2):		# if therea are extrajet
				if (len(bjets) >= 3):
					for jetTemp in bjets[2:]:
						extraJet = extraJet + jetTemp	# only from the third bjet
					for jetTemp in nonbjets:			# plus oall the non bjet
						extraJet = extraJet + jetTemp	
					append4vector(evFeatures, extraJet)				
				elif (len(bjets) <= 2):
					for jetTemp in nonbjets[2-len(bjets):]:
						extraJet = extraJet + jetTemp
					append4vector(evFeatures, extraJet)

			else:
				append4vector(evFeatures, zero)

			mlb_min = 0
			for i in range(nbjets):
				bjettemp.SetPtEtaPhiM(bjets[i].Pt(),bjets[i].Eta(),bjets[i].Phi(),bjets[i].M())
				comb1 = (lep1+bjettemp).M()	
				comb2 = (lep2+bjettemp).M()
				comb = comb1 if comb1<comb2 else comb2
				if (i==0):
					mlb_min=comb
				elif(comb < mlb_min):
					mlb_min = comb
			evFeatures.append(mlb_min)											# 9

			append4vector(evFeatures, sortedJets[0])										# 12
			evFeatures.append(btag[0])											# 13

			append4vector(evFeatures, sortedJets[1])
			evFeatures.append(btag[1])											# 18

			append4vector(evFeatures, lep1)
			append4vector(evFeatures, lep2)
			evFeatures.append(met.Pt())											# 27
			evFeatures.append(met.Phi())
			evFeatures.append(ht)
			
			dR_lepton1_jet1 = lep1.DeltaR(sortedJets[0])
			dR_lepton1_jet2 = lep1.DeltaR(sortedJets[1])
			dR_lepton2_jet1 = lep2.DeltaR(sortedJets[0])
			dR_lepton2_jet2 = lep2.DeltaR(sortedJets[1])
			# DeltaR with additional jets ?
			dR_jet1_jet2 = sortedJets[0].DeltaR(sortedJets[1])	
			
			evFeatures.append(dR_lepton1_jet1)
			evFeatures.append(dR_lepton1_jet2)
			evFeatures.append(dR_lepton2_jet1)
			evFeatures.append(dR_lepton2_jet2)
			evFeatures.append(dR_jet1_jet2)
			evFeatures.append(channelID)

			
			eventIn.append(evFeatures)
			eventOut.append([ttbar.M()])
		tree.SetBranchStatus("*",1)

	return eventIn, eventOut, weights, lKinRecoOut, kinRecoOut


def loadRegressionData(path, treeName, maxJets=10, maxEvents=0, withBTag = False, pTEtaPhiMode=False):
    '''
    Pass the list of input output features, weights, and output of the invariant mass from LKR and FKR everything in list of list
    '''
    print("loadRegressionData called \n\n")
    print("Searching root files in ", path)
    fileNames = glob.glob(path+'/*.root')		# List of files.root in the directory
    print (len(fileNames), " files to be used\n")
    eventIn, eventOut, weights,lkrM,krM = [],[],[],[],[]	# Empty lists for input, output, weights, outputs of analytical results
    n=0
    for filename in fileNames:					# Looping over the file names
        n = n+1
        print ("\n",filename,"("+str(n)+"/"+str(len(fileNames))+")")
        i,o,w,l,k = loadMDataFlat(filename, filename, treeName, maxJets=maxJets, maxEvents=maxEvents, withBTag = withBTag, pTEtaPhiMode = pTEtaPhiMode)
        eventIn+=i
        eventOut+=o
        weights+=w
        lkrM+=l
        krM+=k
    return eventIn, eventOut, weights, lkrM, krM


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