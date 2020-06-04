import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import ROOT

import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_WARNINGS'] = 'off'
from keras import backend as K
import tensorflow as tf
import keras.backend as K
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
K.tensorflow_backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))))

import uproot
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


def doTraining(xData, yData, model, outputFolder, learning_rate=1e-4, batch_size=3500, epochs=50, splitFactor=0.33, index=0):

	X_train, X_test, y_train, y_test = train_test_split(xData, yData, test_size=splitFactor)

	optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

	metrics = [
	    tf.keras.metrics.mean_squared_error,
	    tf.keras.metrics.mean_absolute_error
	]



	model.compile(optimizer=optimizer, loss=keras.losses.MeanSquaredError(),metrics=metrics)
	# model.compile(optimizer=optimizer, loss=keras.losses.mean_squared_error,metrics=metrics)


	callbacks=[]

	# reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
	# 	                              patience=5, min_lr=learning_rate/10.)
	reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3)
	earlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=4)
	if (('pretrain' in outputFolder) or ('training' in outputFolder)):
		if not os.path.exists(outputFolder+"/model/checkpoints/"):
			os.makedirs(outputFolder+"/model/checkpoints/")
		checkpoint_path = outputFolder+"/model/checkpoints/cp.ckpt"
		# checkpoint_dir = os.path.dirname(checkpoint_path)

		# Create checkpoint callback
		cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
														 save_weights_only=True,
														 verbose=0)

	callbacks.append(reduce_lr)
	callbacks.append(earlyStop)
	if (('pretrain' in outputFolder) or ('training' in outputFolder)):
		callbacks.append(cp_callback)

	print (y_train.shape)

	fit = model.fit(
	        X_train,
	        y_train,
	        validation_data=(X_test, y_test),
	        batch_size=batch_size,
	        epochs=epochs,
	        # shuffle=True,
	        shuffle=False,
	        callbacks=callbacks,
	verbose=1)


	if (('pretrain' in outputFolder) or ('final' in outputFolder)):
		addition = ''
		folder_resultArrays = outputFolder+"/values/"
		folder_resultPlots = outputFolder+"/plots/"
	else:
		addition = str(index)
		folder_resultArrays = outputFolder+"/values/"+addition+"/"
		folder_resultPlots = outputFolder+"/plots/"+addition+"/"
	if not os.path.exists(folder_resultArrays):
		os.makedirs(folder_resultArrays)
	if not os.path.exists(folder_resultPlots):
		os.makedirs(folder_resultPlots)

	np.save(folder_resultArrays + 'loss.npy', fit.history["loss"])
	np.save(folder_resultArrays + 'val_loss.npy', fit.history["val_loss"])
	np.save(folder_resultArrays + 'mse.npy', fit.history["mean_squared_error"])
	np.save(folder_resultArrays + 'val_mse.npy', fit.history["val_mean_squared_error"])
	np.save(folder_resultArrays + 'mae.npy', fit.history["mean_absolute_error"])
	np.save(folder_resultArrays + 'val_mae.npy', fit.history["val_mean_absolute_error"])


	# plot loss
	f = plt.figure()
	plt.plot(fit.history["loss"])
	plt.plot(fit.history["val_loss"])
	plt.xlabel("epochs")
	plt.ylabel("loss")
	plt.legend(["training loss", "validation loss"], loc="best")
	plt.grid()
	f.savefig(folder_resultPlots +"loss.pdf")

	# plot MSE
	f = plt.figure()
	plt.plot(fit.history["mean_squared_error"])
	plt.plot(fit.history["val_mean_squared_error"])
	plt.xlabel("epochs")
	plt.ylabel("MSE")
	plt.legend(["training MSE", "validation MSE"], loc="best")
	plt.grid()
	f.savefig(folder_resultPlots +"mse.pdf")

	# plot MAE
	f = plt.figure()
	plt.plot(fit.history["mean_absolute_error"])
	plt.plot(fit.history["val_mean_absolute_error"])
	plt.xlabel("epochs")
	plt.ylabel("MAE")
	plt.legend(["training MAE", "validation MAE"], loc="best")
	plt.grid()
	f.savefig(folder_resultPlots +"mae.pdf")

	return  np.min(fit.history["loss"]), \
		np.min(fit.history["val_loss"]), \
		np.min(fit.history["mean_squared_error"]), \
		np.min(fit.history["val_mean_squared_error"]), \
		np.min(fit.history["mean_absolute_error"]), \
		np.min(fit.history["val_mean_absolute_error"])

def doTrainingNew(xData1,xData2,xData3, yData, model, outputFolder, learning_rate=1e-4, batch_size=3500, epochs=50):



	optimizer = tf.keras.optimizers.Adam(lr=learning_rate)


	metrics = [
	    tf.keras.metrics.mean_squared_error,
	    tf.keras.metrics.mean_absolute_error
	]


	model.compile(optimizer=optimizer, loss=tf.keras.losses.mean_squared_error,metrics=metrics)


	callbacks=[]

	reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
		                              patience=5, min_lr=learning_rate/10.)
	earlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3)


	callbacks.append(reduce_lr)
	callbacks.append(earlyStop)


	fit = model.fit(
	        [xData1,xData2,xData3],
	        yData,
	        validation_split=0.33,
	        batch_size=batch_size,
	        epochs=epochs,
	        # shuffle=True,
	        shuffle=False,
	        callbacks=callbacks,
	verbose=1)


	if (('pretrain' in outputFolder) or ('final' in outputFolder)):
		addition = ''
		folder_resultArrays = outputFolder+"/values/"
		folder_resultPlots = outputFolder+"/plots/"
	else:
		addition = str(index)
		folder_resultArrays = outputFolder+"/values/"+addition+"/"
		folder_resultPlots = outputFolder+"/plots/"+addition+"/"
	if not os.path.exists(folder_resultArrays):
		os.makedirs(folder_resultArrays)
	if not os.path.exists(folder_resultPlots):
		os.makedirs(folder_resultPlots)


def loadRhoData(path, treeName, nJets=20, maxEvents=0, normX=False, normY=False, normValue=170.):
    eventInJet=[]
    eventInOther=[]
    eventOut=[]
    eventNBJets=[]
    eventNJets=[]
    maxJets=nJets


    f = ROOT.TFile.Open(path)
    tree = f.Get(treeName)

    print("Read TTree: {} (Entries: {})".format(treeName, tree.GetEntries()))

    # passStep3 = np.array([0], dtype=np.float32)
    passStep3 = np.array([0], dtype=bool)
    tree.SetBranchAddress("passStep3", passStep3)


    jetPt = ROOT.std.vector('float')()
    tree.SetBranchAddress("jets_pt", ROOT.AddressOf(jetPt))

    jetEta = ROOT.std.vector('float')()
    tree.SetBranchAddress("jets_eta", ROOT.AddressOf(jetEta))

    jetPhi = ROOT.std.vector('float')()
    tree.SetBranchAddress("jets_phi", ROOT.AddressOf(jetPhi))

    jetM = ROOT.std.vector('float')()
    tree.SetBranchAddress("jets_m", ROOT.AddressOf(jetM))

    jetBTagged = ROOT.std.vector('bool')()
    tree.SetBranchAddress("jets_btag", ROOT.AddressOf(jetBTagged))

    jetTopMatched = ROOT.std.vector('bool')()
    tree.SetBranchAddress("jets_topMatched", ROOT.AddressOf(jetTopMatched))

    genjetPt = ROOT.std.vector('float')()
    tree.SetBranchAddress("gen_additional_jets_pt", ROOT.AddressOf(genjetPt))

    genjetEta = ROOT.std.vector('float')()
    tree.SetBranchAddress("gen_additional_jets_eta", ROOT.AddressOf(genjetEta))

    genjetPhi = ROOT.std.vector('float')()
    tree.SetBranchAddress("gen_additional_jets_phi", ROOT.AddressOf(genjetPhi))

    genjetM = ROOT.std.vector('float')()
    tree.SetBranchAddress("gen_additional_jets_m", ROOT.AddressOf(genjetM))

    genrho = np.array([0], dtype='f')
    tree.SetBranchAddress("gen_rho", genrho)



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

    ##Loop over event tree
    for i in range(tree.GetEntries()):
        jets_info=[]
        jets_matchInfo=[]
        other_info=[]
        tree.GetEntry(i)

        if(i%100000==0):
            print("Reading event: {}".format(i))

        if(maxEvents!=0):
        	if(i%maxEvents==0 and i!=0):
        		break

        pass3 = bool(passStep3)


        if(pass3==True):
            numJets = jetPt.size()
            countB=0.
            numGenJets = genjetPt.size()
            if(numJets>1):
                if(numGenJets>2 and genrho>0.):

                    for idx in range(maxJets):
                        jet=[]
                        if(idx<numJets):
                            jet4=ROOT.TLorentzVector(0.,0.,0.,0.)
                            jet4.SetPtEtaPhiM(jetPt[idx],jetEta[idx],jetPhi[idx],jetM[idx])
                            jet.append(jet4.Px())
                            jet.append(jet4.Py())
                            jet.append(jet4.Pz())
                            jet.append(jet4.E())

                            bTagged=int(bool(jetBTagged[idx]))
                            jet.append(bTagged)
                            # jets_matchInfo.append(int(bool(jetTopMatched[idx])))
                            countB=countB+bTagged
                        else:
                            jet.append(0.)
                            jet.append(0.)
                            jet.append(0.)
                            jet.append(0.)
                            jet.append(0.)
                            jets_matchInfo.append(0.)
                        jets_info.append(jet)
                    eventNJets.append(numJets)
                    eventNBJets.append(countB)

                    lep1=ROOT.TLorentzVector(0.,0.,0.,0.)
                    lep1.SetPtEtaPhiM(lepton1_pt,lepton1_eta,lepton1_phi,lepton1_m)
                    lep2=ROOT.TLorentzVector(0.,0.,0.,0.)
                    lep2.SetPtEtaPhiM(lepton2_pt,lepton2_eta,lepton2_phi,lepton2_m)
                    met=ROOT.TLorentzVector(0.,0.,0.,0.)
                    met.SetPtEtaPhiM(met_pt,0.,met_phi,0.)

                    l1=[]
                    l2=[]
                    miss=[]
                    l1.append(lep1.Px())
                    l1.append(lep1.Py())
                    l1.append(lep1.Pz())
                    l1.append(lep1.E())

                    l2.append(lep2.Px())
                    l2.append(lep2.Py())
                    l2.append(lep2.Pz())
                    l2.append(lep2.E())

                    miss.append(met.Px())
                    miss.append(met.Py())
                    miss.append(met.Pz())
                    miss.append(met.E())

                    other_info.append(l1)
                    other_info.append(l2)
                    other_info.append(miss)

                    eventInJet.append(jets_info)
                    eventInOther.append(other_info)
                    eventOut.append(genrho[0])
    return eventInJet,eventInOther,eventOut,eventNJets,eventNBJets

def loadRhoDataFlat(path, treeName, nJets=3, maxEvents=0, withBTag = False, withCharge = False):
    eventInJet=[]
    eventOut=[]
    weights=[]
    maxJets=nJets

    f = ROOT.TFile.Open(path)
    tree = f.Get(treeName)

    print("Read TTree: {} (Entries: {})".format(treeName, tree.GetEntries()))

    passStep3 = np.array([0], dtype=bool)
    tree.SetBranchAddress("passStep3", passStep3)

    hasKinRecoSolution = np.array([0], dtype=bool)
    tree.SetBranchAddress("hasKinRecoSolution", hasKinRecoSolution)

    hasLooseKinRecoSolution = np.array([0], dtype=bool)
    tree.SetBranchAddress("hasLooseKinRecoSolution", hasLooseKinRecoSolution)

    jetPt = ROOT.std.vector('float')()
    tree.SetBranchAddress("jets_pt", ROOT.AddressOf(jetPt))
    jetEta = ROOT.std.vector('float')()
    tree.SetBranchAddress("jets_eta", ROOT.AddressOf(jetEta))
    jetPhi = ROOT.std.vector('float')()
    tree.SetBranchAddress("jets_phi", ROOT.AddressOf(jetPhi))
    jetM = ROOT.std.vector('float')()
    tree.SetBranchAddress("jets_m", ROOT.AddressOf(jetM))

    bjetPt = ROOT.std.vector('float')()
    tree.SetBranchAddress("bjets_pt", ROOT.AddressOf(bjetPt))
    bjetEta = ROOT.std.vector('float')()
    tree.SetBranchAddress("bjets_eta", ROOT.AddressOf(bjetEta))
    bjetPhi = ROOT.std.vector('float')()
    tree.SetBranchAddress("bjets_phi", ROOT.AddressOf(bjetPhi))
    bjetM = ROOT.std.vector('float')()
    tree.SetBranchAddress("bjets_m", ROOT.AddressOf(bjetM))

    lkr_nonbjetPt = ROOT.std.vector('float')()
    tree.SetBranchAddress("looseKinReco_nonbjets_pt", ROOT.AddressOf(lkr_nonbjetPt))
    lkr_nonbjetEta = ROOT.std.vector('float')()
    tree.SetBranchAddress("looseKinReco_nonbjets_eta", ROOT.AddressOf(lkr_nonbjetEta))
    lkr_nonbjetPhi = ROOT.std.vector('float')()
    tree.SetBranchAddress("looseKinReco_nonbjets_phi", ROOT.AddressOf(lkr_nonbjetPhi))
    lkr_nonbjetM = ROOT.std.vector('float')()
    tree.SetBranchAddress("looseKinReco_nonbjets_m", ROOT.AddressOf(lkr_nonbjetM))

    kr_nonbjetPt = ROOT.std.vector('float')()
    tree.SetBranchAddress("kinReco_nonbjets_pt", ROOT.AddressOf(kr_nonbjetPt))
    kr_nonbjetEta = ROOT.std.vector('float')()
    tree.SetBranchAddress("kinReco_nonbjets_eta", ROOT.AddressOf(kr_nonbjetEta))
    kr_nonbjetPhi = ROOT.std.vector('float')()
    tree.SetBranchAddress("kinReco_nonbjets_phi", ROOT.AddressOf(kr_nonbjetPhi))
    kr_nonbjetM = ROOT.std.vector('float')()
    tree.SetBranchAddress("kinReco_nonbjets_m", ROOT.AddressOf(kr_nonbjetM))

    # jetBTagged = ROOT.std.vector('bool')()
    # tree.SetBranchAddress("jets_btag", ROOT.AddressOf(jetBTagged))

    # jetCharge = ROOT.std.vector('float')()
    # tree.SetBranchAddress("jets_charge", ROOT.AddressOf(jetCharge))

    # jetTopMatched = ROOT.std.vector('bool')()
    # tree.SetBranchAddress("jets_topMatched", ROOT.AddressOf(jetTopMatched))

    genjetPt = np.array([0], dtype='f')
    tree.SetBranchAddress("gen_additional_jet_pt", genjetPt)

    genjetEta = np.array([0], dtype='f')
    tree.SetBranchAddress("gen_additional_jet_eta", genjetEta)

    genjetPhi = np.array([0], dtype='f')
    tree.SetBranchAddress("gen_additional_jet_phi", genjetPhi)

    genjetM = np.array([0], dtype='f')
    tree.SetBranchAddress("gen_additional_jet_m", genjetM)

    genjetWithNuPt = np.array([0], dtype='f')
    tree.SetBranchAddress("gen_additional_jet_withNu_pt", genjetWithNuPt)

    genjetWithNuEta = np.array([0], dtype='f')
    tree.SetBranchAddress("gen_additional_jet_withNu_eta", genjetWithNuEta)

    genjetWithNuPhi = np.array([0], dtype='f')
    tree.SetBranchAddress("gen_additional_jet_withNu_phi", genjetWithNuPhi)

    genjetWithNuM = np.array([0], dtype='f')
    tree.SetBranchAddress("gen_additional_jet_withNu_m", genjetWithNuM)

    # genjetPt = ROOT.std.vector('float')()
    # tree.SetBranchAddress("gen_additional_jets_pt", ROOT.AddressOf(genjetPt))
    #
    # genjetEta = ROOT.std.vector('float')()
    # tree.SetBranchAddress("gen_additional_jets_eta", ROOT.AddressOf(genjetEta))
    #
    # genjetPhi = ROOT.std.vector('float')()
    # tree.SetBranchAddress("gen_additional_jets_phi", ROOT.AddressOf(genjetPhi))
    #
    # genjetM = ROOT.std.vector('float')()
    # tree.SetBranchAddress("gen_additional_jets_m", ROOT.AddressOf(genjetM))

    genrho = np.array([0], dtype='f')
    tree.SetBranchAddress("gen_rho", genrho)
    genrhoWithNu = np.array([0], dtype='f')
    tree.SetBranchAddress("gen_rhoWithNu", genrhoWithNu)

    weight = np.array([0], dtype='d')
    tree.SetBranchAddress("weight", weight)
    leptonSF = np.array([0], dtype='d')
    tree.SetBranchAddress("leptonSF", leptonSF)
    btagSF = np.array([0], dtype='d')
    tree.SetBranchAddress("btagSF", btagSF)
    pileupSF = np.array([0], dtype='d')
    tree.SetBranchAddress("pileupSF", pileupSF)
    prefiringWeight = np.array([0], dtype='d')
    tree.SetBranchAddress("l1PrefiringWeight", prefiringWeight)



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
    met_significance =  np.array([0], dtype='f')
    tree.SetBranchAddress("met_significance", met_significance)

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
    looseKinReco_ttbar_m =  np.array([0], dtype='f')
    # tree.SetBranchAddress("looseKinReco_ttbar_m", looseKinReco_nonbjets)

    from progress.bar import IncrementalBar

    if maxEvents!=0:
        maxEntries = maxEvents
    else:
        maxEntries = tree.GetEntries()

    maxForBar = int(maxEntries/200000)

    bar = IncrementalBar('Processing', max=maxForBar, suffix='%(percent).1f%% - %(eta)ds')
    # bar = IncrementalBar('Processing', max=maxForBar, suffix='%(percent).1f%%')

    for i in range(tree.GetEntries()):

        jets_info=[]
        jets_matchInfo=[]
        other_info=[]
        tree.GetEntry(i)
        totWeight=1.

        if(i%200000==0):
            bar.next()
        # if(i%200000==0):
            # if maxEvents == 0:
            #     percentage = np.round(float(i)/float(tree.GetEntries())*100.,2)
            # else:
            #     percentage = np.round(float(i)/float(maxEvents)*100.,2)
            # percentage = np.round(float(i)/float(maxEvents)*100.,2)
            # print("Reading event: {} | {} %".format(i, percentage))

        if(maxEvents!=0):
        	if(i%maxEvents==0 and i!=0):
        		break

        pass3= bool(passStep3)
        haskrs = bool(hasKinRecoSolution)
        haslkrs = bool(hasLooseKinRecoSolution)
        # totWeight=weight[0]*btagSF[0]*leptonSF[0]*pileupSF[0]*prefiringWeight[0]
        totWeight=btagSF[0]*leptonSF[0]*pileupSF[0]*prefiringWeight[0]



        if(pass3==True):
            numJets = jetPt.size()
            countB=0.
            # numGenJets = genjetPt.size()
            # if(numJets>1):
            if(numJets>2):
                # if(numGenJets>2 and genrho>0.):
                # if(numGenJets>0 and genrho>0.):
                if(genrhoWithNu>0. and genjetWithNuPt>20. and abs(genjetWithNuEta)<2.4):
                # if(genrho>0. and genjetPt>20. and abs(genjetEta)<2.4):

                    jets = []
                    for idx in range(maxJets):
                        if(idx<numJets):
                            jet4=ROOT.TLorentzVector(0.,0.,0.,0.)
                            jet4.SetPtEtaPhiM(jetPt[idx],jetEta[idx],jetPhi[idx],jetM[idx])
                            jets_info.append(jet4.Px())
                            jets_info.append(jet4.Py())
                            jets_info.append(jet4.Pz())
                            jets_info.append(jet4.E())
                            jets.append(jet4)
                            # bTagged=int(bool(jetBTagged[idx]))
                            # if withBTag:
                            #     jets_info.append(bTagged)
                            # if withCharge:
                            #     jets_info.append(jetCharge[idx])
                        else:
                            jets_info.append(0.)
                            jets_info.append(0.)
                            jets_info.append(0.)
                            jets_info.append(0.)
                            # if withBTag:
                            #     jets_info.append(0.)
                            # if withCharge:
                            #     jets_info.append(0.)
                    weights.append(totWeight)

                    lep1=ROOT.TLorentzVector(0.,0.,0.,0.)
                    lep1.SetPtEtaPhiM(lepton1_pt,lepton1_eta,lepton1_phi,lepton1_m)
                    lep2=ROOT.TLorentzVector(0.,0.,0.,0.)
                    lep2.SetPtEtaPhiM(lepton2_pt,lepton2_eta,lepton2_phi,lepton2_m)
                    met=ROOT.TLorentzVector(0.,0.,0.,0.)
                    met.SetPtEtaPhiM(met_pt,0.,met_phi,0.)

                    kr_top=ROOT.TLorentzVector(0.,0.,0.,0.)
                    kr_top.SetPtEtaPhiM(kinReco_top_pt,kinReco_top_eta,kinReco_top_phi,kinReco_top_m)
                    kr_antitop=ROOT.TLorentzVector(0.,0.,0.,0.)
                    kr_antitop.SetPtEtaPhiM(kinReco_antitop_pt,kinReco_antitop_eta,kinReco_antitop_phi,kinReco_antitop_m)
                    kr_ttbar= kr_antitop + kr_top
                    lkr_ttbar=ROOT.TLorentzVector(0.,0.,0.,0.)
                    lkr_ttbar.SetPtEtaPhiM(looseKinReco_ttbar_pt,looseKinReco_ttbar_eta,looseKinReco_ttbar_phi,looseKinReco_ttbar_m)

                    lkr_rho=0.
                    kr_rho=0.
                    kr_foundAddjet = False
                    lkr_foundAddjet = False
                    for idx in range(lkr_nonbjetPt.size()):
                        if (lkr_foundAddjet) == False:
                            lkr_jettemp=ROOT.TLorentzVector(0.,0.,0.,0.)
                            lkr_jettemp.SetPtEtaPhiM(lkr_nonbjetPt[idx],lkr_nonbjetEta[idx],lkr_nonbjetPhi[idx],lkr_nonbjetM[idx])
                            # if (lkr_jettemp.Pt()>50. and abs(lkr_jettemp.Eta())<2.4):
                            if (lkr_jettemp.Pt()>0. and abs(lkr_jettemp.Eta())<9.):
                                lkr_rho = 340./(lkr_jettemp+lkr_ttbar).M()
                    for idx in range(kr_nonbjetPt.size()):
                        if (kr_foundAddjet) == False:
                            kr_jettemp=ROOT.TLorentzVector(0.,0.,0.,0.)
                            kr_jettemp.SetPtEtaPhiM(kr_nonbjetPt[idx],kr_nonbjetEta[idx],kr_nonbjetPhi[idx],kr_nonbjetM[idx])
                            # if (kr_jettemp.Pt()>50. and abs(kr_jettemp.Eta())<2.4):
                            if (kr_jettemp.Pt()>0. and abs(kr_jettemp.Eta())<9.):
                                kr_rho = 340./(kr_jettemp+kr_ttbar).M()

                    mlb_min = 999.
                    comb = 999
                    comb1 = 999
                    comb2 = 999
                    for i in range(bjetPt.size()):
                        bjettemp=ROOT.TLorentzVector(0.,0.,0.,0.)
                        bjettemp.SetPtEtaPhiM(bjetPt[i],bjetEta[i],bjetPhi[i],bjetM[i])
                        comb1 = (lep1+bjettemp).M()
                        comb2 = (lep2+bjettemp).M()
                        comb = comb1 if comb1<comb2 else comb2
                        if(comb<mlb_min):
                            mlb_min=comb
                    if mlb_min == 999:
                        mlb_min = 0.

                    dR_lepton1_jet1 = lep1.DeltaR(jets[0])
                    dR_lepton1_jet2 = lep1.DeltaR(jets[1])
                    dR_lepton1_jet3 = lep1.DeltaR(jets[2])
                    dR_lepton2_jet1 = lep2.DeltaR(jets[0])
                    dR_lepton2_jet2 = lep2.DeltaR(jets[1])
                    dR_lepton2_jet3 = lep2.DeltaR(jets[2])

                    dR_jet1_jet2 = jets[0].DeltaR(jets[1])
                    dR_jet1_jet3 = jets[0].DeltaR(jets[2])
                    dR_jet2_jet3 = jets[1].DeltaR(jets[2])

                    # jets_info.append(mlb_min)
                    # jets_info.append(dR_lepton1_jet1)
                    # jets_info.append(dR_lepton1_jet2)
                    # jets_info.append(dR_lepton1_jet3)
                    # jets_info.append(dR_lepton2_jet1)
                    # jets_info.append(dR_lepton2_jet2)
                    # jets_info.append(dR_lepton2_jet3)
                    # jets_info.append(dR_jet1_jet2)
                    # jets_info.append(dR_jet1_jet3)
                    # jets_info.append(dR_jet2_jet3)
                    #
                    jets_info.append(lep1.Px())
                    jets_info.append(lep1.Py())
                    jets_info.append(lep1.Pz())
                    jets_info.append(lep1.E())

                    jets_info.append(lep2.Px())
                    jets_info.append(lep2.Py())
                    jets_info.append(lep2.Pz())
                    jets_info.append(lep2.E())
                    #
                    # jets_info.append(met.Px())
                    # jets_info.append(met.Py())
                    # jets_info.append(met.Pz())
                    jets_info.append(met.E())

                    jets_info.append(met_significance[0])

                    # print (kr_rho, lkr_rho)

                    # if (haslkrs and not np.isnan(lkr_rho)):
                    #     jets_info.append(lkr_rho)
                    # else:
                    #     jets_info.append(0.)
                    # if (haskrs and not np.isnan(kr_rho)):
                    #     jets_info.append(kr_rho)
                    # else:
                    #     jets_info.append(0.)

                    # print ("----")
                    # print (jets_info)
                    eventInJet.append(jets_info)
                    # eventOut.append(genrho[0])
                    eventOut.append(genrhoWithNu[0])
    return eventInJet,eventOut,weights

def loadMassDataFlat(path, treeName, nJets=3, maxEvents=0, withBTag = False, withCharge = False):
    eventInJet=[]
    eventOut=[]
    weights=[]
    maxJets=nJets

    f = ROOT.TFile.Open(path)
    tree = f.Get(treeName)

    print("Read TTree: {} (Entries: {})".format(treeName, tree.GetEntries()))

    passStep3 = np.array([0], dtype=bool)
    tree.SetBranchAddress("passStep3", passStep3)

    hasKinRecoSolution = np.array([0], dtype=bool)
    tree.SetBranchAddress("hasKinRecoSolution", hasKinRecoSolution)

    hasLooseKinRecoSolution = np.array([0], dtype=bool)
    tree.SetBranchAddress("hasLooseKinRecoSolution", hasLooseKinRecoSolution)

    jetPt = ROOT.std.vector('float')()
    tree.SetBranchAddress("jets_pt", ROOT.AddressOf(jetPt))

    jetEta = ROOT.std.vector('float')()
    tree.SetBranchAddress("jets_eta", ROOT.AddressOf(jetEta))

    jetPhi = ROOT.std.vector('float')()
    tree.SetBranchAddress("jets_phi", ROOT.AddressOf(jetPhi))

    jetM = ROOT.std.vector('float')()
    tree.SetBranchAddress("jets_m", ROOT.AddressOf(jetM))

    lkr_nonbjetPt = ROOT.std.vector('float')()
    tree.SetBranchAddress("looseKinReco_nonbjets_pt", ROOT.AddressOf(lkr_nonbjetPt))
    lkr_nonbjetEta = ROOT.std.vector('float')()
    tree.SetBranchAddress("looseKinReco_nonbjets_eta", ROOT.AddressOf(lkr_nonbjetEta))
    lkr_nonbjetPhi = ROOT.std.vector('float')()
    tree.SetBranchAddress("looseKinReco_nonbjets_phi", ROOT.AddressOf(lkr_nonbjetPhi))
    lkr_nonbjetM = ROOT.std.vector('float')()
    tree.SetBranchAddress("looseKinReco_nonbjets_m", ROOT.AddressOf(lkr_nonbjetM))

    kr_nonbjetPt = ROOT.std.vector('float')()
    tree.SetBranchAddress("kinReco_nonbjets_pt", ROOT.AddressOf(kr_nonbjetPt))
    kr_nonbjetEta = ROOT.std.vector('float')()
    tree.SetBranchAddress("kinReco_nonbjets_eta", ROOT.AddressOf(kr_nonbjetEta))
    kr_nonbjetPhi = ROOT.std.vector('float')()
    tree.SetBranchAddress("kinReco_nonbjets_phi", ROOT.AddressOf(kr_nonbjetPhi))
    kr_nonbjetM = ROOT.std.vector('float')()
    tree.SetBranchAddress("kinReco_nonbjets_m", ROOT.AddressOf(kr_nonbjetM))

    # jetBTagged = ROOT.std.vector('bool')()
    # tree.SetBranchAddress("jets_btag", ROOT.AddressOf(jetBTagged))

    # jetCharge = ROOT.std.vector('float')()
    # tree.SetBranchAddress("jets_charge", ROOT.AddressOf(jetCharge))

    # jetTopMatched = ROOT.std.vector('bool')()
    # tree.SetBranchAddress("jets_topMatched", ROOT.AddressOf(jetTopMatched))

    gen_top_pt = np.array([0], dtype='f')
    tree.SetBranchAddress("gen_top_pt", gen_top_pt)
    gen_top_eta = np.array([0], dtype='f')
    tree.SetBranchAddress("gen_top_eta", gen_top_eta)
    gen_top_phi = np.array([0], dtype='f')
    tree.SetBranchAddress("gen_top_phi", gen_top_phi)
    gen_top_m = np.array([0], dtype='f')
    tree.SetBranchAddress("gen_top_m", gen_top_m)

    gen_antitop_pt = np.array([0], dtype='f')
    tree.SetBranchAddress("gen_antitop_pt", gen_antitop_pt)
    gen_antitop_eta = np.array([0], dtype='f')
    tree.SetBranchAddress("gen_antitop_eta", gen_antitop_eta)
    gen_antitop_phi = np.array([0], dtype='f')
    tree.SetBranchAddress("gen_antitop_phi", gen_antitop_phi)
    gen_antitop_m = np.array([0], dtype='f')
    tree.SetBranchAddress("gen_antitop_m", gen_antitop_m)

    weight = np.array([0], dtype='d')
    tree.SetBranchAddress("weight", weight)
    leptonSF = np.array([0], dtype='d')
    tree.SetBranchAddress("leptonSF", leptonSF)
    btagSF = np.array([0], dtype='d')
    tree.SetBranchAddress("btagSF", btagSF)
    pileupSF = np.array([0], dtype='d')
    tree.SetBranchAddress("pileupSF", pileupSF)
    prefiringWeight = np.array([0], dtype='d')
    tree.SetBranchAddress("l1PrefiringWeight", prefiringWeight)



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
    looseKinReco_ttbar_m =  np.array([0], dtype='f')
    # tree.SetBranchAddress("looseKinReco_ttbar_m", looseKinReco_nonbjets)

    from progress.bar import IncrementalBar

    if maxEvents!=0:
        maxEntries = maxEvents
    else:
        maxEntries = tree.GetEntries()

    maxForBar = int(maxEntries/200000)

    bar = IncrementalBar('Processing', max=maxForBar, suffix='%(percent).1f%% - %(eta)ds')
    # bar = IncrementalBar('Processing', max=maxForBar, suffix='%(percent).1f%%')

    for i in range(tree.GetEntries()):

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

        pass3= bool(passStep3)
        haskrs = bool(hasKinRecoSolution)
        haslkrs = bool(hasLooseKinRecoSolution)
        totWeight=weight[0]*btagSF[0]*leptonSF[0]*pileupSF[0]*prefiringWeight[0]
        # totWeight=btagSF[0]*leptonSF[0]*pileupSF[0]*prefiringWeight[0]

        if(pass3==True):
            numJets = jetPt.size()
            countB=0.
            # numGenJets = genjetPt.size()
            if(numJets>1):
                if(True):

                    for idx in range(maxJets):
                        if(idx<numJets):
                            jet4=ROOT.TLorentzVector(0.,0.,0.,0.)
                            jet4.SetPtEtaPhiM(jetPt[idx],jetEta[idx],jetPhi[idx],jetM[idx])
                            jets_info.append(jet4.Px())
                            jets_info.append(jet4.Py())
                            jets_info.append(jet4.Pz())
                            jets_info.append(jet4.E())

                            # bTagged=int(bool(jetBTagged[idx]))
                            # if withBTag:
                            #     jets_info.append(bTagged)
                            # if withCharge:
                            #     jets_info.append(jetCharge[idx])
                        else:
                            jets_info.append(0.)
                            jets_info.append(0.)
                            jets_info.append(0.)
                            jets_info.append(0.)
                            # if withBTag:
                            #     jets_info.append(0.)
                            # if withCharge:
                            #     jets_info.append(0.)
                    weights.append(totWeight)

                    gen_top = ROOT.TLorentzVector(0.,0.,0.,0.)
                    gen_top.SetPtEtaPhiM(gen_top_pt, gen_top_eta, gen_top_phi, gen_top_m)
                    gen_antitop = ROOT.TLorentzVector(0.,0.,0.,0.)
                    gen_antitop.SetPtEtaPhiM(gen_antitop_pt, gen_antitop_eta, gen_antitop_phi, gen_antitop_m)
                    gen_ttbarM = (gen_top+gen_antitop).M()

                    lep1=ROOT.TLorentzVector(0.,0.,0.,0.)
                    lep1.SetPtEtaPhiM(lepton1_pt,lepton1_eta,lepton1_phi,lepton1_m)
                    lep2=ROOT.TLorentzVector(0.,0.,0.,0.)
                    lep2.SetPtEtaPhiM(lepton2_pt,lepton2_eta,lepton2_phi,lepton2_m)
                    met=ROOT.TLorentzVector(0.,0.,0.,0.)
                    met.SetPtEtaPhiM(met_pt,0.,met_phi,0.)

                    kr_top=ROOT.TLorentzVector(0.,0.,0.,0.)
                    kr_top.SetPtEtaPhiM(kinReco_top_pt,kinReco_top_eta,kinReco_top_phi,kinReco_top_m)
                    kr_antitop=ROOT.TLorentzVector(0.,0.,0.,0.)
                    kr_antitop.SetPtEtaPhiM(kinReco_antitop_pt,kinReco_antitop_eta,kinReco_antitop_phi,kinReco_antitop_m)
                    kr_ttbar= kr_antitop + kr_top
                    lkr_ttbar=ROOT.TLorentzVector(0.,0.,0.,0.)
                    lkr_ttbar.SetPtEtaPhiM(looseKinReco_ttbar_pt,looseKinReco_ttbar_eta,looseKinReco_ttbar_phi,looseKinReco_ttbar_m)

                    lkr_mass=lkr_ttbar.M()
                    kr_mass=kr_ttbar.M()

                    jets_info.append(lep1.Px())
                    jets_info.append(lep1.Py())
                    jets_info.append(lep1.Pz())
                    jets_info.append(lep1.E())

                    jets_info.append(lep2.Px())
                    jets_info.append(lep2.Py())
                    jets_info.append(lep2.Pz())
                    jets_info.append(lep2.E())

                    # jets_info.append(met.Px())
                    # jets_info.append(met.Py())
                    # jets_info.append(met.Pz())
                    jets_info.append(met.E())


                    # if (haslkrs and np.isnan(lkr_mass)):
                    #     jets_info.append(lkr_mass)
                    # else:
                    #     jets_info.append(0.)
                    # if (haskrs and np.isnan(kr_mass)):
                    #     jets_info.append(kr_mass)
                    # else:
                    #     jets_info.append(0.)

                    eventInJet.append(jets_info)
                    eventOut.append(gen_ttbarM)
    return eventInJet,eventOut,weights

# def correlation_coefficient_loss(y_true, y_pred):
#     x = y_true
#     y = y_pred
#     mx = K.mean(x)
#     my = K.mean(y)
#     xm, ym = x-mx, y-my
#     r_num = K.sum(tf.multiply(xm,ym))
#     r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
#     r = r_num / r_den
#
#     r = K.maximum(K.minimum(r, 1.0), -1.0)
#     # print (1. - K.square(r))
#     return 1. - K.square(r)
# def correlationLoss(x,y, axis=-2):
# def correlation_coefficient_loss(x,y,sample_weight=None, axis=-2):
# def correlation_coefficient_loss(x,y, axis=-2):
def correlation_coefficient_loss(y_true,y_pred,sample_weight=None, axis=-2):
  """Loss function that maximizes the pearson correlation coefficient between the predicted values and the labels,
  while trying to have the same mean and variance"""
  x=y_true
  y=y_pred

  x = tf.convert_to_tensor(x)
  # y = tf.math_ops.cast(y, x.dtype)
  y = tf.cast(y, x.dtype)
  n = tf.cast(tf.shape(x)[axis], x.dtype)
  xsum = tf.reduce_sum(x, axis=axis)
  ysum = tf.reduce_sum(y, axis=axis)
  xmean = xsum / n
  ymean = ysum / n
  xsqsum = tf.reduce_sum( tf.squared_difference(x, xmean), axis=axis)
  ysqsum = tf.reduce_sum( tf.squared_difference(y, ymean), axis=axis)
  cov = tf.reduce_sum( (x - xmean) * (y - ymean), axis=axis)
  corr = cov / tf.sqrt(xsqsum * ysqsum)
  # absdif = tmean(tf.abs(x - y), axis=axis) / tf.sqrt(yvar)
  sqdif = tf.reduce_sum(tf.squared_difference(x, y), axis=axis) / n / tf.sqrt(ysqsum / n)
  # meandif = tf.abs(xmean - ymean) / tf.abs(ymean)
  # vardif = tf.abs(xvar - yvar) / yvar
  # return tf.convert_to_tensor( K.mean(tf.constant(1.0, dtype=x.dtype) - corr + (meandif * 0.01) + (vardif * 0.01)) , dtype=tf.float32 )
  return tf.convert_to_tensor( K.mean(tf.constant(1.0, dtype=x.dtype) - corr + (0.01 * sqdif)) , dtype=tf.float32 )


def loadTaggerData(path, treeName, nJets=20, maxEvents=0, normX=False, normY=False, normValue=170.):
    ##Initliaze all containers
    # jets_info=[]
    # jets_matchInfo=[]
    # other_info=[]
    eventInJet=[]
    eventInOther=[]
    eventOut=[]
    maxJets=nJets


    f = ROOT.TFile.Open(path)


    tree = f.Get(treeName)

    print("Read TTree: {} (Entries: {})".format(treeName, tree.GetEntries()))

    # passStep3 = np.array([0], dtype=np.float32)
    passStep3 = np.array([0], dtype=bool)
    tree.SetBranchAddress("passStep3", passStep3)

    jetPt = ROOT.std.vector('float')()
    tree.SetBranchAddress("jets_pt", ROOT.AddressOf(jetPt))

    jetEta = ROOT.std.vector('float')()
    tree.SetBranchAddress("jets_eta", ROOT.AddressOf(jetEta))

    jetPhi = ROOT.std.vector('float')()
    tree.SetBranchAddress("jets_phi", ROOT.AddressOf(jetPhi))

    jetM = ROOT.std.vector('float')()
    tree.SetBranchAddress("jets_m", ROOT.AddressOf(jetM))

    jetBTagged = ROOT.std.vector('bool')()
    tree.SetBranchAddress("jets_btag", ROOT.AddressOf(jetBTagged))

    jetCharge = ROOT.std.vector('bool')()
    tree.SetBranchAddress("jets_charge", ROOT.AddressOf(jetCharge))

    jetTopMatched = ROOT.std.vector('bool')()
    tree.SetBranchAddress("jets_topMatched", ROOT.AddressOf(jetTopMatched))

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

    ##Loop over event tree
    for i in range(tree.GetEntries()):
    # for i in range(100):
        jets_info=[]
        jets_matchInfo=[]
        other_info=[]
        tree.GetEntry(i)

        if(i%100000==0):
            print("Reading event: {}".format(i))

        if(maxEvents!=0):
            if(i%maxEvents==0 and i!=0):
                break

        pass3= bool(passStep3)

        if(pass3==True):
            # print("pass ",passStep3,pass3)
            numJets = jetPt.size()
            if(numJets>1):
                # for idx in range(jetPt.size()):
                for idx in range(maxJets):
                    jet=[]
                    if(idx<numJets):
                        jet4=ROOT.TLorentzVector(0.,0.,0.,0.)
                        jet4.SetPtEtaPhiM(jetPt[idx],jetEta[idx],jetPhi[idx],jetM[idx])
                        jet.append(jet4.Px())
                        jet.append(jet4.Py())
                        jet.append(jet4.Pz())
                        jet.append(jet4.E())
                        jet.append(int(bool(jetBTagged[idx])))
                        jet.append(jetCharge[idx])
                        # print (int(bool(jetBTagged[idx])))
                        jets_matchInfo.append(int(bool(jetTopMatched[idx])))
                    else:
                        jet.append(-999.)
                        jet.append(-999.)
                        jet.append(-999.)
                        jet.append(-999.)
                        jet.append(-999.)
                        jet.append(-999.)
                        jets_matchInfo.append(0.)
                    jets_info.append(jet)


                lep1=ROOT.TLorentzVector(0.,0.,0.,0.)
                lep1.SetPtEtaPhiM(lepton1_pt,lepton1_eta,lepton1_phi,lepton1_m)
                lep2=ROOT.TLorentzVector(0.,0.,0.,0.)
                lep2.SetPtEtaPhiM(lepton2_pt,lepton2_eta,lepton2_phi,lepton2_m)
                met=ROOT.TLorentzVector(0.,0.,0.,0.)
                met.SetPtEtaPhiM(met_pt,0.,met_phi,0.)

                l1=[]
                l2=[]
                miss=[]
                l1.append(lep1.Px())
                l1.append(lep1.Py())
                l1.append(lep1.Pz())
                l1.append(lep1.E())

                l2.append(lep2.Px())
                l2.append(lep2.Py())
                l2.append(lep2.Pz())
                l2.append(lep2.E())

                miss.append(met.Px())
                miss.append(met.Py())
                miss.append(met.Pz())
                miss.append(met.E())

                other_info.append(l1)
                other_info.append(l2)
                other_info.append(miss)

                # print (jets_info)
                # print (jets_matchInfo)
                # print (other_info)

                eventInJet.append(jets_info)
                eventInOther.append(other_info)
                eventOut.append(jets_matchInfo)
    return eventInJet,eventInOther,eventOut


def loadTaggerDataBinned(path, treeName, nJets=6, maxEvents=0, normX=False, normY=False, normValue=170.):
    ##Initliaze all containers
    # jets_info=[]
    # jets_matchInfo=[]
    # other_info=[]
    eventInJet=[]
    # eventInOther=[]
    eventOut=[]
    eventNBJets=[]
    eventNJets=[]
    weights=[]
    maxJets=nJets


    f = ROOT.TFile.Open(path)


    tree = f.Get(treeName)

    print("Read TTree: {} (Entries: {})".format(treeName, tree.GetEntries()))

    # passStep3 = np.array([0], dtype=np.float32)
    passStep3 = np.array([0], dtype=bool)
    tree.SetBranchAddress("passStep3", passStep3)

    jetPt = ROOT.std.vector('float')()
    tree.SetBranchAddress("jets_pt", ROOT.AddressOf(jetPt))

    jetEta = ROOT.std.vector('float')()
    tree.SetBranchAddress("jets_eta", ROOT.AddressOf(jetEta))

    jetPhi = ROOT.std.vector('float')()
    tree.SetBranchAddress("jets_phi", ROOT.AddressOf(jetPhi))

    jetM = ROOT.std.vector('float')()
    tree.SetBranchAddress("jets_m", ROOT.AddressOf(jetM))

    jetBTagged = ROOT.std.vector('bool')()
    tree.SetBranchAddress("jets_btag", ROOT.AddressOf(jetBTagged))

    jetCharge = ROOT.std.vector('float')()
    tree.SetBranchAddress("jets_charge", ROOT.AddressOf(jetCharge))

    jetTopMatched = ROOT.std.vector('bool')()
    tree.SetBranchAddress("jets_topMatched", ROOT.AddressOf(jetTopMatched))

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

    weight = np.array([0], dtype='d')
    tree.SetBranchAddress("weight", weight)
    leptonSF = np.array([0], dtype='d')
    tree.SetBranchAddress("leptonSF", leptonSF)
    btagSF = np.array([0], dtype='d')
    tree.SetBranchAddress("btagSF", btagSF)
    pileupSF = np.array([0], dtype='d')
    tree.SetBranchAddress("pileupSF", pileupSF)

    ##Loop over event tree
    for i in range(tree.GetEntries()):
    # for i in range(100):
        jets_info=[]
        jets_matchInfo=[]
        other_info=[]
        tree.GetEntry(i)
        # totWeight=1.

        breakEvents=tree.GetEntries() if maxEvents==0 else maxEvents

        if(i%100000==0):
            # print("Reading event: "+str(i)+" ("+str(np.round(float(i)/tree.GetEntries()*100.,2))+"%)")
            print("Reading event: "+str(i)+" ("+str(np.round(float(i)/breakEvents*100.,2))+"%)")

        if(maxEvents!=0):
        	if(i%maxEvents==0 and i!=0):
        		break

        pass3= bool(passStep3)

        if(pass3==True):
            # print("pass ",passStep3,pass3)
            numJets = jetPt.size()

            countB=0
            if(numJets>1):
				# for idx in range(jetPt.size()):
                for idx in range(maxJets):
                    # jet=[]
                    if(idx<numJets):
                        jet4=ROOT.TLorentzVector(0.,0.,0.,0.)
                        jet4.SetPtEtaPhiM(jetPt[idx],jetEta[idx],jetPhi[idx],jetM[idx])
                        jets_info.append(jet4.Px())
                        jets_info.append(jet4.Py())
                        jets_info.append(jet4.Pz())
                        jets_info.append(jet4.E())
                        jets_info.append(int(bool(jetBTagged[idx])))
                        jets_info.append(jetCharge[idx])
                        countB+=int(bool(jetBTagged[idx]))
                        # print (int(bool(jetBTagged[idx])))
                        jets_matchInfo.append(int(bool(jetTopMatched[idx])))
                    else:
                        # jet.append(-999.)
                        # jet.append(-999.)
                        # jet.append(-999.)
                        # jet.append(-999.)
                        # jet.append(-999.)
                        jets_info.append(0.)
                        jets_info.append(0.)
                        jets_info.append(0.)
                        jets_info.append(0.)
                        jets_info.append(0.)
                        jets_info.append(0.)
                        jets_matchInfo.append(0.)
                    # jets_info.append(jet)
                eventNJets.append([numJets])
                eventNBJets.append([countB])

                lep1=ROOT.TLorentzVector(0.,0.,0.,0.)
                lep1.SetPtEtaPhiM(lepton1_pt,lepton1_eta,lepton1_phi,lepton1_m)
                lep2=ROOT.TLorentzVector(0.,0.,0.,0.)
                lep2.SetPtEtaPhiM(lepton2_pt,lepton2_eta,lepton2_phi,lepton2_m)
                met=ROOT.TLorentzVector(0.,0.,0.,0.)
                met.SetPtEtaPhiM(met_pt,0.,met_phi,0.)

                # l1=[]
                # l2=[]
                # miss=[]
                jets_info.append(lep1.Px())
                jets_info.append(lep1.Py())
                jets_info.append(lep1.Pz())
                jets_info.append(lep1.E())

                jets_info.append(lep2.Px())
                jets_info.append(lep2.Py())
                jets_info.append(lep2.Pz())
                jets_info.append(lep2.E())

                jets_info.append(met.Px())
                jets_info.append(met.Py())
                jets_info.append(met.Pz())
                jets_info.append(met.E())

                # other_info.append(l1)
                # other_info.append(l2)
                # other_info.append(miss)

                # print (jets_info)
                # print (jets_matchInfo)
                # print (other_info)

                eventInJet.append(jets_info)
                # eventInOther.append(other_info)
                eventOut.append(jets_matchInfo)
                totWeight=weight[0]*btagSF[0]*leptonSF[0]*pileupSF[0]
                weights.append(totWeight)
    # return eventInJet,eventInOther,eventOut,eventNJets,eventNBJets
    return eventInJet,eventOut,eventNJets,eventNBJets,weights




def normalizeData(data, scaleValue, invert=False):
	if invert:
		out = data * scaleValue
	else:
		out = data / scaleValue
	return out



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
    # print (nx,xmin,xmax,ny,ymin,ymax)
    # h.SetBins(nx,xmin,xmax,ny,ymin,ymax)
    h2.SetBins(nx,xmin,xmax,ny,ymin,ymax)

    for biny in range(1,nbinsy+1):
        for binx in range(nbinsx+1):
            ibin = h2.GetBin(binx,biny)
            h2.SetBinContent(ibin,0)

    # for (biny=1;biny<=nbinsy;biny++):
    for biny in range(1,nbinsy+1):
        by  = hold.GetYaxis().GetBinCenter(biny)
        iy  = h2.GetYaxis().FindBin(by)
        for binx in range(1,nbinsx+1):
        # for (binx=1;binx<=nbinsx;binx++):
            bx = hold.GetXaxis().GetBinCenter(binx)
            ix  = h2.GetXaxis().FindBin(bx)
            bin = hold.GetBin(binx,biny)
            ibin= h2.GetBin(ix,iy)
            cu  = hold.GetBinContent(bin)
            # h.AddBinContent(ibin,cu)
            h2.AddBinContent(ibin,cu)
    return h2

def drawAsGraph(h):

    g = ROOT.TGraphAsymmErrors()
    # g.SetDirectory(0)
    # for(int b = 0; b < h.GetNbinsX(); b++)
    for b in range(h.GetNbinsX()):
        x = h.GetBinLowEdge(b + 1) + 0.5 * h.GetBinWidth(b + 1)
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
    # drawOption = option
    # if(!flagUnc)
    # drawOption += "X"
    # g.Draw("ZP0X")
    return g


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
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
        if clear_devices:
            for node in graphdef_inf.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, graphdef_inf, output_names, freeze_var_names)
        return frozen_graph
