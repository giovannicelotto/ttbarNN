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

        pass3= bool(passStep3)

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

def loadRhoDataFlat(path, treeName, nJets=6, maxEvents=0, normX=False, normY=False, normValue=170.):
    eventInJet=[]
    eventOut=[]
    weights=[]
    maxJets=nJets

    f = ROOT.TFile.Open(path)
    tree = f.Get(treeName)

    print("Read TTree: {} (Entries: {})".format(treeName, tree.GetEntries()))

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

    for i in range(tree.GetEntries()):
        jets_info=[]
        jets_matchInfo=[]
        other_info=[]
        tree.GetEntry(i)
        totWeight=1.

        if(i%200000==0):
            print("Reading event: {}".format(i))

        if(maxEvents!=0):
        	if(i%maxEvents==0 and i!=0):
        		break

        pass3= bool(passStep3)
        # totWeight=weight[0]*btagSF[0]*leptonSF[0]*pileupSF[0]*prefiringWeight[0]
        totWeight=btagSF[0]*leptonSF[0]*pileupSF[0]*prefiringWeight[0]

        if(pass3==True):
            numJets = jetPt.size()
            countB=0.
            numGenJets = genjetPt.size()
            if(numJets>1):
                if(numGenJets>2 and genrho>0.):

                    for idx in range(maxJets):
                        if(idx<numJets):
                            jet4=ROOT.TLorentzVector(0.,0.,0.,0.)
                            jet4.SetPtEtaPhiM(jetPt[idx],jetEta[idx],jetPhi[idx],jetM[idx])
                            jets_info.append(jet4.Px())
                            jets_info.append(jet4.Py())
                            jets_info.append(jet4.Pz())
                            jets_info.append(jet4.E())

                            # bTagged=int(bool(jetBTagged[idx]))
                            # jets_info.append(bTagged)
                            # jets_info.append(jetCharge[idx])
                        else:
                            jets_info.append(0.)
                            jets_info.append(0.)
                            jets_info.append(0.)
                            jets_info.append(0.)
                    weights.append(totWeight)

                    lep1=ROOT.TLorentzVector(0.,0.,0.,0.)
                    lep1.SetPtEtaPhiM(lepton1_pt,lepton1_eta,lepton1_phi,lepton1_m)
                    lep2=ROOT.TLorentzVector(0.,0.,0.,0.)
                    lep2.SetPtEtaPhiM(lepton2_pt,lepton2_eta,lepton2_phi,lepton2_m)
                    met=ROOT.TLorentzVector(0.,0.,0.,0.)
                    met.SetPtEtaPhiM(met_pt,0.,met_phi,0.)

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

                    eventInJet.append(jets_info)
                    eventOut.append(genrho[0])
    return eventInJet,eventOut,weights


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


def	doPredictionPlots(y_test, y_predicted, folderName):
    E_test = y_test[:,0]
    Px_test = y_test[:,1]
    Py_test = y_test[:,2]
    Pz_test = y_test[:,3]

    E_predicted = y_predicted[:,0]
    Px_predicted = y_predicted[:,1]
    Py_predicted = y_predicted[:,2]
    Pz_predicted = y_predicted[:,3]

    nbins=40

    x_bins = np.linspace(100, 1500, nbins)
    y_bins = np.linspace(100, 1500, nbins)

    outFolderName = folderName + "/plots/"
    if not os.path.exists(outFolderName):
    	os.makedirs(outFolderName)

    f=plt.figure()
    # plt.subplot(211)
    values,bins,patches = plt.hist(E_test,bins=55,range=(300,1400),label="true",alpha=0.5)
    values2,bins2,patches2 = plt.hist(E_predicted,bins=55,range=(300,1400),label="reco",alpha=0.5)
    plt.legend(loc="best")
    plt.ylabel('Events')
    plt.xlabel('E(tt) [GeV]')
    # plt.subplot(212)
    # ratio = values2/values
    # v= []
    # for i in range(len(bins)-1):
    # 	v.append((bins[i]+bins[i+1])/2)
    # v=np.array(v)
    # plt.plot(v,ratio)
    # plt.ylim(0.5,1.5)
    # plt.ylabel("reco/true")
    # plt.xlabel('E(tt) [GeV]')
    f.savefig(outFolderName+"eval_Ett.pdf")

    f=plt.figure()
    # plt.subplot(211)
    values,bins,patches = plt.hist(Px_test,bins=20,range=(100,500),label="true",alpha=0.5)
    values2,bins2,patches2 = plt.hist(Px_predicted,bins=20,range=(100,500),label="reco",alpha=0.5)
    plt.legend(loc="best")
    plt.ylabel('Events')
    plt.xlabel('Px(tt) [GeV]')
    # plt.subplot(212)
    # ratio = values2/values
    # v= []
    # for i in range(len(bins)-1):
    # 	v.append((bins[i]+bins[i+1])/2)
    # v=np.array(v)
    # plt.plot(v,ratio)
    # plt.ylim(0.5,1.5)
    # plt.ylabel("reco/true")
    # plt.xlabel('Px(tt) [GeV]')
    f.savefig(outFolderName+"eval_Pxtt.pdf")
    f=plt.figure()
    # plt.subplot(211)
    values,bins,patches = plt.hist(Py_test,bins=20,range=(100,500),label="true",alpha=0.5)
    values2,bins2,patches2 = plt.hist(Py_predicted,bins=20,range=(100,500),label="reco",alpha=0.5)
    plt.legend(loc="best")
    plt.ylabel('Events')
    plt.xlabel('Py(tt) [GeV]')
    # plt.subplot(212)
    # ratio = values2/values
    # v= []
    # for i in range(len(bins)-1):
    # 	v.append((bins[i]+bins[i+1])/2)
    # v=np.array(v)
    # plt.plot(v,ratio)
    # plt.ylim(0.5,1.5)
    # plt.ylabel("reco/true")
    # plt.xlabel('Py(tt) [GeV]')
    f.savefig(outFolderName+"eval_Pytt.pdf")

    f=plt.figure()
    # plt.subplot(211)
    values,bins,patches = plt.hist(Pz_test,bins=45,range=(100,1000),label="true",alpha=0.5)
    values2,bins2,patches2 = plt.hist(Pz_predicted,bins=45,range=(100,1000),label="reco",alpha=0.5)
    plt.legend(loc="best")
    plt.ylabel('Events')
    plt.xlabel('Pz(tt) [GeV]')
    # plt.subplot(212)
    # ratio = values2/values
    # v= []
    # for i in range(len(bins)-1):
    # 	v.append((bins[i]+bins[i+1])/2)
    # v=np.array(v)
    # plt.plot(v,ratio)
    # plt.ylim(0.5,1.5)
    # plt.ylabel("reco/true")
    # plt.xlabel('Pz(tt) [GeV]')
    f.savefig(outFolderName+"eval_Pztt.pdf")


    f=plt.figure()
    plt.hist2d(E_test,E_predicted,bins=55,range=np.array([[100,1400],[100,1400]]))
    plt.xlabel('E(tt) true [GeV]')
    plt.ylabel('E(tt) reco [GeV]')
    f.savefig(outFolderName+"eval2D_Ett.pdf")
    f=plt.figure()
    plt.hist2d(Px_test,Px_predicted,bins=20,range=np.array([[100,500],[100,500]]))
    plt.xlabel('Px(tt) true [GeV]')
    plt.ylabel('Px(tt) reco [GeV]')
    f.savefig(outFolderName+"eval2D_Pxtt.pdf")
    f=plt.figure()
    plt.hist2d(Py_test,Py_predicted,bins=20,range=np.array([[100,500],[100,500]]))
    plt.xlabel('Py(tt) true [GeV]')
    plt.ylabel('Py(tt) reco [GeV]')
    f.savefig(outFolderName+"eval2D_Pytt.pdf")
    f=plt.figure()
    plt.hist2d(Pz_test,Pz_predicted,bins=45,range=np.array([[100,500],[100,500]]))
    plt.xlabel('Pz(tt) true [GeV]')
    plt.ylabel('Pz(tt) reco [GeV]')
    f.savefig(outFolderName+"eval2D_Pztt.pdf")


    histo = ROOT.TH2F( "bla", "RMS vs Gen", 3000, 0, 3000, 6000, -3000, 3000 )
    histo.SetDirectory(0)
    histoRecoGen = ROOT.TH2F( "bla2", "Reco vs Gen", 2000, 0, 2000, 2000, 0, 2000 )
    histoRecoGen.SetDirectory(0)
    histoRecoGen2 = ROOT.TH2F( "bla2", "Reco vs Gen", 8, 0, 2000, 8, 0, 2000 )
    histoRecoGen2.SetDirectory(0)
    histoGen = ROOT.TH1F ( "GenTTBarMass", "Gen", 2000, 0, 2000 )
    histoGen.SetDirectory(0)
    histoRecoGen3 = ROOT.TH2F( "bla2", "Reco vs Gen", 80, 0, 2000, 80, 0, 2000 )
    histoRecoGen3.SetDirectory(0)
    histoGen2 = ROOT.TH1F ( "GenTTBarMass", "Gen", 80, 0, 2000 )
    histoGen2.SetDirectory(0)
    for Etrue, Epred,pxtrue,pxpred,pytrue,pypred,pztrue,pzpred in zip(E_test, E_predicted,Px_test,Px_predicted,Py_test,Py_predicted,Pz_test,Pz_predicted):
        mpred = np.sqrt(Epred*Epred - pxpred*pxpred - pypred*pypred - pzpred*pzpred)
        mtrue = np.sqrt(Etrue*Etrue - pxtrue*pxtrue - pytrue*pytrue - pztrue*pztrue)
        diff = mtrue-mpred
        # print (mtrue,diff)
        histo.Fill(mtrue,diff)
        histoGen.Fill(mtrue)
        histoGen2.Fill(mtrue)
        histoRecoGen.Fill(mpred,mtrue)
        histoRecoGen2.Fill(mpred,mtrue)
        histoRecoGen3.Fill(mpred,mtrue)
    c=ROOT.TCanvas("c1","c1",800,800)

    histo.GetXaxis().SetRangeUser(350,1500)
    histo.GetYaxis().SetRangeUser(-1200,1200)
    histo.Draw("colz")
    c.SaveAs("testDiff.pdf")
    c.Clear()
    # histoRecoGen.Draw("text colz")
    # c.SaveAs("testMigTEST.pdf")
    # c.Clear()
    histoRecoGen2.Draw("text colz")
    c.SaveAs("testMigTEST2.pdf")
    c.Clear()

    # migration matrix
    # hMig = histoRecoGen.Clone().Rebin(15)
    # hMig = rebin2D(histoRecoGen,15,15)
    # hMig = histoRecoGen.Clone()
    hMig = histoRecoGen2.Clone()
    # hMig.SetDirectory(0)
    # hMig = hMig.Rebin2D(15,15,"new")
    # # hMig = histoRecoGen,15,15)
    hMig.SetTitle("")
    hMig.GetXaxis().SetTitle("ttbar mass" + ", rec. level")
    hMig.GetYaxis().SetTitle("ttbar mass" + ", gen. level")
    n = hMig.GetNbinsX()
    # for(int by = 1; by <= n; by++):
    for by in range(1,n+1):
    # for by in range(1,n):
        GenAndRec = hMig.Integral(1, n, by, by)
        # for(int bx = 1; bx <= n; bx++):
        for bx in range(1,n+1):
        # for bx in range(1,n):
            val = hMig.GetBinContent(bx, by)
            # print (bx,by,val,GenAndRec)
            val2 = 0. if GenAndRec==0. else val / GenAndRec
            # hMig.SetBinContent(bx, by, val / GenAndRec)
            hMig.SetBinContent(bx, by, val2)
    # c=ROOT.TCanvas("c1","c1",800,800)
    hMig.Draw("text colz")
    c.SaveAs("testMig.pdf")
    c.Clear()

    # PSE Plot



    # hResp = histoRecoGen.Clone().Rebin(15)

    # hResp = rebin2D(histoRecoGen.Clone(), 15, 15)
    hResp = histoRecoGen3.Clone()
    hResp.SetDirectory(0)
    # hResp = hResp.Rebin2D(15, 15,"new2")
    # hResp.SetDirectory(0)
    hGen = histoGen2.Clone()
    hGen.SetDirectory(0)
    # hGen = hGen.Rebin(15,"new3")
    # hGen.SetDirectory(0)


    markerSize = 1.5;

    n = hResp.GetNbinsX()

    hp = ROOT.TH1F("", "", n, 0.0, n)
    hs = ROOT.TH1F("", "", n, 0.0, n)
    he = ROOT.TH1F("", "", n, 0.0, n)
    hp.SetDirectory(0)
    hs.SetDirectory(0)
    he.SetDirectory(0)

    # for(int b = 1; b <= n; b++)
    for b in range(1,n+1):
        GenInBinAndRecInBin = hResp.GetBinContent(b, b)
        RecInBin = hResp.Integral(b, b, 1, n)
        GenInBinAndRec = hResp.Integral(1, n, b, b)
        GenInBinAll = hGen.GetBinContent(b)
        hp.SetBinContent(b, 0. if RecInBin==0 else GenInBinAndRecInBin / RecInBin)
        hs.SetBinContent(b, 0. if GenInBinAndRec==0 else GenInBinAndRecInBin / GenInBinAndRec)
        he.SetBinContent(b, 0. if GenInBinAll==0 else GenInBinAndRec / GenInBinAll)


    leg = ROOT.TLegend(0.15, 0.67, 0.40, 0.85)
    leg.SetTextFont(62)
    hr = ROOT.TH2F("", "", 1, he.GetBinLowEdge(1), he.GetBinLowEdge(n + 1), 1, 0.0, 1.0)
    hr.GetXaxis().SetTitle("Bin")
    hr.SetStats(0)
    hr.Draw()
    hp.SetMarkerColor(4)
    hp.SetMarkerStyle(23)
    hp.SetMarkerSize(markerSize)
    leg.AddEntry(hp, "Purity", "p")
    hp2=drawAsGraph(hp)
    hs.SetMarkerColor(2)
    hs.SetMarkerStyle(22)
    hs.SetMarkerSize(markerSize)
    leg.AddEntry(hs, "Stability", "p")
    hs2=drawAsGraph(hs)
    he.SetMarkerColor(8)
    he.SetMarkerStyle(20)
    he.SetMarkerSize(markerSize)
    leg.AddEntry(he, "Efficiency", "p")
    he2=drawAsGraph(he)

    hp2.Draw("ZP0X")
    hs2.Draw("ZP0X")
    he2.Draw("ZP0X")

    leg.Draw()
    c.SaveAs("testPSE.pdf")
    c.Clear()



    Xnb=14
    Xr1=350.
    Xr2=1500.
    dXbin=(Xr2-Xr1)/((Xnb));
    titleRMSVsGen_ptTop_full ="RMS (M_{true}^{t#bart} - M_{reco}^{t#bart}) vs M_{true}^{t#bart}; M_{true}^{t#bart}, GeV; RMS"
    titleMeanVsGen_ptTop_full ="Mean (M_{true}^{t#bart} - M_{reco}^{t#bart}) vs M_{true}^{t#bart}; M_{true}^{t#bart}, GeV;Mean"
    h_RMSVsGen_=ROOT.TH1F()
    h_RMSVsGen_.SetDirectory(0)
    h_RMSVsGen_.SetBins(Xnb,Xr1,Xr2)
    h_RMSVsGen_.SetTitleOffset(2.0)
    h_RMSVsGen_.GetXaxis().SetTitleOffset(1.20)
    h_RMSVsGen_.GetYaxis().SetTitleOffset(1.30)
    h_RMSVsGen_.SetTitle(titleRMSVsGen_ptTop_full);
    h_RMSVsGen_.SetStats(0)
    h_meanVsGen_=ROOT.TH1F()
    h_meanVsGen_.SetDirectory(0)
    h_meanVsGen_.SetBins(Xnb,Xr1,Xr2)
    h_meanVsGen_.SetTitleOffset(2.0)
    h_meanVsGen_.GetXaxis().SetTitleOffset(1.20)
    h_meanVsGen_.GetYaxis().SetTitleOffset(1.30)
    h_meanVsGen_.SetTitle(titleMeanVsGen_ptTop_full)
    h_meanVsGen_.SetStats(0)
    for i in range(Xnb):
    	h_RMSVsGen_.SetBinContent(i+1,(histo.ProjectionY("_py",histo.GetXaxis().FindFixBin(Xr1+i*dXbin) ,histo.GetXaxis().FindFixBin(Xr1+(i+1)*dXbin),"")).GetRMS());
    	h_RMSVsGen_.SetBinError(i+1,(histo.ProjectionY("_py",histo.GetXaxis().FindFixBin(Xr1+i*dXbin) ,histo.GetXaxis().FindFixBin(Xr1+(i+1)*dXbin),"")).GetRMSError());

    	h_meanVsGen_.SetBinContent(i+1,(histo.ProjectionY("_py",histo.GetXaxis().FindFixBin(Xr1+i*dXbin) ,histo.GetXaxis().FindFixBin(Xr1+(i+1)*dXbin),"")).GetMean());
    	h_meanVsGen_.SetBinError(i+1,(histo.ProjectionY("_py",histo.GetXaxis().FindFixBin(Xr1+i*dXbin) ,histo.GetXaxis().FindFixBin(Xr1+(i+1)*dXbin),"")).GetMeanError());
    h_RMSVsGen_.SaveAs("testRMS.root")
    h_meanVsGen_.SaveAs("testMEAN.root")
    # c=ROOT.TCanvas("c1","c1",800,800)
    h_RMSVsGen_.Draw()
    c.SaveAs("testRMS.pdf")
    c.Clear()
    h_meanVsGen_.Draw()
    c.SaveAs("testMEAN.pdf")

def	doPredictionPlots2(y_test, y_predicted, folderName):
	Pt_test = y_test[:,0]
	Y_test = y_test[:,1]
	Phi_test = y_test[:,2]
	M_test = y_test[:,3]

	Pt_predicted = y_predicted[:,0]
	Y_predicted = y_predicted[:,1]
	Phi_predicted = y_predicted[:,2]
	M_predicted = y_predicted[:,3]

	nbins=70

	x_binsPt = np.linspace(0, 1500, nbins)
	y_binsPt = np.linspace(0, 1500, nbins)
	x_binsY = np.linspace(-10, 10, nbins)
	y_binsY = np.linspace(-10, 10, nbins)
	x_binsPhi = np.linspace(-3., 3., nbins)
	y_binsPhi = np.linspace(-3., 3., nbins)
	x_binsM = np.linspace(100, 1500, nbins)
	y_binsM = np.linspace(100, 1500, nbins)

	outFolderName = folderName + "/plots/"
	if not os.path.exists(outFolderName):
		os.makedirs(outFolderName)

	f=plt.figure()
	# plt.subplot(211)
	values,bins,patches = plt.hist(Pt_test,bins=nbins,range=(100,1500),label="true",alpha=0.5)
	values2,bins2,patches2 = plt.hist(Pt_predicted,bins=nbins,range=(100,1500),label="reco",alpha=0.5)
	plt.legend(loc="best")
	plt.ylabel('Events')
	plt.xlabel('Pt(tt) [GeV]')
	# plt.subplot(212)
	# ratio = values2/values
	# v= []
	# for i in range(len(bins)-1):
	# 	v.append((bins[i]+bins[i+1])/2)
	# v=np.array(v)
	# plt.plot(v,ratio)
	# plt.ylim(0.5,1.5)
	# plt.ylabel("reco/true")
	# plt.xlabel('Pt(tt) [GeV]')
	f.savefig(outFolderName+"eval_Pttt.pdf")

	f=plt.figure()
	# plt.subplot(211)
	values,bins,patches = plt.hist(Y_test,bins=nbins,range=(-20,20),label="true",alpha=0.5)
	values2,bins2,patches2 = plt.hist(Y_predicted,bins=nbins,range=(-20,20),label="reco",alpha=0.5)
	plt.legend(loc="best")
	plt.ylabel('Events')
	plt.xlabel('Y(tt) [GeV]')
	# plt.subplot(212)
	# ratio = values2/values
	# v= []
	# for i in range(len(bins)-1):
	# 	v.append((bins[i]+bins[i+1])/2)
	# v=np.array(v)
	# plt.plot(v,ratio)
	# plt.ylim(0.5,1.5)
	# plt.ylabel("reco/true")
	# plt.xlabel('Y(tt) [GeV]')
	f.savefig(outFolderName+"eval_Ytt.pdf")
	f=plt.figure()
	# plt.subplot(211)
	values,bins,patches = plt.hist(Phi_test,bins=nbins,range=(-3,3),label="true",alpha=0.5)
	values2,bins2,patches2 = plt.hist(Phi_predicted,bins=nbins,range=(-3,3),label="reco",alpha=0.5)
	plt.legend(loc="best")
	plt.ylabel('Events')
	plt.xlabel('Phi(tt) [GeV]')
	# plt.subplot(212)
	# ratio = values2/values
	# v= []
	# for i in range(len(bins)-1):
	# 	v.append((bins[i]+bins[i+1])/2)
	# v=np.array(v)
	# plt.plot(v,ratio)
	# plt.ylim(0.5,1.5)
	# plt.ylabel("reco/true")
	# plt.xlabel('Phi(tt) [GeV]')
	f.savefig(outFolderName+"eval_Phitt.pdf")

	f=plt.figure()
	# plt.subplot(211)
	values,bins,patches = plt.hist(M_test,bins=nbins,range=(100,800),label="true",alpha=0.5)
	values2,bins2,patches2 = plt.hist(M_predicted,bins=nbins,range=(100,800),label="reco",alpha=0.5)
	plt.legend(loc="best")
	plt.ylabel('Events')
	plt.xlabel('M(tt) [GeV]')
	# plt.subplot(212)
	# ratio = values2/values
	# v= []
	# for i in range(len(bins)-1):
	# 	v.append((bins[i]+bins[i+1])/2)
	# v=np.array(v)
	# plt.plot(v,ratio)
	# plt.ylim(0.5,1.5)
	# plt.ylabel("reco/true")
	# plt.xlabel('M(tt) [GeV]')
	f.savefig(outFolderName+"eval_Mtt.pdf")


	f=plt.figure()
	plt.hist2d(Pt_test,Pt_predicted,bins=nbins,range=np.array([[0,1500],[0,1500]]))
	plt.xlabel('Pt(tt) true [GeV]')
	plt.ylabel('Pt(tt) reco [GeV]')
	f.savefig(outFolderName+"eval2D_Pttt.pdf")
	f=plt.figure()
	plt.hist2d(Y_test,Y_predicted,bins=nbins,range=np.array([[-20,20],[-20,20]]))
	plt.xlabel('Y(tt) true [GeV]')
	plt.ylabel('Y(tt) reco [GeV]')
	f.savefig(outFolderName+"eval2D_Ytt.pdf")
	f=plt.figure()
	plt.hist2d(Phi_test,Phi_predicted,bins=nbins,range=np.array([[-3,3],[-3,3]]))
	plt.xlabel('Phi(tt) true [GeV]')
	plt.ylabel('Phi(tt) reco [GeV]')
	f.savefig(outFolderName+"eval2D_Phitt.pdf")
	f=plt.figure()
	plt.hist2d(M_test,M_predicted,bins=nbins,range=np.array([[100,1500],[100,1500]]))
	plt.xlabel('M(tt) true [GeV]')
	plt.ylabel('M(tt) reco [GeV]')
	f.savefig(outFolderName+"eval2D_Mtt.pdf")


	histo = ROOT.TH2F( "bla", "RMS vs Gen", 3000, 0, 3000, 6000, -3000, 3000 )
	histo.SetDirectory(0)
	for Pttrue, Ptpred, Ytrue, Ypred, Phitrue, Phipred, Mtrue, Mpred in zip(Pt_test, Pt_predicted,Y_test,Y_predicted,Phi_test,Phi_predicted,M_test,M_predicted):
		# mpred = np.sqrt(Epred*Epred - pxpred*pxpred - pypred*pypred - pzpred*pzpred)
		# mtrue = np.sqrt(Etrue*Etrue - pxtrue*pxtrue - pytrue*pytrue - pztrue*pztrue)
		mpred = Mpred
		mtrue = Mtrue
		diff = mtrue-mpred
		# print (mtrue,diff)
		histo.Fill(mtrue,diff)

	Xnb=14
	Xr1=350.
	Xr2=1500.
	dXbin=(Xr2-Xr1)/((Xnb));
	titleRMSVsGen_ptTop_full ="RMS (p_{t, true}^{top} - p_{t, reco}^{top}) vs p_{t, true}^{top}; p_{t, true}^{top}, GeV;RMS"
	titleMeanVsGen_ptTop_full ="Mean (p_{t, true}^{top} - p_{t, reco}^{top}) vs p_{t, true}^{top}; p_{t, true}^{top}, GeV;Mean"
	h_RMSVsGen_=ROOT.TH1F()
	h_RMSVsGen_.SetDirectory(0)
	h_RMSVsGen_.SetBins(Xnb,Xr1,Xr2)
	h_RMSVsGen_.SetTitleOffset(2.0)
	h_RMSVsGen_.GetXaxis().SetTitleOffset(1.20)
	h_RMSVsGen_.GetYaxis().SetTitleOffset(1.30)
	h_RMSVsGen_.SetTitle(titleRMSVsGen_ptTop_full);
	h_RMSVsGen_.SetStats(0)
	h_meanVsGen_=ROOT.TH1F()
	h_meanVsGen_.SetDirectory(0)
	h_meanVsGen_.SetBins(Xnb,Xr1,Xr2)
	h_meanVsGen_.SetTitleOffset(2.0)
	h_meanVsGen_.GetXaxis().SetTitleOffset(1.20)
	h_meanVsGen_.GetYaxis().SetTitleOffset(1.30)
	h_meanVsGen_.SetTitle(titleMeanVsGen_ptTop_full)
	h_meanVsGen_.SetStats(0)
	for i in range(Xnb):
		h_RMSVsGen_.SetBinContent(i+1,(histo.ProjectionY("_py",histo.GetXaxis().FindFixBin(Xr1+i*dXbin) ,histo.GetXaxis().FindFixBin(Xr1+(i+1)*dXbin),"")).GetRMS());
		h_RMSVsGen_.SetBinError(i+1,(histo.ProjectionY("_py",histo.GetXaxis().FindFixBin(Xr1+i*dXbin) ,histo.GetXaxis().FindFixBin(Xr1+(i+1)*dXbin),"")).GetRMSError());

		h_meanVsGen_.SetBinContent(i+1,(histo.ProjectionY("_py",histo.GetXaxis().FindFixBin(Xr1+i*dXbin) ,histo.GetXaxis().FindFixBin(Xr1+(i+1)*dXbin),"")).GetMean());
		h_meanVsGen_.SetBinError(i+1,(histo.ProjectionY("_py",histo.GetXaxis().FindFixBin(Xr1+i*dXbin) ,histo.GetXaxis().FindFixBin(Xr1+(i+1)*dXbin),"")).GetMeanError());
	h_RMSVsGen_.SaveAs("testRMS.root")
	h_meanVsGen_.SaveAs("testMEAN.root")
	c=ROOT.TCanvas("c1","c1",800,800)
	h_RMSVsGen_.Draw()
	c.SaveAs("testRMS.pdf")
	c.Clear()
	h_meanVsGen_.Draw()
	c.SaveAs("testMEAN.pdf")




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
