from utils import helpers, models,style
from sklearn.model_selection import train_test_split

import numpy as np
import tensorflow as tf
from keras import optimizers,losses
import keras.backend as K
K.tensorflow_backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))))

inputFile="rhoInput/total.root"
treeName="miniTree"
normVal = 1.
# inJetX,inOtherX,outY,NJet,NBJet = helpers.loadRhoData(inputFile, treeName, nJets=6, maxEvents=0, normX=False, normY=False, normValue=normVal)
# inX,outY,NJet,NBJet,weights = helpers.loadRhoDataFlat(inputFile, treeName, nJets=6, maxEvents=0, normX=False, normY=False, normValue=normVal)
inX,outY,NJet,NBJet,weights = helpers.loadRhoDataFlat(inputFile, treeName, nJets=4, maxEvents=0, normX=False, normY=False, normValue=normVal)

# inJetX=np.array(inJetX)
# inOtherX=np.array(inOtherX)
inX=np.array(inX)
outY=np.array(outY)
NJet=np.array(NJet)
NBJet=np.array(NBJet)
weights=np.array(weights)


# np.save("rhoInput/eval_inJetX.npy", inJetX)
# np.save("rhoInput/eval_inOtherX.npy", inOtherX)
# np.save("rhoInput/eval_outY.npy", outY)
# np.save("rhoInput/eval_njet.npy", NJet)
# np.save("rhoInput/eval_nbjet.npy", NBJet)
# np.save("rhoInput/flat_inX.npy", inX)
# # # np.save("rhoInput/flat_inOtherX.npy", inOtherX)
# np.save("rhoInput/flat_outY.npy", outY)
# np.save("rhoInput/flat_njet.npy", NJet)
# np.save("rhoInput/flat_nbjet.npy", NBJet)
# np.save("rhoInput/flat_weights.npy", weights)
np.save("rhoInput/flat_inX_less.npy", inX)
np.save("rhoInput/flat_outY_less.npy", outY)
np.save("rhoInput/flat_njet_less.npy", NJet)
np.save("rhoInput/flat_nbjet_less.npy", NJet)
np.save("rhoInput/flat_weights_less.npy", weights)

# inJetX = np.load("rhoInput/eval_inJetX.npy")
# inOtherX = np.load("rhoInput/eval_inOtherX.npy")
# outY = np.load("rhoInput/eval_outY.npy")
# NJet = np.load("rhoInput/eval_njet.npy")
# NBJet = np.load("rhoInput/eval_nbjet.npy")
# inX = np.load("rhoInput/flat_inX.npy")
# # inOtherX = np.load("rhoInput/flat_inOtherX.npy")
# outY = np.load("rhoInput/flat_outY.npy")
# NJet = np.load("rhoInput/flat_njet.npy")
# NBJet = np.load("rhoInput/flat_nbjet.npy")
# weights = np.load("rhoInput/flat_weights.npy")
inX = np.load("rhoInput/flat_inX_less.npy")
outY = np.load("rhoInput/flat_outY_less.npy")
NJet = np.load("rhoInput/flat_njet_less.npy")
NBJet = np.load("rhoInput/flat_nbjet_less.npy")
weights = np.load("rhoInput/flat_weights_less.npy")

# print (inJetX.shape)
# print (inJetX[0])
# print (inOtherX.shape)
print (inX.shape)
print (inX[0])
# inX = np.delete(inX, [4,9,14,19,24,29], 1)
# inX = np.delete(inX, [12,13,14,15,24], 1)
inX = np.delete(inX, [24], 1)




# inX = inX/340.
print (inX.shape)
print (inX[0])
# print (inX[0])
# print (inOtherX[0])
print (outY.shape)
print (outY[0])
# print (NJet.shape)
# print (NJet[0])
# print (NBJet.shape)
# print (NBJet[0])

print (weights.shape)
print (weights[0])

from sklearn.preprocessing import *
scaler=QuantileTransformer(output_distribution='uniform')
# scaler=RobustScaler()
# scaler=QuantileTransformer(output_distribution='normal')
# scaler=StandardScaler()
outY=outY.reshape((outY.shape[0],1))
scaler.fit(outY)
# outY=scaler.transform(outY)

# inJetX_train, inJetX_test, inOtherX_train, inOtherX_test,outY_train,outY_test, NJet_train, NJet_test,NBJet_train,NBJet_test= train_test_split(inJetX, inOtherX,outY,NJet,NBJet, test_size=0.33)
# inX_train, inX_test,outY_train,outY_test, NJet_train, NJet_test,NBJet_train,NBJet_test,weights_train,weights_test= train_test_split(inX, outY, NJet, NBJet,weights, test_size=0.2)
inX_train, inX_test,outY_train,outY_test,weights_train,weights_test= train_test_split(inX, outY,weights, test_size=0.2)
# outY_train=outY_train.reshape((outY_train.shape[0],1))
# print (outY_train)

# model = models.getRhoRegModel(regRate=1e-7,activation='selu',dropout=0.05,nDense=5,nNodes=100)
# model = models.getRhoRegModelFlat(regRate=1e-7,activation='sigmoid',dropout=0.05,nDense=3,nNodes=3500)
# model = models.getRhoRegModelFlat(regRate=1e-7,activation='sigmoid',dropout=0.05,nDense=6,nNodes=500)
# model = models.getRhoRegModelFlat(regRate=1e-7,activation='sigmoid',dropout=0.05,nDense=4,nNodes=1500)#GOOD WORKING!!! corr 0.853 RMS<0.1
# model = models.getRhoRegModelFlat(regRate=1e-7,activation='sigmoid',dropout=0.01,nDense=3,nNodes=500)
model = models.getRhoRegModelFlat(regRate=1e-7,activation='sigmoid',dropout=0.01,nDense=3,nNodes=500)#GOOD WORKING, shown in talk
# model = models.getRhoRegModelFlat(regRate=1e-6,activation='sigmoid',dropout=0.15,nDense=4,nNodes=300)
# model = models.getRhoRegModelFlat(regRate=1e-6,activation='sigmoid',dropout=0.15,nDense=4,nNodes=200)
# model = models.getRhoRegModelFlat(regRate=1e-6,activation='sigmoid',dropout=0.15,nDense=4,nNodes=500)
# model = models.getRhoRegModelFlat(regRate=1e-8,activation='sigmoid',dropout=0.001,nDense=10,nNodes=250)

# model.load_weights("rhoTemp/trainedRhoModel")
# optimizer = tf.keras.optimizers.Adam(lr=0.0001)
optimizer = tf.keras.optimizers.Adam(lr=0.001)


model.compile(optimizer=optimizer, loss='mean_squared_error',metrics=['mean_absolute_error','mean_squared_error'])
# model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(),metrics=['mean_absolute_error','mean_squared_error'])
# model.compile(optimizer=optimizer, loss=tf.keras.losses.logcosh,metrics=['mean_absolute_error','mean_squared_error'])
# model.compile(optimizer=optimizer, loss='mean_absolute_error',metrics=['mean_absolute_error','mean_squared_error'])
# model.compile(optimizer=optimizer, loss=helpers.mmd_loss,metrics=['mean_absolute_error','mean_squared_error'])


callbacks=[]
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3)
earlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=10)


callbacks.append(reduce_lr)
callbacks.append(earlyStop)


fit = model.fit(
        # [inJetX_train,inOtherX_train],
        inX_train,
        outY_train,
        # validation_data=(X_test, y_test),
        sample_weight=weights_train,
        validation_split=0.25,
        batch_size=4096,
        epochs=150,
        shuffle=False,
        callbacks=callbacks,
        verbose=1)
# model.load_weights("rhoTemp/trainedRhoModel")

# model.save_weights('rhoTemp/trainedRhoModel')
# print("Saved model to disk:",'rhoTemp/trainedRhoModel')

# y_predicted = model.predict([inJetX_test,inOtherX_test])
y_predicted = model.predict(inX_test)
# outY_test=scaler.inverse_transform(outY_test)
# y_predicted=scaler.inverse_transform(y_predicted)
y_predicted = y_predicted.reshape(y_predicted.shape[0])
outY_test = outY_test.reshape(outY_test.shape[0])
# y_predicted = outY_test
# print (outY)
# print (outY_train)
# print (outY_test)

print (outY_test.shape)
print (y_predicted.shape)

outDir="rhoTemp"
model.save("rhoTemp/rhoRegModel_full.h5")
tf.keras.backend.clear_session()
# --- convert model to estimator and save model as frozen graph for c++

# has to be done to get a working frozen graph in c++
tf.keras.backend.set_learning_phase(0)

# model has to be re-loaded
model = tf.keras.models.load_model("rhoTemp/rhoRegModel_full.h5")

print('inputs: ', [input.op.name for input in model.inputs])
print('outputs: ', [output.op.name for output in model.outputs])

# from Utils.KerasToTensorflow import freeze_session
frozen_graph = helpers.freeze_session(tf.keras.backend.get_session(), output_names=[out.op.name for out in model.outputs])
tf.train.write_graph(frozen_graph, outDir+'/', 'rhoRegModel_full.pbtxt', as_text=True)
tf.train.write_graph(frozen_graph, outDir+'/', 'rhoRegModel_full.pb', as_text=False)



import ROOT
import matplotlib.pyplot as plt


f=plt.figure()
# plt.subplot(211)
values,bins,patches = plt.hist(outY_test,bins=10,range=(0,1),label="true",alpha=0.5)
values2,bins2,patches2 = plt.hist(y_predicted,bins=10,range=(0,1),label="reco",alpha=0.5)
plt.legend(loc="best")
plt.ylabel('Events')
plt.xlabel('rho')
outFolderName="rhoTemp/"
f.savefig(outFolderName+"eval_rho.pdf")


style.style2d()
s = style.style2d()
histo = ROOT.TH2F( "bla", ";rho reco;#rho true - #rho reco", 500, 0, 1, 1000, -1, 1 )
histo.SetDirectory(0)
histoRecoGen = ROOT.TH2F( "bla2", ";#rho reco;#rho true", 500, 0, 1, 500, 0, 1 )
histoRecoGen.SetDirectory(0)
histoRecoGen2 = ROOT.TH2F( "bla2", ";reco bin;true bin", 8, 0, 1, 8, 0, 1 )
histoRecoGen2.SetDirectory(0)
print (outY_test.shape,y_predicted.shape)
for rhotrue, rhopred in zip(outY_test, y_predicted):
    diff = rhotrue-rhopred
    # print (rhotrue,diff)
    histo.Fill(rhotrue,diff)
    histoRecoGen.Fill(rhopred,rhotrue)
    histoRecoGen2.Fill(rhopred,rhotrue)

c=ROOT.TCanvas("c1","c1",800,800)

histo.GetXaxis().SetRangeUser(350,1500)
histo.GetYaxis().SetRangeUser(-1200,1200)
histo.SetStats(0)
histo.Draw("colz")
c.SaveAs(outFolderName+"testDiff.pdf")
c.Clear()

# histoRecoGen.GetXaxis().SetRangeUser(350,1500)
# histoRecoGen.GetYaxis().SetRangeUser(350,1500)
histoRecoGen.Draw("colz")
corrLatex = ROOT.TLatex()
corrLatex.SetTextSize(0.65 * corrLatex.GetTextSize())
corrLatex.DrawLatexNDC(0.65, 0.85, str(np.round(histoRecoGen.GetCorrelationFactor(),3)))
c.SaveAs(outFolderName+"testGenReco.pdf")
c.Clear()
histoRecoGen2.Scale(1./histoRecoGen2.Integral())
ROOT.gStyle.SetPaintTextFormat("1.2f");
histoRecoGen2.SetStats(0)
histoRecoGen2.Draw("colz text")
c.SaveAs(outFolderName+"testResp.pdf")
c.Clear()




style.style1d()
s = style.style1d()
c=ROOT.TCanvas()
Xnb=20
Xr1=0.
Xr2=1.
dXbin=(Xr2-Xr1)/((Xnb));
# titleRMSVsGen_ptTop_full ="RMS (#rho_{true} - #rho_{reco}) vs #rho_{true}; #rho_{true}; RMS"
# titleMeanVsGen_ptTop_full ="Mean (#rho_{true} - #rho_{reco}) vs #rho_{true}; #rho_{true};Mean"
titleRMSVsGen_ptTop_full ="; #rho_{true};RMS"
titleMeanVsGen_ptTop_full ="; #rho_{true};Mean"
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
# h_RMSVsGen_.SaveAs(outFolderName+"testRMS.root")
# h_meanVsGen_.SaveAs(outFolderName+"testMEAN.root")
# c=ROOT.TCanvas("c1","c1",800,800)
h_RMSVsGen_.SetStats(0)
h_RMSVsGen_.Draw()
c.SaveAs(outFolderName+"testRMS.pdf")
c.Clear()
h_meanVsGen_.SetStats(0)
h_meanVsGen_.Draw()
c.SaveAs(outFolderName+"testMEAN.pdf")
