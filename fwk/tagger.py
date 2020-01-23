from utils import helpers, models
from sklearn.model_selection import train_test_split

import numpy as np
import tensorflow as tf
from keras import optimizers,losses
# import energyflow as ef
# from energyflow.archs import PFN
# from energyflow.datasets import qg_jets
# from energyflow.utils import data_split,ptyphims_from_p4s,ms_from_p4s
# import tensorflow as tf
import keras.backend as K
K.tensorflow_backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))))

# inputFile="taggerInput/total.root"
inputFile="rhoInput/total.root"
# inputFile="taggerInput/totalHiggs.root"
# inputFile="taggerInput/totalBoth.root"
treeName="miniTree"
normVal = 1.
# path, treeName, nJets=20, maxEvents=0, normX=False, normY=False, normValue=170.
# inJetX,inOtherX,outY = helpers.loadTaggerData(inputFile, treeName, nJets=20, maxEvents=1000000, normX=False, normY=False, normValue=normVal)
# inJetX,inOtherX,outY = helpers.loadTaggerData(inputFile, treeName, nJets=20, maxEvents=0, normX=False, normY=False, normValue=normVal)
# inJetX,inOtherX,outY,NJet,NBJet = helpers.loadTaggerDataBinned(inputFile, treeName, nJets=10, maxEvents=5000000, normX=False, normY=False, normValue=normVal)
# inJetX,inOtherX,outY,NJet,NBJet = helpers.loadTaggerDataBinned(inputFile, treeName, nJets=6, maxEvents=500000, normX=False, normY=False, normValue=normVal)
# inX,outY,NJet,NBJet = helpers.loadTaggerDataBinned(inputFile, treeName, nJets=6, maxEvents=900000, normX=False, normY=False, normValue=normVal)
# inX,outY,NJet,NBJet = helpers.loadTaggerDataBinned(inputFile, treeName, nJets=6, maxEvents=0, normX=False, normY=False, normValue=normVal)
# inX,outY,NJet,NBJet,weights = helpers.loadTaggerDataBinned(inputFile, treeName, nJets=4, maxEvents=0, normX=False, normY=False, normValue=normVal)

# inJetX=np.array(inJetX)
# inOtherX=np.array(inOtherX)

# inX=np.array(inX)
# outY=np.array(outY)
# NJet=np.array(NJet)
# NBJet=np.array(NBJet)
# weights=np.array(weights)

# np.save("inJetX.npy", inJetX)
# np.save("inOtherX.npy", inOtherX)
# np.save("outY.npy", outY)
# np.save("eval_inJetX.npy", inJetX)
# np.save("eval_inOtherX.npy", inOtherX)

# np.save("taggerInput/eval_inX.npy", inX)
# np.save("taggerInput/eval_outY.npy", outY)
# np.save("taggerInput/eval_njet.npy", NJet)
# np.save("taggerInput/eval_nbjet.npy", NBJet)
# np.save("taggerInput/eval_weights.npy", weights)
# np.save("taggerInput/evalHiggs_inX.npy", inX)
# np.save("taggerInput/evalHiggs_outY.npy", outY)
# np.save("taggerInput/evalHiggs_njet.npy", NJet)
# np.save("taggerInput/evalHiggs_nbjet.npy", NBJet)
# np.save("taggerInput/evalBoth_inX.npy", inX)
# np.save("taggerInput/evalBoth_outY.npy", outY)
# np.save("taggerInput/evalBoth_njet.npy", NJet)
# np.save("taggerInput/evalBoth_nbjet.npy", NBJet)

#
# print (inJetX)
# print (inOtherX)
# print (outY)

# inJetX = np.load("inJetX.npy")
# inOtherX = np.load("inOtherX.npy")
# outY = np.load("outY.npy")
# inJetX = np.load("eval_inJetX.npy")
# inOtherX = np.load("eval_inOtherX.npy")
inX = np.load("taggerInput/eval_inX.npy")
outY = np.load("taggerInput/eval_outY.npy")
NJet = np.load("taggerInput/eval_njet.npy")
NBJet = np.load("taggerInput/eval_nbjet.npy")
weights = np.load("taggerInput/eval_weights.npy")
# inX = np.load("taggerInput/evalHiggs_inX.npy")
# outY = np.load("taggerInput/evalHiggs_outY.npy")
# NJet = np.load("taggerInput/evalHiggs_njet.npy")
# NBJet = np.load("taggerInput/evalHiggs_nbjet.npy")
# inX = np.load("taggerInput/evalBoth_inX.npy")
# outY = np.load("taggerInput/evalBoth_outY.npy")
# NJet = np.load("taggerInput/evalBoth_njet.npy")
# NBJet = np.load("taggerInput/evalBoth_nbjet.npy")

# print (inJetX.shape)
# print (inOtherX.shape)
print (inX.shape)
print (inX[0])
print (outY[0])
print (outY.shape)
print (NJet.shape)
print (NBJet.shape)
print (weights.shape)
print (weights[0])


# inJetX_train, inJetX_test, inOtherX_train, inOtherX_test,outY_train,outY_test, NJet_train, NJet_test,NBJet_train,NBJet_test= train_test_split(inJetX, inOtherX,outY,NJet,NBJet, test_size=0.33)
inX_train, inX_test,outY_train,outY_test, NJet_train, NJet_test,NBJet_train,NBJet_test,weights_train,weights_test= train_test_split(inX,outY,NJet,NBJet,weights, test_size=0.25)

# print (outY[0])
# print (outY[1])

# optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate,momentum=0.1)
#~ optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
#~ optimizer = tf.keras.optimizers.SGD(lr=learning_rate)

# metrics = [
#     tf.keras.metrics.accuracy
# ]

# model = models.getTaggerModel(regRate=1e-5,activation='selu',dropout=0.05,nDense=3,nNodes=100)
# model = models.getTaggerModelFlat(regRate=1e-5,activation='relu',dropout=0.05,nDense=5,nNodes=150) #well working
# model = models.getTaggerModelFlat(regRate=1e-5,activation='relu',dropout=0.05,nDense=5,nNodes=450)
model = models.getTaggerModelFlat(regRate=1e-5,activation='relu',dropout=0.05,nDense=5,nNodes=450)

# model.load_weights("taggerTemp/trainedTaggerModel")
# model.load_weights("taggerTemp/trainedTaggerModel_top")
# model.load_weights("taggerTemp/trainedTaggerModel_BOTH")
# optimizer = tf.keras.optimizers.Adam(lr=0.0001)
optimizer = tf.keras.optimizers.Adam(lr=0.001)


model.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=['accuracy'])


callbacks=[]

# reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
# 	                              patience=5, min_lr=learning_rate/10.)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3)
earlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5)


callbacks.append(reduce_lr)
callbacks.append(earlyStop)


fit = model.fit(
        # [inJetX_train,inOtherX_train],
        [inX_train],
        outY_train,
        # validation_data=(X_test, y_test),
        validation_split=0.25,
        # sample_weight=weights_train,
        # batch_size=2048,
        batch_size=4096,
        epochs=150,
        shuffle=False,
        callbacks=callbacks,
        verbose=1)

# model.save_weights('taggerTemp/trainedTaggerModel_BOTH')
# model.save_weights('taggerTemp/trainedTaggerModel_top')
# print("Saved model to disk:",'taggerTemp/trainedTaggerModel')



outDir="taggerTemp"
model.save("taggerTemp/taggerModel_full.h5")
tf.keras.backend.clear_session()
# --- convert model to estimator and save model as frozen graph for c++

# has to be done to get a working frozen graph in c++
tf.keras.backend.set_learning_phase(0)

# model has to be re-loaded
model = tf.keras.models.load_model("taggerTemp/taggerModel_full.h5")

print('inputs: ', [input.op.name for input in model.inputs])
print('outputs: ', [output.op.name for output in model.outputs])

# from Utils.KerasToTensorflow import freeze_session
frozen_graph = helpers.freeze_session(tf.keras.backend.get_session(), output_names=[out.op.name for out in model.outputs])
tf.train.write_graph(frozen_graph, outDir+'/', 'taggerModel_full.pbtxt', as_text=True)
tf.train.write_graph(frozen_graph, outDir+'/', 'taggerModel_full.pb', as_text=False)




# y_predicted = model.predict([inJetX_test,inOtherX_test])
y_predicted = model.predict([inX_test])
# print (y_predicted.shape)
# print (outY_test.shape)
# numTrueJets=[]
# numRecoJets=[]
acc=[]
accDict={
    (2,0): [],
    (2,1): [],
    (2,2): [],
    (2,99): [],
    (3,99): [],
    (3,0): [],
    (3,1): [],
    (3,2): [],
    (3,3): [],
    (4,99): [],
    (4,0): [],
    (4,1): [],
    (4,2): [],
    (4,3): [],
    (4,4): [],
    (5,99): [],
    (5,0): [],
    (5,1): [],
    (5,2): [],
    (5,3): [],
    (5,4): [],
    (5,5): [],
    (6,99): [],
    (6,0): [],
    (6,1): [],
    (6,2): [],
    (6,3): [],
    (6,4): [],
    (6,5): [],
    (6,6): [],
}
bkgShape=[]
topShape=[]

passed=0
rejected=0

for reco,true,nj,nbj in zip(y_predicted,outY_test,NJet_test,NBJet_test):
    # a = np.where(reco > 0.5)
    # reco2=np.round(reco,2)
    # print (sum(true))
    if sum(true)>1:
        t=np.argsort(true)[-2:]
        t_=np.argsort(true)
        r=np.argsort(reco)[-2:]
        r_=np.argsort(reco)
        # print (r)
        # print (t)
        # if reco[r[0]]> 0.45 and reco[r[1]]>0.35:
        if reco[r[0]]> 0.0 and reco[r[1]]>0.0:
            passed+=1
            if r[0] in t:
                if r[1] in t:

                    for key in accDict:
                        njet=key[0]
                        nbjet=key[1]
                        if nj==njet:
                            if nbjet==nbj:
                                accDict[key].append(1.)
                            if nbjet==99.:
                                accDict[key].append(1.)
                    acc.append(1.)
            else:
                acc.append(0.)
                for key in accDict:
                    njet=key[0]
                    nbjet=key[1]
                    if nj==njet:
                        if nbjet==nbj:
                            accDict[key].append(0.)
                        if nbjet==99.:
                            accDict[key].append(0.)
        else: rejected+=1
        # for i in range(6):
        for i in range(4):
            if t_[i]>0.5:
                topShape.append(reco[r_[i]])
            else:
                bkgShape.append(reco[r_[i]])


    # print (reco2)
    # print (true)
    # print ("")
    # numRecoJets.append(sum(np.where(reco > 0.5)))
    # numTrueJets.append(sum(np.where(true > 0.5)))
# print (np.mean(numTrueJets))
# print (np.mean(numRecoJets))
print ("passed",passed)
print ("rejected",rejected)

print ("Overall accuracy to find both quarks")
print(np.mean(acc))


# for key in accDict:
    # print (key)
    # print (np.mean(accDict[key]))

matrix=[[np.mean(accDict[(2,0)]),np.mean(accDict[(2,1)]),np.mean(accDict[(2,2)]),0.,0.,0.,0.,np.mean(accDict[(2,99)])],
        [np.mean(accDict[(3,0)]),np.mean(accDict[(3,1)]),np.mean(accDict[(3,2)]),np.mean(accDict[(3,3)]),0.,0.,0.,np.mean(accDict[(3,99)])],
        [np.mean(accDict[(4,0)]),np.mean(accDict[(4,1)]),np.mean(accDict[(4,2)]),np.mean(accDict[(4,3)]),np.mean(accDict[(4,4)]),0.,0.,np.mean(accDict[(4,99)])],
        [np.mean(accDict[(5,0)]),np.mean(accDict[(5,1)]),np.mean(accDict[(5,2)]),np.mean(accDict[(5,3)]),np.mean(accDict[(5,4)]),np.mean(accDict[(5,5)]),0.,np.mean(accDict[(5,99)])],
        [np.mean(accDict[(6,0)]),np.mean(accDict[(6,1)]),np.mean(accDict[(6,2)]),np.mean(accDict[(6,3)]),np.mean(accDict[(6,4)]),np.mean(accDict[(6,5)]),np.mean(accDict[(6,6)]),np.mean(accDict[(6,99)])]
        ]
# print (matrix.shape)
print ("Accuracy matrix to find both quarks")
print (matrix)
matrix=np.array(matrix)
matrix=np.round(matrix,2)
njlabels = ["2j", "3j", "4j", "5j","6j"]
nbjlabels = ["0", "1", "2",
           "3", "4", "5", "6","incl"]

import matplotlib.pyplot as plt
f=plt.figure()
weights = np.ones_like(bkgShape)/float(len(bkgShape))
plt.hist(bkgShape,weights=weights,bins=20,label="bkg",alpha=0.5,normed=False)
weights2 = np.ones_like(topShape)/float(len(topShape))
plt.hist(topShape,weights=weights2,bins=20,label="true",alpha=0.5,normed=False)
# f.savefig("taggerTemp/evalTEST.pdf")
f.savefig("taggerTemp/evalTESTHiggs.pdf")

fig, ax = plt.subplots()
im = ax.imshow(matrix)
# We want to show all ticks...
ax.set_xticks(np.arange(len(nbjlabels)))
ax.set_yticks(np.arange(len(njlabels)))
# ... and label them with the respective list entries
ax.set_xticklabels(nbjlabels)
ax.set_yticklabels(njlabels)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
         # Loop over data dimensions and create text annotations.
for i in range(len(njlabels)):
    for j in range(len(nbjlabels)):
        text = ax.text(j, i, matrix[i, j],
                       ha="center", va="center", color="w")

ax.set_title("Accuracy")
fig.tight_layout()
# fig.savefig("taggerTemp/NJETmatrixHiggsOnly.pdf")
fig.savefig("taggerTemp/NJETmatrixTopOnly.pdf")
# fig.savefig("taggerTemp/NJETmatrix.pdf")
