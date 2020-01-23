from utils import helpers, models
from sklearn.model_selection import train_test_split

# import numpy as np
import tensorflow as tf
# from keras import optimizers,losses
# import energyflow as ef
# from energyflow.archs import PFN
# from energyflow.datasets import qg_jets
# from energyflow.utils import data_split,ptyphims_from_p4s,ms_from_p4s
# import tensorflow as tf
import keras.backend as K
K.tensorflow_backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))))
# train, val, test = 75000, 10000, 15000

# X, y = qg_jets.load(train + val + test)
inputFile=".bkp_input/input/total_CP5.root"
# inputFile="input/ee_ttbarsignalplustau_fromDilepton_TuneCP5.root"
# inputFile="/nfs/dust/cms/user/sewuchte/analysisFWK/dnn/CMSSW_9_4_13_patch2/src/TopAnalysis/Configuration/analysis/diLeptonic/plainTree_2016/Nominal/ee/ee_ttbarsignalplustau_fromDilepton.root"
treeName="plainTree_rec_step8"
# normVal = 170.
normValX = 340.
# normValX = 1.
normValY = 1.
# normVal = 1.
# xData, yData = helpers.loadData2(inputFile, treeName, loadGenInfo=False, zeroJetPadding=True, normX=False, normY=False, normValue=normVal)
# xData, yData = helpers.loadData3(inputFile, treeName, loadGenInfo=False, zeroJetPadding=True, normX=False, normY=False, normValue=normVal)
# xData, yData = helpers.loadData3(inputFile, treeName, loadGenInfo=False, zeroJetPadding=True, normX=False, normY=False, normValue=normVal)
# xData, yData = helpers.loadData(inputFile, treeName, loadGenInfo=False, zeroJetPadding=True, normX=False, normY=False, normValue=normVal)
# xData1,xData2,xData3, yData = helpers.loadDataNew(inputFile, treeName)
xData, yData = helpers.loadDataNew2(inputFile, treeName,zeroJetPadding=True)
# frac = int(0.35*len(xData))
# xData = xData[:frac:]
# yData = yData[:frac:]
# x_, x_test, y_, y_test = train_test_split(xData, yData, test_size=0.33)
# x_train, x_val, y_train, y_val = train_test_split(x_, y_, test_size=0.33)
x_train, x_test, y_train, y_test = train_test_split(xData, yData, test_size=0.33)
# x_train1, x_test1,x_train2, x_test2,x_train3, x_test3, y_train, y_test = train_test_split(xData1,xData2,xData3, yData, test_size=0.25)



x_train = x_train.reshape((x_train.shape[0],38*4))
x_test = x_test.reshape((x_test.shape[0],38*4))
# x_val = x_val.reshape((x_val.shape[0],44))
# x_train = x_train.reshape((x_train.shape[0],36))
# x_test = x_test.reshape((x_test.shape[0],36))
# x_val = x_val.reshape((x_val.shape[0],44))
# x_train1 = x_train1.reshape((x_train1.shape[0],4*3))
# x_test1 = x_test1.reshape((x_test1.shape[0],4*3))
# x_train2 = x_train2.reshape((x_train2.shape[0],4*25))
# x_test2 = x_test2.reshape((x_test2.shape[0],4*25))
# x_train3 = x_train3.reshape((x_train3.shape[0],4*10))
# x_test3 = x_test3.reshape((x_test3.shape[0],4*10))
# # x_val = x_val.reshape((x_val.shape[0],44))

from sklearn.preprocessing import StandardScaler
# scaler_x = StandardScaler()
# scaler_x.fit(x_train)
# scaler_x1 = StandardScaler()
# scaler_x1.fit(x_train1)
# scaler_x2 = StandardScaler()
# scaler_x2.fit(x_train2)
# scaler_x3 = StandardScaler()
# scaler_x3.fit(x_train3)

# x_train = scaler_x.transform(x_train)
# x_test = scaler_x.transform(x_test)
# x_val = scaler_x1.transform(x_val1)
# x_train1 = scaler_x1.transform(x_train1)
# x_test1 = scaler_x1.transform(x_test1)
# # x_val1 = scaler_x1.transform(x_val1)
# x_train2 = scaler_x2.transform(x_train2)
# x_test2 = scaler_x2.transform(x_test2)
# # x_val2 = scaler_x2.transform(x_val2)
# x_train3 = scaler_x3.transform(x_train3)
# x_test3 = scaler_x3.transform(x_test3)
# # x_val3 = scaler_x3.transform(x_val3)

# x_train=x_train/340.
# x_test=x_test/340.
x_train=x_train/normValX
x_test=x_test/normValX

x_train = x_train.reshape((x_train.shape[0],38,4))
x_test = x_test.reshape((x_test.shape[0],38,4))
# x_train1 = x_train1.reshape((x_train1.shape[0],3,4))
# x_test1 = x_test1.reshape((x_test1.shape[0],3,4))
# x_train2 = x_train2.reshape((x_train2.shape[0],25,4))
# x_test2 = x_test2.reshape((x_test2.shape[0],25,4))
# x_train3 = x_train3.reshape((x_train3.shape[0],10,4))
# x_test3 = x_test3.reshape((x_test3.shape[0],10,4))
# x_val = x_val.reshape((x_val.shape[0],11,4))


scaler_y = StandardScaler()
scaler_y.fit(y_train)

# y_train=y_train/340.
# y_test=y_test/340.
y_train=y_train/normValY
y_test=y_test/normValY

# y_train = scaler_y.transform(y_train)
# y_test = scaler_y.transform(y_test)
# y_val = scaler_y.transform(y_val)

# regRate = 1e-07
# activation = 'selu'
# dropout = 0.3
# batchSize = 1024
# nNodes=500
# nDense=8
# normVal = 1.
# learningRate = 0.0001
regRate = 1e-07
activation = 'selu'
# activation = 'relu'
# activation = 'tanh'
dropout = 0.03
# batchSize = 1024
batchSize = 2048
# batchSize = 4096
nNodes=500
# nNodes=100
nDense=8
# normVal = 1.
learningRate = 0.0001
# nNodes=350
# nDense=4
# normVal = 1.
# learningRate = 0.00005
# regRate = 1e-07
# activation = 'softplus'
# dropout = 0.1
# batchSize = 128
# nNodes=200
# nDense=5
# normVal = 1
# learningRate = 0.001
model = models.getJetModel(regRate=regRate, activation=activation, dropout=dropout, nNodes=nNodes, nDense=nDense)
# model = models.lstmModel(regRate=regRate, activation=activation, dropout=dropout, nNodes=nNodes, nDense=nDense)




# results = helpers.doTrainingNew(x_train1,x_train2,x_train3, y_train, model, "output/lstm/", learning_rate=learningRate, batch_size=batchSize, epochs=45)
results = helpers.doTraining(x_train, y_train, model, "output/lstm/", learning_rate=learningRate, batch_size=batchSize, epochs=1000)




y_predicted_norm = model.predict(x_test)
# y_predicted = scaler_y.inverse_transform(y_predicted_norm)
# y_test_nonNorm = scaler_y.inverse_transform(y_test)
# y_predicted = y_predicted_norm*340.
# y_test_nonNorm=y_test *340.
y_predicted = y_predicted_norm*normValY
y_test_nonNorm=y_test *normValY


helpers.doPredictionPlots(y_test_nonNorm, y_predicted, "output/lstm/")
# helpers.doPredictionPlots(y_test, y_predicted_norm, "output/lstm/")
# helpers.doPredictionPlots2(y_test_nonNorm, y_predicted, "output/dnn/")
