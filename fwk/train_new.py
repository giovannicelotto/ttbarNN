from utils import helpers, models
from sklearn.model_selection import train_test_split

import numpy as np
import tensorflow as tf
from keras import optimizers,losses
import energyflow as ef
from energyflow.archs import PFN
from energyflow.datasets import qg_jets
from energyflow.utils import data_split,ptyphims_from_p4s,ms_from_p4s
# import tensorflow as tf
import keras.backend as K
K.tensorflow_backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))))
# train, val, test = 75000, 10000, 15000

# X, y = qg_jets.load(train + val + test)
inputFile="input/total_CP5.root"
# inputFile="/nfs/dust/cms/user/sewuchte/analysisFWK/dnn/CMSSW_9_4_13_patch2/src/TopAnalysis/Configuration/analysis/diLeptonic/plainTree_2016/Nominal/ee/ee_ttbarsignalplustau_fromDilepton.root"
treeName="plainTree_rec_step8"
# normVal = 170.
# normVal = 340.
normVal = 1.
# xData, yData = helpers.loadData2(inputFile, treeName, loadGenInfo=False, zeroJetPadding=True, normX=False, normY=False, normValue=normVal)
# xData, yData = helpers.loadData3(inputFile, treeName, loadGenInfo=False, zeroJetPadding=True, normX=False, normY=False, normValue=normVal)
xData, yData = helpers.loadData3(inputFile, treeName, loadGenInfo=False, zeroJetPadding=True, normX=False, normY=False, normValue=normVal)
# x_, x_test, y_, y_test = train_test_split(xData, yData, test_size=0.33)
# x_train, x_val, y_train, y_val = train_test_split(x_, y_, test_size=0.33)
x_train, x_test, y_train, y_test = train_test_split(xData, yData, test_size=0.33)



# x_train = x_train.reshape((x_train.shape[0],36))
# x_test = x_test.reshape((x_test.shape[0],36))
# x_val = x_val.reshape((x_val.shape[0],44))
x_train = x_train.reshape((x_train.shape[0],44))
x_test = x_test.reshape((x_test.shape[0],44))
# # x_val = x_val.reshape((x_val.shape[0],44))

from sklearn.preprocessing import StandardScaler
scaler_x = StandardScaler()
scaler_x.fit(x_train)

x_train = scaler_x.transform(x_train)
x_test = scaler_x.transform(x_test)
# x_val = scaler_x.transform(x_val)

# x_train = x_train.reshape((x_train.shape[0],11,4))
# x_test = x_test.reshape((x_test.shape[0],11,4))
# x_val = x_val.reshape((x_val.shape[0],11,4))


scaler_y = StandardScaler()
scaler_y.fit(y_train)

y_train = scaler_y.transform(y_train)
y_test = scaler_y.transform(y_test)
# y_val = scaler_y.transform(y_val)

# regRate = 1e-07
# activation = 'selu'
# dropout = 0.3
# batchSize = 1024
# nNodes=500
# nDense=8
# normVal = 340.
# learningRate = 0.0001
regRate = 1e-07
activation = 'softplus'
dropout = 0.1
batchSize = 128
nNodes=200
nDense=5
normVal = 1
learningRate = 0.001
model = models.getModel(regRate=regRate, activation=activation, dropout=dropout, nNodes=nNodes, nDense=nDense)


results = helpers.doTraining(x_train, y_train, model, "output/dnn/", learning_rate=learningRate, batch_size=batchSize, epochs=5, splitFactor=0.33)




y_predicted_norm = model.predict(x_test)
y_predicted = scaler_y.inverse_transform(y_predicted_norm)
y_test_nonNorm = scaler_y.inverse_transform(y_test)


helpers.doPredictionPlots(y_test_nonNorm, y_predicted, "output/dnn/")
# helpers.doPredictionPlots2(y_test_nonNorm, y_predicted, "output/dnn/")
