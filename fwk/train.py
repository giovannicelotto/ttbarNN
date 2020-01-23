# from utils.imports import *
# from utils.helpers import *
# import utils.helpers
# import utils.models
from utils import helpers, models
from sklearn.model_selection import train_test_split

def main(regRate, activation, dropout, learningRate, batchSize, nNodes, nDense, normVal):
	# inputFile="input/total_wCP5.root"
	# inputFile="/nfs/dust/cms/user/sewuchte/analysisFWK/dnn/CMSSW_9_4_13_patch2/src/TopAnalysis/Configuration/analysis/diLeptonic/plainTree_2016/Nominal/ee/ee_ttbarsignalplustau_fromDilepton.root"
	inputFile="input/total_CP5.root"
	treeName="plainTree_rec_step8"

	# xData, yData = helpers.loadData(inputFile, treeName, loadGenInfo=False, zeroJetPadding=True, normX=True, normY=True, normValue=normVal)
	xData, yData = helpers.loadData2(inputFile, treeName, loadGenInfo=False, zeroJetPadding=True, normX=False, normY=False, normValue=normVal)

	x_train, x_test, y_train, y_test = train_test_split(xData, yData, test_size=0.33)

	model = models.getModel(regRate=regRate, activation=activation, dropout=dropout, nNodes=nNodes, nDense=nDense)

	# model.load_weights("output/pretrain/model/pretrainedModel")
	# model.load_weights("output/final/trainedModel2")
	# print("Loaded pretrained model from disk:",'output/pretrain/model/pretrainedModel')
	print("")
	print ("Train using following parameters:")
	print ("regRate ", regRate)
	print ("activation ", activation)
	print ("dropout ", dropout)
	print ("learningRate ", learningRate)
	print ("batchSize ", batchSize)
	print ("nNodes ", nNodes)
	print ("nDense ", nDense)

	results = helpers.doTraining(x_train, y_train, model, "output/final/", learning_rate=learningRate, batch_size=batchSize, epochs=25, splitFactor=0.33)

	# model.save_weights('output/final/trainedModel3')
	# print("Saved model to disk:",'output/final/trainedModel3')


	y_predicted_norm = model.predict(x_test)
	# y_predicted = helpers.normalizeData(y_predicted_norm, normVal, invert=True)
	y_predicted = y_predicted_norm
	# y_test_nonNorm = helpers.normalizeData(y_test, normVal, invert=True)

	y_test_nonNorm = y_test

	# helpers.doPredictionPlots(y_test_nonNorm, y_predicted, "output/final/")
	helpers.doPredictionPlots2(y_test_nonNorm, y_predicted, "output/final/")


if __name__ == '__main__':
	regRate = 1e-07
	activation = 'selu'
	# dropout = 0.15
	dropout = 0.3
	# learningRate = 0.00001
	batchSize = 1024
	nNodes=500
	nDense=8
	normVal = 340.
	# regRate = 1e-07
	# activation = 'softplus'
	# dropout = 0.25
	learningRate = 0.0001
	# batchSize = 2048
	# nNodes=2500
	# nDense=5
	# normVal = 340.

	main(regRate, activation, dropout, learningRate, batchSize, nNodes, nDense, normVal)
