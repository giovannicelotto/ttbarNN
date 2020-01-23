from utils import helpers, models
# from utils.imports import *
from sklearn.model_selection import train_test_split


def main(regRate, activation, dropout, learningRate, batchSize, nNodes, nDense, normVal):
	inputFile="input/total_wCP5.root"
	# inputFile="input/ee_ttbarsignalplustau_fromDilepton.root"
	treeName="plainTree_rec_step8"

	xData, yData = helpers.loadData(inputFile, treeName, loadGenInfo=True, zeroJetPadding=True, normX=True, normY=True, normValue=normVal)

	x_train, x_test, y_train, y_test = train_test_split(xData, yData, test_size=0.5)

	model = models.getModel(regRate=regRate, activation=activation, dropout=dropout, nNodes=nNodes, nDense=nDense)

	print ("Pretrain using following parameters:")
	print ("regRate ", regRate)
	print ("activation ", activation)
	print ("dropout ", dropout)
	print ("learningRate ", learningRate)
	print ("batchSize ", batchSize)
	print ("nNodes ", nNodes)
	print ("nDense ", nDense)

	# results = helpers.doTraining(xData, yData, model, "output/pretrain/", learning_rate=learningRate, batch_size=batchSize, epochs=1, splitFactor=0.33)
	results = helpers.doTraining(x_train, y_train, model, "output/pretrain/", learning_rate=learningRate, batch_size=batchSize, epochs=20, splitFactor=0.33)

	model.save_weights('output/pretrain/model/pretrainedModel')
	print("Saved model to disk:",'output/pretrain/model/pretrainedModel')

	y_predicted_norm = model.predict(x_test)
	y_predicted = helpers.normalizeData(y_predicted_norm, normVal, invert=True)
	y_test_nonNorm = helpers.normalizeData(y_test, normVal, invert=True)

	helpers.doPredictionPlots(y_test_nonNorm, y_predicted, "output/pretrain/")

if __name__ == '__main__':
	regRate = 1e-07
	activation = 'selu'
	# dropout = 0.1
	dropout = 0.3
	learningRate = 0.00001
	batchSize = 1024
	nNodes=500
	nDense=8
	normVal = 340.

	main(regRate, activation, dropout, learningRate, batchSize, nNodes, nDense, normVal)
