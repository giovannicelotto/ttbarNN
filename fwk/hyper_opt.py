from utils.imports import *
from utils.helpers import *
import utils.models

from pretrain import main as doPretraining
from train import main as doTraining

def main():
	arNParticles = [10]
	arRegRate = [1e-4,1e-5,1e-6,1e-7,1e-8]
	arActivation = ['relu','selu','softplus']
	arDropout = [0.01,0.05,0.1,0.2,0.3,0.6]
	arLR = [1e-1,1e-2,1e-3,1e-4]
	arBS = [128,256,512,1024,2048,4096]

	arNodes=[5,10,20,30,40,50,100,150,200,300,500,1000,2000]
	# arDense=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
	arDense=[1,2,3,4,5,6,7,8,9,10]

	nSamples=1

	out=[]

	for i in range(1,nSamples+1):
		nParticles = 10
		#~ nParticles = 11
		regRate = 1e-07
		# regRate = 1e-06
		# regRate = 1e-05
		#~ activation = 'relu'
		activation = 'selu'
		# activation = 'softplus'
		#~ dropout = 0.01
		#~ dropout = 0.05
		#~ dropout = 0.09
		dropout = 0.1
		learningRate = 0.0001
		#~ learningRate = 0.1
		# learningRate = 0.001
		# ~ learningRate = 0.01
		#~ batchSize = 2500
		# batchSize = 512
		batchSize = 1024
		#~ batchSize = 128
		# batchSize = 256
		nNodes=500
		nDense=8
		# nParticles = random.choice(arNParticles)
		# regRate = random.choice(arRegRate)
		# activation = random.choice(arActivation)
		# # dropout = random.choice(arDropout)
		# learningRate = random.choice(arLR)
		# batchSize = random.choice(arBS)
		# nNodes =random.choice(arNodes)
		# nDense =random.choice(arDense)

		# resDict=dict(index =i,
		# 			nParticles =  nParticles,
		#             regRate    = regRate,
		#             activation  = activation,
		#             dropout    = dropout,
		#             learning_rate  = learningRate,
		#             batch_size  = batchSize
		#             )

		model = models.getModel(nParticles=nParticles,regRate=regRate,activation=activation,dropout=dropout,nNodes=nNodes,nDense=nDense)

		#~ model.load_weights("training/pretrain")
		#~ model.load_weights("training/final")
		#~ print("Loaded model from disk")

		print (i,nParticles,regRate,activation,dropout,learningRate,batchSize)


		# results = doTraining(xData,yData,model,learning_rate=learningRate,batch_size=batchSize,epochs=20,splitFactor=0.33,index=i)
		# results = doTraining(xData,yData,model,learning_rate=learningRate,batch_size=batchSize,epochs=40,splitFactor=0.33,index=i)
		# results = doTraining(outNorm2,yData,model,learning_rate=learningRate,batch_size=batchSize,epochs=2,splitFactor=0.33,index=i)
		# results = doTraining(xData,outNorm2y,model,learning_rate=learningRate,batch_size=batchSize,epochs=20,splitFactor=0.33,index=i)
		# results = doTraining(xData,yData,model,learning_rate=learningRate,batch_size=batchSize,epochs=50,splitFactor=0.33,index=i)
		# results = doTraining(outNorm2,outNorm2y,model,learning_rate=learningRate,batch_size=batchSize,epochs=5,splitFactor=0.33,index=i)
		results = doTraining(outNorm2x,outNorm2y,model,learning_rate=learningRate,batch_size=batchSize,epochs=50,splitFactor=0.33,index=i)
		outDict={}
		outDict['index']=str(i)
		outDict['nParticles']=str(nParticles)
		outDict['regRate']=str(regRate)
		outDict['activation']=str(activation)
		outDict['dropout']=str(dropout)
		outDict['learning_rate']=str(learningRate)
		outDict['batch_size']=str(batchSize)
		outDict['nNodes']=str(nNodes)
		outDict['nDense']=str(nDense)

		outDict['val_loss']=str(results[1])
		outDict['mse']=str(results[2])
		outDict['val_mse']=str(results[3])
		outDict['mae']=str(results[4])
		outDict['val_mae']=str(results[5])
		out.append(outDict)

		# model.save_weights('training/final2')
		model.save_weights('training/pretrain2')
		print("Saved model to disk")

		doPrediction = (nSamples==1)

		if(doPrediction):

			X_train, X_test, y_train, y_test = train_test_split(xData, yData, test_size=0.5)

			# x_test_norm = std_scale.transform(X_test)
			x_test_norm = X_test/meanToUse
			# y_test_norm = std_scale2.transform(y_test)

			# predictions = model.predict(X_test)
			predictions = model.predict(x_test_norm)

			predictions2 = predictions * meanToUse
			# predictions2 = predictions
			# predictions2 = std_scale2.inverse_transform(predictions)

			# pred_in=y_test*meanToUse
			pred_in=y_test
			#~ pred_in=std_scale2.inverse_transform(y_test_norm)





			a = pred_in[:,0]
			a2 = pred_in[:,1]
			a3 = pred_in[:,2]
			a4 = pred_in[:,3]

			b = predictions2[:,0]
			b2 = predictions2[:,1]
			b3 = predictions2[:,2]
			b4 = predictions2[:,3]

			# nbins=280
			nbins=140

			x_bins = np.linspace(100,1500,nbins)
			y_bins = np.linspace(100,1500,nbins)

			f=plt.figure()
			plt.hist(a,bins=nbins,range=(100,1500),label="true",alpha=0.5)
			plt.hist(b,bins=nbins,range=(100,1500),label="reco",alpha=0.5)
			plt.legend(loc="best")
			plt.ylabel('Events')
			plt.xlabel('E_tt [GeV]')
			f.savefig("output/eval.pdf")
			f=plt.figure()
			plt.hist(a2,bins=nbins,range=(100,1500),label="true",alpha=0.5)
			plt.hist(b2,bins=nbins,range=(100,1500),label="reco",alpha=0.5)
			plt.legend(loc="best")
			plt.ylabel('Events')
			plt.xlabel('Px_tt [GeV]')
			f.savefig("output/eval2.pdf")
			f=plt.figure()
			plt.hist(a3,bins=nbins,range=(100,1500),label="true",alpha=0.5)
			plt.hist(b3,bins=nbins,range=(100,1500),label="reco",alpha=0.5)
			plt.legend(loc="best")
			plt.ylabel('Events')
			plt.xlabel('Py_tt [GeV]')
			f.savefig("output/eval3.pdf")
			f=plt.figure()
			plt.hist(a4,bins=nbins,range=(100,1500),label="true",alpha=0.5)
			plt.hist(b4,bins=nbins,range=(100,1500),label="reco",alpha=0.5)
			plt.legend(loc="best")
			plt.ylabel('Events')
			plt.xlabel('Pz_tt [GeV]')
			f.savefig("output/eval4.pdf")


			f=plt.figure()
			#~ plt.hist2d(a2,b2,bins=[x_bins,y_bins])
			#~ plt.hist2d(a2,b2,bins=140)
			plt.hist2d(a,b,bins=nbins,range=np.array([[100,1500],[100,1500]]))
			plt.xlabel('E_tt true [GeV]')
			plt.xlabel('E_tt reco [GeV]')
			f.savefig("output/eval5.pdf")

			f=plt.figure()
			plt.hist2d(a2,b2,bins=nbins,range=np.array([[100,1500],[100,1500]]))
			plt.xlabel('Px_tt true [GeV]')
			plt.xlabel('Px_tt reco [GeV]')
			f.savefig("output/eval6.pdf")
			f=plt.figure()
			plt.hist2d(a3,b3,bins=nbins,range=np.array([[100,1500],[100,1500]]))
			plt.xlabel('Py_tt true [GeV]')
			plt.xlabel('Py_tt reco [GeV]')
			f.savefig("output/eval7.pdf")
			f=plt.figure()
			plt.hist2d(a4,b4,bins=nbins,range=np.array([[100,1500],[100,1500]]))
			plt.xlabel('Pz_tt true [GeV]')
			plt.xlabel('Pz_tt reco [GeV]')
			f.savefig("output/eval8.pdf")
		else:
			for a in out:
				print (a)
			import json
			with open('output/hyper.json', 'w') as fout:
				json.dump(out , fout)

main()
