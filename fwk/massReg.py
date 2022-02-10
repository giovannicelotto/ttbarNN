from utils import helpers, models,style
from sklearn.model_selection import train_test_split

import os

import numpy as np
import tensorflow as tf
from keras import optimizers,losses
import keras.backend as K
K.tensorflow_backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))))


def loadData(inPathFolder = "rhoInput/years", year = "2017", additionalName = "_massReg", testFraction = 0.2, overwrite = False,
            withBTag = False, withCharge = False):

    print ("loadData for year "+year+" from "+inPathFolder+"/flat_[..]"+additionalName+".npy")
    print ("\t overwrite = "+str(overwrite))
    print ("\t testFraction = "+str(testFraction))
    print ("\t withBTag = "+str(withBTag))
    print ("\t withCharge = "+str(withCharge))

    loadData = False
    if not os.path.exists(inPathFolder+"/flat_inX"+additionalName+".npy"):
        loadData = True
    else:
        loadData = overwrite

    if loadData:
        print ("\t need to load from root file with settings: nJets = "+str(2)+"; max Events = all")
        inX, outY, weights = helpers.loadMassDataFlat(inPathFolder+"/total"+year+".root", "miniTree", nJets = 2, maxEvents = 0,
                                                    withBTag=withBTag, withCharge=withCharge)

        inX=np.array(inX)
        outY=np.array(outY)
        weights=np.array(weights)

        np.save(inPathFolder+"/flat_inX"+additionalName+".npy", inX)
        np.save(inPathFolder+"/flat_outY"+additionalName+".npy", outY)
        np.save(inPathFolder+"/flat_weights"+additionalName+".npy", weights)

    inX = np.load(inPathFolder+"/flat_inX"+additionalName+".npy")
    outY = np.load(inPathFolder+"/flat_outY"+additionalName+".npy")
    weights = np.load(inPathFolder+"/flat_weights"+additionalName+".npy")

    # from sklearn.preprocessing import *
    outY=300./outY
    outY = outY.reshape((outY.shape[0], 1))

    inX_train, inX_test, outY_train, outY_test, weights_train, weights_test= train_test_split(inX, outY, weights, test_size = testFraction)

    print ("\t data splitted succesfully")
    print ("loadData end")

    print("N training events: ",inX_train.shape[0])
    print(inX_train.shape)
    # print(weights_train.shape)
    # print(inX_test.shape)
    # print(outY_test.shape)
    # print(weights_test.shape)

    return inX_train, inX_test, outY_train, outY_test, weights_train, weights_test


def doEvaluationPlots(yTest, yPredicted, weightTest, year = "", outFolder = "massOutput/"):

    import ROOT
    import matplotlib.pyplot as plt

    outDir = outFolder+"/"+year+"/"
    if not os.path.exists(outDir):
        os.makedirs(outDir)

    f = plt.figure()
    # values, bins, patches = plt.hist(yTest, bins=10, range=(0,1), label="true", alpha=0.5)
    # values2, bins2, patches2 = plt.hist(yPredicted, bins=10, range=(0,1), label="reco", alpha=0.5)
    values, bins, patches = plt.hist(yTest, bins=100, range=(200,1000), label="true", alpha=0.5)
    values2, bins2, patches2 = plt.hist(yPredicted, bins=100, range=(0,500), label="reco", alpha=0.5)
    plt.legend(loc = "best")
    plt.ylabel('Events')
    plt.xlabel('mass')
    f.savefig(outDir+"eval_mass.pdf")

    style.style2d()
    s = style.style2d()
    histo = ROOT.TH2F( "bla", ";mass reco;#mass true - #mass reco", 400, 200, 1000, 1000, -500, 500 )
    histo.SetDirectory(0)
    histoRecoGen = ROOT.TH2F( "bla2", ";#mass reco;#mass true", 400, 200, 400, 1000, 200, 1000 )
    histoRecoGen.SetDirectory(0)
    # histoRecoGen2 = ROOT.TH2F( "bla2", ";reco bin;true bin", 8, 0, 1, 8, 0, 1 )
    # histoRecoGen2.SetDirectory(0)
    # print (yTest.shape, yPredicted.shape)
    for masstrue, masspred, weight in zip(yTest, yPredicted, weightTest):
        diff = masstrue-masspred
        histo.Fill(masstrue, diff, weight)
        histoRecoGen.Fill(masspred, masstrue, weight)
        # histoRecoGen2.Fill(masspred, masstrue, weight)

    c=ROOT.TCanvas("c1","c1",800,800)

    histo.GetXaxis().SetRangeUser(350,1500)
    histo.GetYaxis().SetRangeUser(-1200,1200)
    histo.SetStats(0)
    histo.Draw("colz")
    c.SaveAs(outDir+"GenMassDiff2d.pdf")
    c.Clear()

    histoRecoGen.Draw("colz")
    corrLatex = ROOT.TLatex()
    corrLatex.SetTextSize(0.65 * corrLatex.GetTextSize())
    corrLatex.DrawLatexNDC(0.65, 0.85, str(np.round(histoRecoGen.GetCorrelationFactor(),3)))
    c.SaveAs(outDir+"GenMass2d.pdf")
    c.Clear()
    # histoRecoGen2.Scale(1./histoRecoGen2.Integral())
    ROOT.gStyle.SetPaintTextFormat("1.2f");
    # histoRecoGen2.SetStats(0)
    # histoRecoGen2.Draw("colz text")
    # c.SaveAs(outDir+"resp.pdf")
    # c.Clear()

    style.style1d()
    s = style.style1d()
    c=ROOT.TCanvas()
    Xnb=20
    Xr1=0.
    # Xr2=1.
    Xr2=500.
    dXbin=(Xr2-Xr1)/((Xnb));
    titleRMSVsGen_ptTop_full ="; #mass_{true};RMS"
    titleMeanVsGen_ptTop_full ="; #mass_{true};Mean"
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
    h_RMSVsGen_.SetStats(0)
    h_RMSVsGen_.Draw()
    c.SaveAs(outDir+"rms.pdf")
    c.Clear()
    h_meanVsGen_.SetStats(0)
    h_meanVsGen_.Draw()
    c.SaveAs(outDir+"mean.pdf")




def doFitForBaysian(inX_train, outY_train, weights_train,
                    inX_test, outY_test, weights_test,
                    learningRate = 0.001, batchSize = 4096,
                    dropout = 0.05,
                    nDense = 3,
                    nNodes = 500,
                    regRate = 1e-7):

    nDense = int(nDense)
    nNodes = int(nNodes)
    batchSize = int(batchSize)
    indexActivation = int(indexActivation)

    activations = ["sigmoid", "relu", "softmax", "selu", "softplus"]

    model = models.getMassRegModelFlat(regRate = regRate, activation = 'sigmoid', dropout = dropout, nDense = nDense,
                                      nNodes = nNodes, inputDim = 17, outputActivation = 'sigmoid')#GOOD WORKING, shown in talk
                                      # nNodes = nNodes, inputDim = 27, outputActivation = 'sigmoid')#GOOD WORKING, shown in talk
    optimizer = tf.keras.optimizers.Adam(lr = learningRate)

    model.compile(optimizer=optimizer, loss='mean_squared_error',metrics=['mean_absolute_error','mean_squared_error'])

    callbacks=[]
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', patience=3)
    earlyStop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=10)

    callbacks.append(reduce_lr)
    callbacks.append(earlyStop)

    fit = model.fit(
            inX_train,
            outY_train,
            sample_weight = weights_train,
            validation_split = 0.25,
            batch_size = batchSize,
            epochs = 150,
            shuffle = False,
            callbacks = callbacks,
            verbose = 0)
            # verbose = 1)

    # model.save_weights('rhoTemp/trainedRhoModel')
    # print("Saved model to disk:",'rhoTemp/trainedRhoModel')

    score = model.evaluate(
                        inX_test,
                        outY_test,
                        sample_weight = weights_test,
                        batch_size = batchSize,
                        verbose = 0)
    # y_predicted = model.predict(inX_test)
    # y_predicted = y_predicted.reshape(y_predicted.shape[0])
    # outY_test = outY_test.reshape(outY_test.shape[0])
    print('Test loss:', score[0])
    print('Test MSE:', score[1])

    # return 1./score[1]
    return 1./score[0]


def doBaysianOptim():



    inX_train, inX_test, outY_train, outY_test, weights_train, weights_test = loadData(
                                    inPathFolder = "rhoInput/years", year = "",
                                    # additionalName = "_3JetKin", testFraction = 0.2, overwrite = False)
                                    additionalName = "_3JetKin_correctNJet_WithBAndCharge", testFraction = 0.2, overwrite = False)



    from functools import partial

    fit_with_partial = partial(doFitForBaysian,
                            inX_train, outY_train, weights_train,
                            inX_test, outY_test, weights_test)



    from bayes_opt import BayesianOptimization

    # Bounded region of parameter space
    pbounds = {'batchSize': (128, 10000),
                'regRate': (1e-7, 1e-3),
                'learningRate': (1e-4, 1e-2),
                'dropout': (1e-4, 5e-1),
                'nDense': (2, 20),
                'nNodes': (28, 2048)
              }

    print ("doBaysianOptim")

    optimizer = BayesianOptimization(
        f=fit_with_partial,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )

    # optimizer.maximize(init_points=10, n_iter=10,)
    optimizer.maximize(init_points=30, n_iter=30,)


    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))

    print(optimizer.max)



def doTrainingAndEvaluation():
    inX_train, inX_test, outY_train, outY_test, weights_train, weights_test = loadData(
                                    inPathFolder = "rhoInput/years", year = "2017",
                                    # additionalName = "_3JetKin_correctNJet_WithBAndCharge", testFraction = 0.2,
                                    # additionalName = "_3JetKin_correctNJet", testFraction = 0.2,
                                    additionalName = "_massReg", testFraction = 0.2,
                                    # overwrite = False, withBTag = True, withCharge = True)
                                    overwrite = False, withBTag = False, withCharge = False)

    print (inX_train)
    print (inX_train.shape)
    print (inX_train[0])
    print (outY_train)
    print (outY_train.shape)


    learningRate = 0.005
    # learningRate = 0.007034
    batchSize = 12096
    # dropout = 0.05
    dropout = 0.3673
    nDense = 5
    nNodes = 500
    regRate = 1e-7
    # regRate = 0.000179
    activation = 'relu'
    # activation = 'sigmoid'
    outputActivation = 'linear'
    # outputActivation = 'sigmoid'

    model = models.getMassRegModelFlat(regRate = regRate, activation = activation, dropout = dropout, nDense = nDense,
                                      nNodes = nNodes, inputDim = 17, outputActivation = outputActivation)#GOOD WORKING, shown in talk
                                      # nNodes = nNodes, inputDim = 26, outputActivation = outputActivation)#GOOD WORKING, shown in talk
    optimizer = tf.keras.optimizers.Adam(lr = learningRate)

    model.compile(optimizer=optimizer, loss='mean_squared_error',metrics=['mean_absolute_error','mean_squared_error'])

    callbacks=[]
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', patience=3)
    earlyStop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=5)

    callbacks.append(reduce_lr)
    callbacks.append(earlyStop)

    fit = model.fit(
            inX_train,
            outY_train,
            sample_weight = weights_train,
            validation_split = 0.25,
            batch_size = batchSize,
            epochs = 150,
            shuffle = False,
            callbacks = callbacks,
            verbose = 1)

    y_predicted = model.predict(inX_test)
    y_predicted = y_predicted.reshape(y_predicted.shape[0])
    y_predicted = 300./y_predicted
    outY_test = outY_test.reshape(outY_test.shape[0])
    outY_test = 300./outY_test

    outDir = "massOutput/"
    # outName = "rhoRegModel_optim_2017"
    outName = "massRegModel"
    # doEvaluationPlots(outY_test, y_predicted , weights_test, year = "2017", outFolder = outDir)
    doEvaluationPlots(outY_test, y_predicted , weights_test, year = "2017", outFolder = outDir)
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    model.save(outDir+outName+".h5")
    tf.keras.backend.clear_session()
    tf.keras.backend.set_learning_phase(0)
    model = tf.keras.models.load_model(outDir+outName+".h5")
    print('inputs: ', [input.op.name for input in model.inputs])
    print('outputs: ', [output.op.name for output in model.outputs])

    frozen_graph = helpers.freeze_session(tf.keras.backend.get_session(), output_names=[out.op.name for out in model.outputs])
    tf.train.write_graph(frozen_graph, outDir+'/', outName+'.pbtxt', as_text=True)
    tf.train.write_graph(frozen_graph, outDir+'/', outName+'.pb', as_text=False)


def main():
    # doBaysianOptim()
    doTrainingAndEvaluation()


if __name__ == "__main__":
    main()


# For later use

# outDir="rhoOutput/"
# if not os.path.exists(outDir):
#     os.makedirs(outDir)
# model.save(outDir+"rhoRegModel_full_inclSyst.h5")
# tf.keras.backend.clear_session()
# # --- convert model to estimator and save model as frozen graph for c++
#
# # has to be done to get a working frozen graph in c++
# tf.keras.backend.set_learning_phase(0)
#
# # model has to be re-loaded
# model = tf.keras.models.load_model(outDir+"rhoRegModel_full_inclSyst.h5")
#
# print('inputs: ', [input.op.name for input in model.inputs])
# print('outputs: ', [output.op.name for output in model.outputs])
#
# frozen_graph = helpers.freeze_session(tf.keras.backend.get_session(), output_names=[out.op.name for out in model.outputs])
# tf.train.write_graph(frozen_graph, outDir+'/', 'rhoRegModel_full_inclSyst.pbtxt', as_text=True)
# tf.train.write_graph(frozen_graph, outDir+'/', 'rhoRegModel_full_inclSyst.pb', as_text=False)
