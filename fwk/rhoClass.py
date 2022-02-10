from utils import helpers, models,style
from sklearn.model_selection import train_test_split
from utils.helpers import *
import os
import numpy as np
import tensorflow as tf
from keras import optimizers,losses
import keras.backend as K
from bayes_opt import BayesianOptimization
from bayes_opt import SequentialDomainReductionTransformer
from bayes_opt.util import load_logs
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from functools import partial
# import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import shap
from sklearn import preprocessing
from scipy.stats.stats import pearsonr
from scipy.stats.mstats import gmean
from scipy.stats import chisquare
from scipy.spatial.distance import jensenshannon
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if(len(physical_devices)>0):
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.compat.v1.disable_eager_execution()

import ROOT

def loadData(inPathFolder = "rhoInput/years", year = "2017", additionalName = "_3JetKinPlusRecoSolRight", testFraction = 0.2, overwrite = False, withBTag = False, withCharge = False, pTEtaPhiMode=False, maxEvents = None):

    print ("loadData for year "+year+"/"+" from "+inPathFolder+year+"/flat_[..]"+additionalName+".npy")
    print ("\t overwrite = "+str(overwrite))
    print ("\t testFraction = "+str(testFraction))
    print ("\t withBTag = "+str(withBTag))
    print ("\t withCharge = "+str(withCharge))

    loadData = False
    if not os.path.exists(inPathFolder+year+"/flat_inX"+additionalName+".npy"):
        loadData = True
    else:
        loadData = overwrite

    if loadData:
        print ("\t need to load from root file with settings: nJets = "+str(3)+"; max Events = all")
        inX, outY, weights, lkrRho, krRho = helpers.loadRegressionData(inPathFolder+year+"/", "miniTree", nJets = 3, maxEvents = 0 if maxEvents==None else maxEvents, withBTag=withBTag, withCharge=withCharge, pTEtaPhiMode=pTEtaPhiMode)
        inX=np.array(inX)
        outY=np.array(outY)
        weights=np.array(weights)
        lkrRho=np.array(lkrRho)
        krRho=np.array(krRho)
        np.save(inPathFolder+year+"/flat_inX"+additionalName+".npy", inX)
        np.save(inPathFolder+year+"/flat_outY"+additionalName+".npy", outY)
        np.save(inPathFolder+year+"/flat_weights"+additionalName+".npy", weights)
        np.save(inPathFolder+year+"/flat_lkrRho"+additionalName+".npy", lkrRho)
        np.save(inPathFolder+year+"/flat_krRho"+additionalName+".npy", krRho)

    inX = np.load(inPathFolder+year+"/flat_inX"+additionalName+".npy")
    outY = np.load(inPathFolder+year+"/flat_outY"+additionalName+".npy")
    weights = np.load(inPathFolder+year+"/flat_weights"+additionalName+".npy")
    lkrRho = np.load(inPathFolder+year+"/flat_lkrRho"+additionalName+".npy")
    krRho = np.load(inPathFolder+year+"/flat_krRho"+additionalName+".npy")

    if maxEvents is not None:
        inX = inX[:maxEvents]
        outY = outY[:maxEvents]
        weights = weights[:maxEvents]
        lkrRho = lkrRho[:maxEvents]
        krRho = krRho[:maxEvents]

    outY = outY.reshape((outY.shape[0], 1))

    # transform y to uniform
    scaler = preprocessing.StandardScaler().fit(outY)

    inX_train, inX_test, outY_train, outY_test, weights_train, weights_test, lkrRho_train, lkrRho_test, krRho_train, krRho_test = train_test_split(inX, outY, weights, lkrRho, krRho, test_size = testFraction)

    print ("\t data splitted succesfully")
    print ("loadData end")
    print("N training events: ",inX_train.shape[0])

    return inX_train, inX_test, outY_train, outY_test, weights_train, weights_test,lkrRho_train, lkrRho_test, krRho_train, krRho_test, scaler

def reweight(outY_test, outY_train, weights_test, weights_train):

    print (outY_test, outY_train, weights_test, weights_train)

    proc_test = outY_test
    outY_test_ = []
    for i in range(len(outY_test)):
        # print (outY_test[i])
        if(outY_test[i]==0):
            outY_test_.append([1,0,0,0])
        elif(outY_test[i]==1):
            outY_test_.append([0,1,0,0])
        elif(outY_test[i]==2):
            outY_test_.append([0,0,1,0])
        elif(outY_test[i]==3):
            outY_test_.append([0,0,0,1])
        else:
            print (outY_test[i][0],"SHOULD NOT BE HERE!!")
    outY_test = np.array(outY_test_)

    proc_train = outY_train
    outY_train_ = []
    for i in range(len(outY_train)):
        # print (outY_train[i])
        if(outY_train[i]==0):
            outY_train_.append([1,0,0,0])
        elif(outY_train[i]==1):
            outY_train_.append([0,1,0,0])
        elif(outY_train[i]==2):
            outY_train_.append([0,0,1,0])
        elif(outY_train[i]==3):
            outY_train_.append([0,0,0,1])
        else:
            print (outY_train[i][0],"SHOULD NOT BE HERE!!")
    outY_train = np.array(outY_train_)

    integral=0.
    counts=[0. for i in range(4)]
    weights=[]
    arrays=[
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1],
    ]
    # reweight all to be 1/third
    for i in range(len(outY_train)):
        # print (outY_train[i],np.argmax(outY_train[i]))
        counts[np.argmax(outY_train[i])] = counts[np.argmax(outY_train[i])] +abs(weights_train[i])
        # counts[np.argmax(outY_train[i])] = counts[np.argmax(outY_train[i])] +abs(1.)
    print ("event sums signal0, signal1, signal2, signal3",counts)
    integral = np.sum(counts)
    for i in range(4):
        weights.append(1./(counts[i]/integral))

    # weights_train_original=weights_train
    weights_train_=[]
    for y, w in zip(outY_train, weights_train):
        if np.argmax(y)==0:
            # weights_train_.append(abs(w*weights[np.argmax(y)]))
            weights_train_.append(abs(1.0*w*weights[np.argmax(y)]))
            # weights_train_.append(abs(weights[np.argmax(y)]))
        elif np.argmax(y)==1:
            # weights_train_.append(abs(1.5*w*weights[np.argmax(y)]))
            # weights_train_.append(abs(1.2*w*weights[np.argmax(y)]))
            weights_train_.append(abs(1.0*w*weights[np.argmax(y)]))
            # weights_train_.append(abs(weights[np.argmax(y)]))
        elif np.argmax(y)==2:
            # weights_train_.append(abs(10.*w*weights[np.argmax(y)]))
            # weights_train_.append(abs(0.9*w*weights[np.argmax(y)]))
            weights_train_.append(abs(1.0*w*weights[np.argmax(y)]))
            # weights_train_.append(abs(w*weights[np.argmax(y)]))
            # weights_train_.append(abs(weights[np.argmax(y)]))
        elif np.argmax(y)==3:
            # weights_train_.append(abs(10.*w*weights[np.argmax(y)]))
            # weights_train_.append(abs(0.9*w*weights[np.argmax(y)]))
            weights_train_.append(abs(1.0*w*weights[np.argmax(y)]))
            # weights_train_.append(abs(w*weights[np.argmax(y)]))
            # weights_train_.append(abs(weights[np.argmax(y)]))
        else:
            print ("ERRRROOOOR")
    weights_train=np.array(weights_train_)

    # count to verify
    counts_after=[0. for i in range(4)]
    for i in range(len(outY_train)):
        counts_after[np.argmax(outY_train[i])] = counts_after[np.argmax(outY_train[i])] +weights_train[i]
    print ("after first weighting event sums signal0, signal1, signal2, signal3",counts_after)

    weight_m_train = np.mean(weights_train)
    weights_train = weights_train/weight_m_train
    #
    print ("divide by mean of weights:",weight_m_train)
    print ("median of weights is:",np.median(weights_train))

    return outY_test, outY_train, weights_test, weights_train, proc_test, proc_train

def doEvaluationPlots(yTest, yPredicted, weightTest, lkrRho, krRho, year = "", outFolder = "rhoOutput/"):

    # import ROOT
    # import matplotlib.pyplot as plt

    outDir = outFolder+"/"+year+"/"
    if not os.path.exists(outDir):
        os.makedirs(outDir)


    style.style2d()
    s = style.style2d()
    # binning = [0.0, 0.65, 0.75, 1.0]
    # binning = [0.0, 0.45, 0.75, 1.0]
    binning = [0.0, 0.3, 0.45, 0.7, 1.0]
    from array import array
    ar_bins = array("d", binning)
    histo = ROOT.TH2F( "resp class", ";rho true;#rho reco", 4, ar_bins, 4, ar_bins )
    histo1d = ROOT.TH1F("reco",";rho reco", 4, ar_bins)
    histo1dTrue = ROOT.TH1F("true",";rho true", 4, ar_bins)
    histo1dKR = ROOT.TH1F("kr",";rho kr", 4, ar_bins)
    histo1dLKR = ROOT.TH1F("lkr",";rho lkr", 4, ar_bins)
    confmatrix          =   ROOT.TH2F("conf",";predicted label;true label",4,0,4,4,0,4)
    # histo = ROOT.TH2F( "resp class", ";rho reco bin;#rho true bin", 11, 0.,11., 11,0.,11. )
    histo.SetDirectory(0)
    histo1d.SetDirectory(0)
    histo1dTrue.SetDirectory(0)

    def getValue(bin):
        if bin==0: return 0.1
        elif bin==1: return 0.35
        elif bin==2: return 0.6
        elif bin==3: return 0.85
        else: print("WTF NOT HERE")

    for trueclass, predclass, weight,lkr,kr in zip(yTest, yPredicted, weightTest,lkrRho,krRho):
        # print trueclass, predclass,weight
        # histo.Fill(trueclass, predclass, weight)
        true = np.argmax(trueclass)
        pred = np.argmax(predclass)
        # print (true, pred)
        histo.Fill(getValue(true), getValue(pred), weight)
        histo1d.Fill(getValue(pred), weight)
        histo1dTrue.Fill(getValue(true), weight)
        if kr>0.:
            histo1dKR.Fill(kr, weight)
        if lkr>0.:
            histo1dLKR.Fill(lkr, weight)
        confmatrix.Fill(pred, true, weight)

    c=ROOT.TCanvas("c1","c1",800,800)

    # histo.GetXaxis().SetRangeUser(350,1500)
    # histo.GetYaxis().SetRangeUser(-1200,1200)
    histo.SetStats(0)
    hResp = histo.Clone()
    histo.Scale(1./(histo.Integral()))

    from array import array
    nbinsx = histo.GetNbinsX()
    nbinsy = histo.GetNbinsY()

    ma = np.empty((nbinsx,nbinsy))
    ma2 = np.empty((nbinsx,nbinsy))
    for iBinX in range(1,nbinsx+1):
        for iBinY in range(1,nbinsy+1):
            ma[iBinX-1,iBinY-1]=histo.GetBinContent(iBinX,iBinY)
            ma2[iBinY-1,iBinX-1]=histo.GetBinContent(iBinX,iBinY)
    print ("Condition number = ",np.linalg.cond(ma))


    histo.Scale(100.)

    ROOT.gStyle.SetPaintTextFormat("2.2f")
    histo.SetMarkerSize(0.7)
    histo.GetZaxis().SetTitle("Transition Probability [%]")
    # style.setPalette("bird")
    ROOT.gStyle.SetPalette(ROOT.kThermometer)
    # ROOT.gStyle.SetPaintTextFormat("1.2f");
    histo.Draw("colz text")
    c.SaveAs(outDir+"class_response.pdf")

    c.Clear()
    style.style1d()
    s = style.style1d()
    # maximum = max(histo1dTrue.GetMaximum(),histo1d.GetMaximum())
    maximum = max(histo1dTrue.GetMaximum(),histo1d.GetMaximum(), histo1dKR.GetMaximum(), histo1dLKR.GetMaximum())
    minimum = min(histo1dTrue.GetMinimum(),histo1d.GetMinimum(), histo1dKR.GetMinimum(), histo1dLKR.GetMinimum())
    histo1d.SetMaximum(maximum*10.5)
    histo1dTrue.SetMaximum(maximum*10.5)
    histo1dKR.SetMaximum(maximum*10.5)
    histo1dLKR.SetMaximum(maximum*10.5)
    histo1d.SetMinimum(minimum/10.5)
    histo1dTrue.SetMinimum(minimum/10.5)
    histo1dKR.SetMinimum(minimum/10.5)
    histo1dLKR.SetMinimum(minimum/10.5)
    histo1dTrue.SetLineColor(ROOT.kBlack)
    histo1d.Draw("hist")
    histo1d.SetLineColor(ROOT.kRed)
    histo1dTrue.Draw("hist same")
    histo1dKR.SetLineColor(ROOT.kBlue)
    histo1dKR.Draw("hist same")
    histo1dLKR.SetLineColor(ROOT.kGreen+1)
    histo1dLKR.Draw("hist same")

    c.SetLogy(1)

    l = ROOT.TLegend(0.2,0.7,0.4,0.9)
    l.AddEntry(histo1dTrue, "truth")
    l.AddEntry(histo1d, "classifier")
    l.AddEntry(histo1dKR, "KR")
    l.AddEntry(histo1dLKR, "LKR")
    l.Draw()

    c.SaveAs(outDir+"class_shape.pdf")
    style.style1d()
    s = style.style1d()

    c.Clear()

    c.SetLogx(False)
    c.SetLogy(False)

    g_purity = purityStabilityGraph(hResp, 0)
    g_stability = purityStabilityGraph(hResp, 1)

    # Styling axis
    histo_ = hResp.ProjectionX(hResp.GetName()+"_px")
    histo_.SetAxisRange(0., 1.1, "Y")
    histo_.GetYaxis().SetTitle("a.u.")
    ROOT.gPad.SetRightMargin(0.05)
    histo_.Draw("AXIS")

    nBins = histo_.GetNbinsX()
    g_relStatError = ROOT.TGraphErrors(nBins)
    for iBin in range(1,nBins+1):
        value = histo_.GetBinContent(iBin)+1e-20
        error = histo_.GetBinError(iBin)
        bin = histo_.GetXaxis().GetBinCenter(iBin)
        binW = histo_.GetXaxis().GetBinWidth(iBin)
        g_relStatError.SetPoint(iBin-1, bin, abs(error/value))
        # g_relStatError.SetPointError(iBin-1, binW/2., error)


    # Styling graphs
    setGraphStyle(g_purity,       1, 1, 2, 21, 1, 1)
    setGraphStyle(g_stability,    1, 2, 2, 20, 2, 1)
    setGraphStyle(g_relStatError, 1, 3, 2, 19, 3, 1)

    # Drawing graphs
    g_purity.Draw("P0same")
    g_stability.Draw("P0same")
    g_relStatError.Draw("P0same")
    ROOT.gPad.Update()

    # Adding a legend
    leg = ROOT.TLegend(.56, .75, .94, .9)
    leg.AddEntry(g_purity, "Purity", "p")
    leg.AddEntry(g_stability, "Stability", "p")
    leg.AddEntry(g_relStatError, "Rel. stat. Err.", "p")
    leg.Draw()

    c.SaveAs(outDir+"class_pse.pdf")

    c.Clear()

    style.style2d()
    s = style.style2d()
    c=ROOT.TCanvas()

    c.SetLogx(False)
    c.SetLogy(False)

    confmatrixNorm = confmatrix.Clone()
    for binX in range(1, confmatrixNorm.GetNbinsX()+1):
        for binY in range(1, confmatrixNorm.GetNbinsY()+1):
            intTrues = confmatrix.Integral(1, confmatrixNorm.GetNbinsX()+1, binY, binY)
            nomVal = confmatrix.GetBinContent(binX, binY)/intTrues
            confmatrixNorm.SetBinContent(binX, binY,nomVal)
    ROOT.gStyle.SetPaintTextFormat("1.3f")
    confmatrixNorm.GetZaxis().SetRangeUser(0.,1.)
    # c.SetMaximum(1.)
    # c.SetMinimum(0.)
    ROOT.gStyle.SetPalette(ROOT.kThermometer)
    confmatrixNorm.Draw("colz text")
    c.SaveAs(outDir+"confmatrixNorm.pdf")

    c.Clear()
    confmatrix.Draw("colz text")
    ROOT.gStyle.SetPaintTextFormat("2.1f")
    confmatrix.SetMarkerSize(0.7)
    confmatrix.GetZaxis().SetTitle("Events")
    ROOT.gStyle.SetPalette(ROOT.kThermometer)
    c.SaveAs(outDir+"confmatrix.pdf")

def uncertaintyBinomial(pass_, all_):
    return (1./all_)*np.sqrt(pass_ - pass_*pass_/all_)


def purityStabilityGraph(h2d, type):
    import ROOT
    nBins = h2d.GetNbinsX()

    graph = ROOT.TGraphErrors(nBins)

    # Calculating each point of graph for each diagonal bin
    # for(int iBin = 1; iBin<=nBins; ++iBin) {
    for iBin in range(1,nBins+1):
        diag = h2d.GetBinContent(iBin, iBin)
        reco = h2d.Integral(iBin, iBin, 1, -1)+1e-30
        gen = h2d.Integral(1, -1, iBin, iBin)+1e-30

        value = diag/reco if (type == 0 ) else diag/gen
        error = uncertaintyBinomial(diag, reco) if (type == 0) else uncertaintyBinomial(diag, gen)

        bin = h2d.GetXaxis().GetBinCenter(iBin)
        binW = h2d.GetXaxis().GetBinWidth(iBin)

        graph.SetPoint(iBin-1, bin, value)
        graph.SetPointError(iBin-1, binW/2., error)

    return graph



def setGraphStyle(graph, line=-1, lineColor=-1, lineWidth=-1, marker=-1, markerColor=-1, markerSize=-1, fill=-1, fillColor=-1):
    import ROOT
    if(line != -1): graph.SetLineStyle(line)
    if(lineColor != -1): graph.SetLineColor(lineColor)
    if(lineWidth != -1): graph.SetLineWidth(lineWidth)

    if(fill != -1): graph.SetFillStyle(fill)
    if(fillColor != -1): graph.SetFillColor(fillColor)

    if(marker != -1): graph.SetMarkerStyle(marker)
    if(markerColor != -1): graph.SetMarkerColor(markerColor)
    if(markerSize != -1): graph.SetMarkerSize(markerSize)





def doFitForBaysian(inX_train, outY_train, weights_train, inX_test, outY_test, weights_test, learningRate = 0.001, batchSize = 4096, dropout = 0.05, nDense = 3, nNodes = 500, regRate = 1e-7):

    # inX_train=np.delete(inX_train,24,1)
    # inX_test=np.delete(inX_test,24,1)

    nDense = int(nDense)
    nNodes = int(nNodes)
    batchSize = int(batchSize)
    # indexActivation = int(indexActivation)
    # indexOutputActivation = int(indexOutputActivation)

    activations = ["sigmoid", "relu", "softmax", "selu", "softplus"]
    # outputActivations = ["sigmoid","linear"]

    # model = models.getRhoRegModelFlat(regRate=1e-7,activation='sigmoid',dropout=0.05,nDense=4,nNodes=1500)
    # model = models.getRhoRegModelFlat(regRate=1e-7, activation='sigmoid', dropout=0.05, nDense=3, nNodes=500)#
    model = models.getRhoRegModelFlat(regRate = regRate, activation = 'relu', dropout = dropout, nDense = nDense,
    # model = models.getRhoRegModelFlat(regRate = regRate, activation = 'selu', dropout = dropout, nDense = nDense,
                                      # nNodes = nNodes, inputDim = 21, outputActivation = 'sigmoid')
                                      # nNodes = nNodes, inputDim = 36, outputActivation = 'sigmoid')
                                      # nNodes = nNodes, inputDim = 27, outputActivation = 'linear')
                                      # nNodes = nNodes, inputDim = 34, outputActivation = 'linear')
                                      nNodes = nNodes, inputDim = inX_train.shape[1], outputActivation = 'linear')
                                      # nNodes = nNodes, inputDim = 24, outputActivation = 'sigmoid')
    # model.load_weights("rhoTemp/trainedRhoModel")
    # optimizer = tf.keras.optimizers.Adam(lr=0.0001)
    # optimizer = tf.keras.optimizers.Adam(lr = 0.001)
    optimizer = tf.keras.optimizers.Adam(lr = learningRate)

    model.compile(optimizer=optimizer, loss='mean_squared_error',metrics=['mean_absolute_error','mean_squared_error'])
    # model.compile(optimizer=optimizer, loss=tf.keras.losses.Huber(),metrics=['mean_absolute_error','mean_squared_error'])
    # model.compile(optimizer=optimizer, loss=helpers.correlation_coefficient_loss,metrics=['mean_absolute_error','mean_squared_error'])

    callbacks=[]
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', patience=3)
    # earlyStop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=10)
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
    y_predicted = model.predict(inX_test)
    y_predicted = y_predicted.reshape(y_predicted.shape[0])
    outY_test = outY_test.reshape(outY_test.shape[0])
    # y_predicted = y_predicted.reshape(y_predicted.shape[0])
    # outY_test = outY_test.reshape(outY_test.shape[0])
    print('Test loss:', score[0])
    print('Test MSE:', score[1])

    from scipy.stats.stats import pearsonr

    corr = pearsonr(outY_test, y_predicted)
    print(corr[0])

    # return 1./score[1]
    # return 1./score[0]
    if np.isnan(corr[0]):
        return 0.
    else:
        return corr[0]


def doBaysianOptim():



    inX_train, inX_test, outY_train, outY_test, weights_train, weights_test = loadData(
                                    # inPathFolder = "rhoInput/years", year = "2017",
                                    inPathFolder = "rhoInput/years", year = "2018",
                                    # additionalName = "_3JetKin", testFraction = 0.2, overwrite = False)
                                    # additionalName = "_3JetKinPlusRecoSol", testFraction = 0.2, overwrite = False, withBTag = False, withCharge = False)
                                    # additionalName = "_3JetKinPlusRecoSol", testFraction = 0.5, overwrite = False, withBTag = False, withCharge = False,
                                    # additionalName = "_3JetKinMore", testFraction = 0.35, overwrite = False, withBTag = True, withCharge = True,
                                    # additionalName = "_3JetKinMoreWithoutNu", testFraction = 0.35, overwrite = False, withBTag = True, withCharge = True,
                                    # additionalName = "_3JetKinWithoutNu", testFraction = 0.35, overwrite = False, withBTag = True, withCharge = True,
                                    # additionalName = "_3JetKinWithNu", testFraction = 0.35, overwrite = False, withBTag = True, withCharge = True,
                                    additionalName = "_2018_PtEtaPhiM_3JetKinWithNu", testFraction = 0.35, overwrite = False, withBTag = True, withCharge = False, pTEtaPhiMode=True,
                                    # additionalName = "_FR2less_3JetKinWithNu", testFraction = 0.35, overwrite = False, withBTag = True, withCharge = True,
                                    # maxEvents = 2000000)
                                    maxEvents = None)


    # inX_train=np.delete(inX_train,[12,13,14,15,16,17,18,19,20,21,32,34],1)
    # inX_test=np.delete(inX_test,[12,13,14,15,16,17,18,19,20,21,32,34],1)
    inX_train=np.delete(inX_train,[12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,41,43,44,45,46,47,48,49,51,52,53,54,55,56,57,58,59,60,61,62],1)
    inX_test=np.delete(inX_test,  [12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,41,43,44,45,46,47,48,49,51,52,53,54,55,56,57,58,59,60,61,62],1)

    from functools import partial

    fit_with_partial = partial(doFitForBaysian,
                            inX_train, outY_train, weights_train,
                            inX_test, outY_test, weights_test)



    from bayes_opt import BayesianOptimization
    from bayes_opt import SequentialDomainReductionTransformer
    # Bounded region of parameter space
    pbounds = {'batchSize': (1000, 10000),
                'regRate': (1e-7, 1e-4),
                'learningRate': (1e-4, 1e-2),
                'dropout': (1e-4, 5e-1),
                'nDense': (2, 10),
                'nNodes': (50, 1000)
              }

    bounds_transformer = SequentialDomainReductionTransformer()

    print ("doBaysianOptim")

    optimizer = BayesianOptimization(
        f=fit_with_partial,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
        bounds_transformer=bounds_transformer,
    )

    # optimizer.maximize(init_points=10, n_iter=10,)
    # optimizer.maximize(init_points=20, n_iter=50,kappa=8, alpha=1e-3)
    optimizer.maximize(init_points=20, n_iter=50,kappa=8, alpha=1e-3)


    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))

    print(optimizer.max)



def doTrainingAndEvaluation(inPathFolder, additionalName, year, tokeep = None, outFolder="outFolder_DEFAULTNAME/", modelName = "rhoClassModel_preUL04_moreInputs_Pt30_madgraph_cutSel"):
    inX_train, inX_test, outY_train, outY_test, weights_train, weights_test,lkrRho_train, lkrRho_test, krRho_train, krRho_test, scaler = loadData(
                                    inPathFolder = inPathFolder, year = year,
                                    additionalName = additionalName, testFraction = 0.3,
                                    overwrite = False, withBTag = True, withCharge = False, pTEtaPhiMode=True,
                                    maxEvents = None)
                                    # maxEvents = 500000)

    feature_names, inX_train, inX_test = helpers.getReducedFeatureNamesAndInputs(inX_train, inX_test, tokeep=tokeep)

    # binning = [0.0, 0.65, 0.75, 1.0]
    binning = [0.0, 0.3, 0.45, 0.7, 1.0]
    class_names=["bin"+str(i) for i in range(5)]

    print (outY_train.shape)
    print (outY_train[0])

    outY_train_=[]
    outY_test_=[]
    for y in outY_train:
        y0 = y[0]
        # print (y0)
        # if y0<0.65:
        if y0<0.3:
            outY_train_.append(0)
        elif y0<0.45:
            outY_train_.append(1)
        elif y0<0.7:
            outY_train_.append(2)
        else:
            outY_train_.append(3)
        # print (outY_train_)

    for y in zip(outY_test):
        y0 = y[0]
        # if y0<0.65:
        if y0<0.3:
            outY_test_.append(0)
        elif y0<0.45:
            outY_test_.append(1)
        elif y0<0.7:
            outY_test_.append(2)
        else:
            outY_test_.append(3)

    # inX_train=np.array(inX_train_)
    outY_train=np.array(outY_train_)
    outY_test=np.array(outY_test_)
    # weights_train=np.array(weights_train_)

    # print (outY_train.shape)
    # print (outY_train[0])

    # from sklearn.preprocessing import LabelEncoder,OneHotEncoder
    # encoder = LabelEncoder()
    # encoder.fit(inX_train)
    # encoded_Y = encoder.transform(Y)
    # dummy_y = np_utils.to_categorical(encoded_Y)
    # onehot_encoder = OneHotEncoder(sparse=False)
    # outY_train = outY_train.reshape(len(outY_train), 1)
    # onehot_encoded = onehot_encoder.fit_transform(outY_train)
    # print(onehot_encoded)

    # outY_train = onehot_encoded


    weights_train_original = weights_train
    weights_test_original = weights_test
    outY_test, outY_train, weights_test, weights_train, proc_test, proc_train = reweight(outY_test, outY_train, weights_test, weights_train)

    print (outY_train.shape)
    print (outY_train[0])

    learningRate = 0.0001
    # learningRate = 0.001
    # batchSize = 2798
    # batchSize = 256
    batchSize = 512
    dropout = 0.316
    nDense = 4
    # nNodes = 120
    nNodes = 250
    regRate = 2.32e-5
    activation = 'selu'
    outputActivation = 'softmax'


    print ("getting model")
    # model = models.getRhoClassModelFlat(regRate = regRate, activation = activation, dropout = dropout, nDense = nDense,
    #                                   nNodes = nNodes, inputDim = inX_train.shape[1], outputActivation = outputActivation)
    model = models.getClassificationModel(outputNodes = 4 ,regRate = regRate, activation = activation, dropout = dropout, nDense = nDense,
                                      nNodes = nNodes, inputDim = inX_train.shape[1], outputActivation = outputActivation)
    optimizer = tf.keras.optimizers.Adam(lr = learningRate)


    print ("compiling model")
    model.compile(optimizer=optimizer, loss=tf.keras.losses.categorical_crossentropy,metrics=['accuracy',])

    callbacks=[]
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', patience=3)
    # earlyStop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=50, restore_best_weights=True)
    # earlyStop = tf.keras.callbacks.EarlyStopping(monitor = 'val_output_loss', patience=250, restore_best_weights=True)
    # earlyStop = tf.keras.callbacks.EarlyStopping(monitor = 'val_output_loss', patience=25, restore_best_weights=True)
    # earlyStop = tf.keras.callbacks.EarlyStopping(monitor = 'val_output_loss', patience=50, restore_best_weights=True)
    earlyStop = tf.keras.callbacks.EarlyStopping(monitor = 'val_output_loss', patience=100, restore_best_weights=True)
    modelCheckpoint = tf.keras.callbacks.ModelCheckpoint(outFolder + '/model_best.h5', monitor='val_output_loss', save_best_only=True)

    # callbacks.append(reduce_lr)
    callbacks.append(earlyStop)
    callbacks.append(modelCheckpoint)

    print ("fitting model")
    fit = model.fit(
            inX_train,
            outY_train,
            sample_weight = weights_train,
            validation_split = 0.25,
            batch_size = batchSize,
            epochs = 2500,
            # epochs = 1000,
            # epochs = 50,
            # epochs = 1,
            shuffle = False,
            callbacks = callbacks,
            verbose = 1)

    y_predicted_train = model.predict(inX_train)
    y_predicted = model.predict(inX_test)

    # y_predicted = model.predict(inX_test)
    # y_predicted = [np.argmax(entry) for entry in y_predicted]
    # y_predicted = np.array(y_predicted)
    # y_predictedTrue = [np.argmax(entry) for entry in outY_test]
    # y_predictedTrue = np.array(y_predictedTrue)

    # print (y_predictedTrue.shape,y_predictedTrue)
    # print (y_predicted.shape,y_predicted)

    # ####################
    # # --- save model ---
    # ####################
    if not os.path.exists(outFolder+"/"+year):
        os.makedirs(outFolder+"/"+year)
    model.save(outFolder + '/model_last.h5')

    if not os.path.exists(outFolder):
        os.makedirs(outFolder)

    y_predicted_train = model.predict(inX_train)
    y_predicted = model.predict(inX_test)

    max_display = inX_test.shape[1]

    matplotlib.pyplot.figure(0)
    matplotlib.pyplot.plot(fit.history['accuracy'])
    matplotlib.pyplot.plot(fit.history['val_accuracy'])
    matplotlib.pyplot.title('model accuracy')
    matplotlib.pyplot.ylabel('accuracy')
    matplotlib.pyplot.xlabel('epoch')
    matplotlib.pyplot.legend(['train', 'validation'], loc='lower right')
    matplotlib.pyplot.savefig(outFolder+"/"+year+"/accuracy.pdf")
    matplotlib.pyplot.figure(1)
    matplotlib.pyplot.plot(fit.history['loss'])
    matplotlib.pyplot.plot(fit.history['val_loss'])
    matplotlib.pyplot.title('model loss')
    matplotlib.pyplot.ylabel('loss')
    matplotlib.pyplot.xlabel('epoch')
    matplotlib.pyplot.ylim(ymax = min(fit.history['loss'])*1.4, ymin = min(fit.history['loss'])*0.9)
    matplotlib.pyplot.legend(['train', 'validation'], loc='upper right')
    matplotlib.pyplot.savefig(outFolder+"/"+year+"/loss.pdf")

    doEvaluationPlots(outY_train, y_predicted_train , weights_train_original, lkrRho_train, krRho_train, year = year, outFolder = outFolder+"/train/")
    doEvaluationPlots(outY_test,  y_predicted ,       weights_test_original,   lkrRho_test,  krRho_test, year = year, outFolder = outFolder+"/test/")

    # doPlots(outY_train, y_predicted_train ,np.array(targetRho), weights_train_original, lumW_train, proc_train, year = year, outFolder = outFolder+"/train/")
    # doPlots(outY_test, y_predicted ,np.array(testRho), weights_test_original, lumW_test, proc_test, year = year, outFolder = outFolder+"/test/")

    # class_names=["signal0", "signal1", "signal2"]
    class_names=["signal0", "signal1", "signal2", "signal3"]

    for explainer, name  in [(shap.GradientExplainer(model,inX_test[:2000]),"GradientExplainer"),]:
        shap.initjs()
        print("... {0}: explainer.shap_values(X)".format(name))
        shap_values = explainer.shap_values(inX_test[:2000])
        print("... shap.summary_plot")
        matplotlib.pyplot.clf()
        shap.summary_plot(shap_values, inX_test[:2000], plot_type = "bar",
            feature_names = feature_names,
            class_names = class_names,
            max_display = max_display, plot_size = [15.0,0.4*max_display+1.5], show = False)
        matplotlib.pyplot.savefig(outFolder+"/"+year+"/"+"/shap_summary_{0}.pdf".format(name))


    if not os.path.exists(outFolder):
        os.makedirs(outFolder)

    saveModel=False
    if saveModel:
        model.save(outFolder+modelName+".h5")
        tf.keras.backend.clear_session()
        tf.keras.backend.set_learning_phase(0)
        model = tf.keras.models.load_model(outFolder+modelName+".h5")
        print('inputs: ', [input.op.name for input in model.inputs])
        print('outputs: ', [output.op.name for output in model.outputs])

        print  (model.outputs)
        print  (model.output)
        frozen_graph = helpers.freeze_session(tf.compat.v1.keras.backend.get_session(), output_names=[out.op.name for out in model.outputs])
        tf.compat.v1.train.write_graph(frozen_graph, outFolder+'/', modelName+'.pbtxt', as_text=True)
        tf.compat.v1.train.write_graph(frozen_graph, outFolder+'/', modelName+'.pb', as_text=False)
        print ("Saved model to",outFolder+'/'+modelName+'.pbtxt/.pb/.h5')


def main():
    # doBaysianOptim()
    # keepReg = [41,42,35,60,84,64,30,76,39,88]
    # keep = [41,42,0,72,5,60,64,30,31,39,3,84,1,53,69,6,68,10,35,24,76,18,19,88,73]
    # keep = [41,60,42,0,64,5,3,31,39,6,72,35,30,1,69,53,68,84,10,76]
    # keep = [42,60,0,41,72,5,31,64,35,30]
    # keep = [41,42,35,60,84,64,30,76,39,88]
    keep = [41,42,35,60,84,64,30,76,39,88, 102, 103]
    # doTrainingAndEvaluation(inPathFolder = "rhoInput/madgraph/", additionalName = "_preUL05_19_01_22", year ="FR2", tokeep = keep,
    #                         modelName = "rhoClassModel_preUL05_19_01_22", outFolder="preUL05_19_01_22/rhoClass_madgraph_v0/")
    doTrainingAndEvaluation(inPathFolder = "rhoInput/madgraph/", additionalName = "_preUL05_03_02_22", year ="FR2", tokeep = keep,
                            # modelName = "rhoClassModel_preUL05_03_02_22", outFolder="preUL05_03_02_22/rhoClass_madgraph_v0/")
                            modelName = "rhoClassModel_preUL05_03_02_22", outFolder="preUL05_03_02_22/rhoClass_madgraph_v1/")





if __name__ == "__main__":
    main()
