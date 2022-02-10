from utils import helpers, models, style
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import shap
import os
from functools import partial
import numpy as np
import tensorflow as tf
from keras import optimizers,losses
import keras.backend as K
import ROOT
# import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import scikitplot as skplt
from bayes_opt import BayesianOptimization
from bayes_opt import SequentialDomainReductionTransformer
from bayes_opt.util import load_logs
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from sklearn import preprocessing
from array import array
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if(len(physical_devices)>0):
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.compat.v1.disable_eager_execution()

os.environ['QT_QPA_PLATFORM']='offscreen'

def reweight(outY_test, outY_train, weights_test, weights_train):
    proc_test = outY_test
    outY_test_ = []
    for i in range(len(outY_test)):
        if(outY_test[i][0]==0):
            outY_test_.append([1,0,0])
        elif(outY_test[i][0]==1):
            outY_test_.append([1,0,0])
        elif(outY_test[i][0]==2):
            outY_test_.append([1,0,0])
        elif(outY_test[i][0]==3):
            outY_test_.append([1,0,0])
        elif(outY_test[i][0]==4):
            outY_test_.append([0,1,0])
        elif(outY_test[i][0]==5):
            outY_test_.append([0,0,1])
        else:
            print (outY_test[i][0],"SHOULD NOT BE HERE!!")
    outY_test = np.array(outY_test_)

    proc_train = outY_train
    outY_train_ = []
    for i in range(len(outY_train)):
        if(outY_train[i][0]==0):
            outY_train_.append([1,0,0])
        elif(outY_train[i][0]==1):
            outY_train_.append([1,0,0])
        elif(outY_train[i][0]==2):
            outY_train_.append([1,0,0])
        elif(outY_train[i][0]==3):
            outY_train_.append([1,0,0])
        elif(outY_train[i][0]==4):
            outY_train_.append([0,1,0])
        elif(outY_train[i][0]==5):
            outY_train_.append([0,0,1])
        else:
            print (outY_train[i][0],"SHOULD NOT BE HERE!!")
    outY_train = np.array(outY_train_)

    integral=0.
    counts=[0. for i in range(3)]
    weights=[]
    arrays=[
            [1,0,0],
            [0,1,0],
            [0,0,1]
    ]
    # reweight all to be 1/third
    for i in range(len(outY_train)):
        counts[np.argmax(outY_train[i])] = counts[np.argmax(outY_train[i])] +abs(weights_train[i])
    print ("event sums signal, bkg, DY",counts)
    integral = np.sum(counts)
    for i in range(3):
        weights.append(1./(counts[i]/integral))

    # weights_train_original=weights_train
    weights_train_=[]
    for y, w in zip(outY_train, weights_train):
        weights_train_.append(abs(w*weights[np.argmax(y)]))
    weights_train=np.array(weights_train_)

    # count to verify
    counts_after=[0. for i in range(3)]
    for i in range(len(outY_train)):
        counts_after[np.argmax(outY_train[i])] = counts_after[np.argmax(outY_train[i])] +weights_train[i]
    print ("after first weighting event sums signal, bkg, DY",counts_after)

    # reweight signal bins
    integral=0.
    counts=[0. for i in range(4)]
    weights=[]
    for i in range(len(outY_train)):
        if (proc_train[i][0]<4):
            counts[proc_train[i][0]] = counts[proc_train[i][0]] +weights_train[i]
    print ("signal sums:",counts)
    integral = np.sum(counts)
    for i in range(4):
        weights.append(1./(counts[i]/integral))

    weights_train_=[]
    for y, w, proc in zip(outY_train, weights_train, proc_train):
        if (proc[0]<4):
            if(proc[0]==0):
                weights_train_.append(w*weights[proc[0]]*1./4.)
            elif(proc[0]==1):
                weights_train_.append(w*weights[proc[0]]*1./4.)
            elif(proc[0]==2):
                weights_train_.append(w*weights[proc[0]]*1./4.)
            elif(proc[0]==3):
                weights_train_.append(w*weights[proc[0]]*1./4.)
            else:
                print (proc[0],"SHOULD NOT BE HERE!!")

        else:
            weights_train_.append(w)
    weights_train=np.array(weights_train_)

    # count to verify second time
    counts_after=[0. for i in range(4)]
    for i in range(len(outY_train)):
        if (proc_train[i][0]<4):
            counts_after[proc_train[i][0]] = counts_after[proc_train[i][0]] +weights_train[i]
    print ("signal sums afterwards:",counts_after,"integral",np.sum(counts_after))

    # weight_m_train = np.median(weights_train)
    weight_m_train = np.mean(weights_train)
    weights_train = weights_train/weight_m_train

    print ("divide by mean of weights:",weight_m_train)
    print ("median of weights is:",np.median(weights_train))

    return outY_test, outY_train, weights_test, weights_train, proc_test, proc_train

def loadData(inPathFolder = "classInput/years", year = "FR2", additionalName = "_3JetKinPlusRecoSolRight", testFraction = 0.2, overwrite = False, withBTag = True, withCharge = False, maxEvents = None):

    print ("loadData for year "+year+" from "+inPathFolder+year+"/"+"/flat_[..]"+additionalName+".npy")
    print ("\t overwrite = "+str(overwrite))
    print ("\t testFraction = "+str(testFraction))
    print ("\t withBTag = "+str(withBTag))
    print ("\t withCharge = "+str(withCharge))

    loadData = False
    if not os.path.exists(inPathFolder+year+"/"+"/flat_inX"+additionalName+".npy"):
        loadData = True
    else:
        loadData = overwrite

    if loadData:
        print ("\t need to load from root file with settings: nJets = "+str(3)+"; max Events = all")
        inX, outY, weights,lumW  = helpers.loadClassificationData(inPathFolder+"/"+year+"/", "miniTree", nJets = 3, maxEvents = 0 if maxEvents==None else maxEvents, withBTag=withBTag, withCharge=withCharge)

        inX=np.array(inX)
        outY=np.array(outY)
        weights=np.array(weights)
        lumW=np.array(lumW)

        np.save(inPathFolder+year+"/"+"/flat_inX"+additionalName+".npy", inX)
        np.save(inPathFolder+year+"/"+"/flat_outY"+additionalName+".npy", outY)
        np.save(inPathFolder+year+"/"+"/flat_weights"+additionalName+".npy", weights)
        np.save(inPathFolder+year+"/"+"/flat_lumW"+additionalName+".npy", lumW)

    inX = np.load(inPathFolder+year+"/"+"/flat_inX"+additionalName+".npy")
    outY = np.load(inPathFolder+year+"/"+"/flat_outY"+additionalName+".npy")
    weights = np.load(inPathFolder+year+"/"+"/flat_weights"+additionalName+".npy")
    lumW = np.load(inPathFolder+year+"/"+"/flat_lumW"+additionalName+".npy")

    if maxEvents is not None:
        inX = inX[:maxEvents]
        outY = outY[:maxEvents]
        weights = weights[:maxEvents]
        lumW = lumW[:maxEvents]


    # transform y to uniform
    scaler = preprocessing.StandardScaler()

    if testFraction>0.:
        inX_train, inX_test, outY_train, outY_test, weights_train, weights_test,lumW_train,lumW_test = train_test_split(inX, outY, weights,lumW, test_size = testFraction)
    else:
        inX_train=inX
        inX_test=[]
        outY_train=outY
        outY_test=[]
        weights_train=weights
        weights_test=[]
        lumW_train=lumW
        lumW_test=[]

    print ("\t data splitted succesfully")
    print ("loadData end")

    print("N training events: ",inX_train.shape[0])
    print(inX_train.shape)

    if testFraction>0.:
        return inX_train, inX_test, outY_train, outY_test, weights_train, weights_test,lumW_train/(1.-testFraction),lumW_test/testFraction, scaler
    else:
        return inX_train, inX_test, outY_train, outY_test, weights_train, weights_test,lumW_train/(1.-testFraction),lumW_test, scaler

def doPlots(yTest, yPredicted, rho, weightTest, lumWTest, processTest,year = "", outFolder = "classOutput/"):

    outDir = outFolder+"/"+year+"/"
    if not os.path.exists(outDir):
        os.makedirs(outDir)

    y_predictedTrue = [np.argmax(entry) for entry in yTest]
    y_predictedTrue = np.array(y_predictedTrue)

    confmatrix          =   ROOT.TH2F("conf",";predicted label;true label",3,0,3,3,0,3)
    # binningDNN = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # correlation          =   ROOT.TH2F("correlation",";rho;outputScore",90,0.,0.9,10,0.,1.)
    correlation          =   ROOT.TH2F("correlation",";rho;outputScore",90,0.,0.9,89,0.01,0.9)
    corBinning = [0., 0.3, 1.]
    arCorBins = array("d",corBinning)
    # correlation_2bin          =   ROOT.TH2F("correlation_2bin",";rho;outputScore",100,0.,1.,2, arCorBins)
    correlation_2bin          =   ROOT.TH2F("correlation_2bin",";rho;outputScore",90,0.,0.9,2, arCorBins)
    # signal node where max
    h_sig_node1_max     =   ROOT.TH1F("h_sig_node1_max",    ";discr;NEvents", 25, 0.,1.)
    h_sig1_node1_max    =   ROOT.TH1F("h_sig1_node1_max",   ";discr;NEvents", 25, 0.,1.)
    h_sig2_node1_max    =   ROOT.TH1F("h_sig2_node1_max",   ";discr;NEvents", 25, 0.,1.)
    h_sig3_node1_max    =   ROOT.TH1F("h_sig3_node1_max",   ";discr;NEvents", 25, 0.,1.)
    h_sig4_node1_max    =   ROOT.TH1F("h_sig4_node1_max",   ";discr;NEvents", 25, 0.,1.)
    h_ttbkg_node1_max   =   ROOT.TH1F("h_ttbkg_node1_max",  ";discr;NEvents", 25, 0.,1.)
    h_dy_node1_max      =   ROOT.TH1F("h_dy_node1_max",     ";discr;NEvents", 25, 0.,1.)
    # used ratio
    h_sig_ratioObservable     =   ROOT.TH1F("h_sig_ratioObservable",    ";#frac{p(t#bar{t}+jet)}{p(t#bar{t}+jet)+p(t#bar{t} BKG)};NEvents", 25, 0.,1.)
    h_sig1_ratioObservable    =   ROOT.TH1F("h_sig1_ratioObservable",   ";#frac{p(t#bar{t}+jet)}{p(t#bar{t}+jet)+p(t#bar{t} BKG)};NEvents", 25, 0.,1.)
    h_sig2_ratioObservable    =   ROOT.TH1F("h_sig2_ratioObservable",   ";#frac{p(t#bar{t}+jet)}{p(t#bar{t}+jet)+p(t#bar{t} BKG)};NEvents", 25, 0.,1.)
    h_sig3_ratioObservable    =   ROOT.TH1F("h_sig3_ratioObservable",   ";#frac{p(t#bar{t}+jet)}{p(t#bar{t}+jet)+p(t#bar{t} BKG)};NEvents", 25, 0.,1.)
    h_sig4_ratioObservable    =   ROOT.TH1F("h_sig4_ratioObservable",   ";#frac{p(t#bar{t}+jet)}{p(t#bar{t}+jet)+p(t#bar{t} BKG)};NEvents", 25, 0.,1.)
    h_ttbkg_ratioObservable   =   ROOT.TH1F("h_ttbkg_ratioObservable",  ";#frac{p(t#bar{t}+jet)}{p(t#bar{t}+jet)+p(t#bar{t} BKG)};NEvents", 25, 0.,1.)
    h_dy_ratioObservable      =   ROOT.TH1F("h_dy_ratioObservable",     ";#frac{p(t#bar{t}+jet)}{p(t#bar{t}+jet)+p(t#bar{t} BKG)};NEvents", 25, 0.,1.)
    h_sig_ratioObservable.GetXaxis().SetTitleSize(1.0)
    h_sig1_ratioObservable.GetXaxis().SetTitleSize(1.0)
    h_sig2_ratioObservable.GetXaxis().SetTitleSize(1.0)
    h_sig3_ratioObservable.GetXaxis().SetTitleSize(1.0)
    h_sig4_ratioObservable.GetXaxis().SetTitleSize(1.0)
    h_ttbkg_ratioObservable.GetXaxis().SetTitleSize(1.0)
    h_dy_ratioObservable.GetXaxis().SetTitleSize(1.0)

    # signal node
    h_sig_node1     =   ROOT.TH1F("h_sig_node1",    ";p(t#bar{t}+jet);NEvents", 25, 0.,1.)
    h_sig1_node1    =   ROOT.TH1F("h_sig1_node1",   ";p(t#bar{t}+jet);NEvents", 25, 0.,1.)
    h_sig2_node1    =   ROOT.TH1F("h_sig2_node1",   ";p(t#bar{t}+jet);NEvents", 25, 0.,1.)
    h_sig3_node1    =   ROOT.TH1F("h_sig3_node1",   ";p(t#bar{t}+jet);NEvents", 25, 0.,1.)
    h_sig4_node1    =   ROOT.TH1F("h_sig4_node1",   ";p(t#bar{t}+jet);NEvents", 25, 0.,1.)
    h_ttbkg_node1   =   ROOT.TH1F("h_ttbkg_node1",  ";p(t#bar{t}+jet);NEvents", 25, 0.,1.)
    h_dy_node1      =   ROOT.TH1F("h_dy_node1",     ";p(t#bar{t}+jet);NEvents", 25, 0.,1.)
    # background nodes
    h_sig_node2     =   ROOT.TH1F("h_sig_node2",    ";p(t#bar{t} BKG);NEvents", 25, 0.,1.)
    h_sig1_node2    =   ROOT.TH1F("h_sig1_node2",   ";p(t#bar{t} BKG);NEvents", 25, 0.,1.)
    h_sig2_node2    =   ROOT.TH1F("h_sig2_node2",   ";p(t#bar{t} BKG);NEvents", 25, 0.,1.)
    h_sig3_node2    =   ROOT.TH1F("h_sig3_node2",   ";p(t#bar{t} BKG);NEvents", 25, 0.,1.)
    h_sig4_node2    =   ROOT.TH1F("h_sig4_node2",   ";p(t#bar{t} BKG);NEvents", 25, 0.,1.)
    h_ttbkg_node2   =   ROOT.TH1F("h_ttbkg_node2",  ";p(t#bar{t} BKG);NEvents", 25, 0.,1.)
    h_dy_node2      =   ROOT.TH1F("h_dy_node2",     ";p(t#bar{t} BKG);NEvents", 25, 0.,1.)

    h_sig_node3     =   ROOT.TH1F("h_sig_node3",    ";p(DY);NEvents", 25, 0.,1.)
    h_sig1_node3    =   ROOT.TH1F("h_sig1_node3",   ";p(DY);NEvents", 25, 0.,1.)
    h_sig2_node3    =   ROOT.TH1F("h_sig2_node3",   ";p(DY);NEvents", 25, 0.,1.)
    h_sig3_node3    =   ROOT.TH1F("h_sig3_node3",   ";p(DY);NEvents", 25, 0.,1.)
    h_sig4_node3    =   ROOT.TH1F("h_sig4_node3",   ";p(DY);NEvents", 25, 0.,1.)
    h_ttbkg_node3   =   ROOT.TH1F("h_ttbkg_node3",  ";p(DY);NEvents", 25, 0.,1.)
    h_dy_node3      =   ROOT.TH1F("h_dy_node3",     ";p(DY);NEvents", 25, 0.,1.)

    h_ttbkg_node1_max.SetLineColor(ROOT.kBlue)
    h_ttbkg_node1.SetLineColor(ROOT.kBlue)
    h_ttbkg_ratioObservable.SetLineColor(ROOT.kBlue)
    h_ttbkg_node2.SetLineColor(ROOT.kBlue)
    h_ttbkg_node3.SetLineColor(ROOT.kBlue)
    h_dy_node1_max.SetLineColor(ROOT.kGreen)
    h_dy_node1.SetLineColor(ROOT.kGreen)
    h_dy_ratioObservable.SetLineColor(ROOT.kGreen)
    h_dy_node2.SetLineColor(ROOT.kGreen)
    h_dy_node3.SetLineColor(ROOT.kGreen)

    h_sig1_node1.SetLineColor(ROOT.kRed)
    h_sig2_node1.SetLineColor(ROOT.kRed+1)
    h_sig3_node1.SetLineColor(ROOT.kRed+2)
    h_sig4_node1.SetLineColor(ROOT.kRed+3)
    h_sig1_ratioObservable.SetLineColor(ROOT.kRed)
    h_sig2_ratioObservable.SetLineColor(ROOT.kRed+1)
    h_sig3_ratioObservable.SetLineColor(ROOT.kRed+2)
    h_sig4_ratioObservable.SetLineColor(ROOT.kRed+3)
    h_sig1_node2.SetLineColor(ROOT.kRed)
    h_sig2_node2.SetLineColor(ROOT.kRed+1)
    h_sig3_node2.SetLineColor(ROOT.kRed+2)
    h_sig4_node2.SetLineColor(ROOT.kRed+3)
    h_sig1_node3.SetLineColor(ROOT.kRed)
    h_sig2_node3.SetLineColor(ROOT.kRed+1)
    h_sig3_node3.SetLineColor(ROOT.kRed+2)
    h_sig4_node3.SetLineColor(ROOT.kRed+3)
    h_sig1_node1_max.SetLineColor(ROOT.kRed)
    h_sig2_node1_max.SetLineColor(ROOT.kRed+1)
    h_sig3_node1_max.SetLineColor(ROOT.kRed+2)
    h_sig4_node1_max.SetLineColor(ROOT.kRed+3)

    y_true_roc3  =[]
    y_pred_roc3  =[]
    y_true_roc2  =[]
    y_pred_roc2  =[]
    y_true_roc1  =[]
    y_pred_roc1  =[]
    y_true_roc0  =[]
    y_pred_roc0  =[]
    print ("ROC score: ",roc_auc_score(y_predictedTrue,yPredicted, average="weighted", multi_class="ovr"))
    # print (roc_auc_score(y_predictedTrue,yPredicted, average="weighted", sample_weight=totWeights, multi_class="ovr"))
    skplt.metrics.plot_roc_curve(y_predictedTrue, yPredicted)
    matplotlib.pyplot.savefig(outDir+"ROC.pdf")

    for trueclass, outScores, weight, lw, process, rhoUse in zip(y_predictedTrue, yPredicted, weightTest,lumWTest,processTest,rho):
        if(process==0):
            if(np.argmax(outScores)==0):
                h_sig_node1_max.Fill(outScores[0],  weight*lw)
                h_sig1_node1_max.Fill(outScores[0], weight*lw)
            h_sig_node1.Fill(outScores[0],  weight*lw)
            h_sig_ratioObservable.Fill(outScores[0]/(outScores[0]+outScores[1]),  weight*lw)
            h_sig1_node1.Fill(outScores[0], weight*lw)
            h_sig1_ratioObservable.Fill(outScores[0]/(outScores[0]+outScores[1]), weight*lw)
            h_sig_node2.Fill(outScores[1],  weight*lw)
            h_sig1_node2.Fill(outScores[1], weight*lw)
            h_sig_node3.Fill(outScores[2],  weight*lw)
            h_sig1_node3.Fill(outScores[2], weight*lw)
            y_true_roc0.append(trueclass)
            y_pred_roc0.append(outScores)
            correlation.Fill(rhoUse,outScores[0], weight*lw)
            correlation_2bin.Fill(rhoUse,outScores[0], weight)
        elif(process==1):
            if(np.argmax(outScores)==0):
                h_sig_node1_max.Fill(outScores[0],  weight*lw)
                h_sig2_node1_max.Fill(outScores[0], weight*lw)
            h_sig_node1.Fill(outScores[0],  weight*lw)
            h_sig_ratioObservable.Fill(outScores[0]/(outScores[0]+outScores[1]),  weight*lw)
            h_sig2_node1.Fill(outScores[0], weight*lw)
            h_sig2_ratioObservable.Fill(outScores[0]/(outScores[0]+outScores[1]), weight*lw)
            h_sig_node2.Fill(outScores[1],  weight*lw)
            h_sig2_node2.Fill(outScores[1], weight*lw)
            h_sig_node3.Fill(outScores[2],  weight*lw)
            h_sig2_node3.Fill(outScores[2], weight*lw)
            y_true_roc1.append(trueclass)
            y_pred_roc1.append(outScores)
            correlation.Fill(rhoUse,outScores[0], weight*lw)
            correlation_2bin.Fill(rhoUse,outScores[0], weight)
        elif(process==2):
            if(np.argmax(outScores)==0):
                h_sig_node1_max.Fill(outScores[0],  weight*lw)
                h_sig3_node1_max.Fill(outScores[0], weight*lw)
            h_sig_node1.Fill(outScores[0],  weight*lw)
            h_sig_ratioObservable.Fill(outScores[0]/(outScores[0]+outScores[1]),  weight*lw)
            h_sig3_node1.Fill(outScores[0], weight*lw)
            h_sig3_ratioObservable.Fill(outScores[0]/(outScores[0]+outScores[1]), weight*lw)
            h_sig_node2.Fill(outScores[1],  weight*lw)
            h_sig3_node2.Fill(outScores[1], weight*lw)
            h_sig_node3.Fill(outScores[2],  weight*lw)
            h_sig3_node3.Fill(outScores[2], weight*lw)
            y_true_roc2.append(trueclass)
            y_pred_roc2.append(outScores)
            correlation.Fill(rhoUse,outScores[0], weight*lw)
            correlation_2bin.Fill(rhoUse,outScores[0], weight)
        elif(process==3):
            if(np.argmax(outScores)==0):
                h_sig_node1_max.Fill(outScores[0],  weight*lw)
                h_sig4_node1_max.Fill(outScores[0], weight*lw)
            h_sig_node1.Fill(outScores[0],  weight*lw)
            h_sig_ratioObservable.Fill(outScores[0]/(outScores[0]+outScores[1]),  weight*lw)
            h_sig4_node1.Fill(outScores[0], weight*lw)
            h_sig4_ratioObservable.Fill(outScores[0]/(outScores[0]+outScores[1]), weight*lw)
            h_sig_node2.Fill(outScores[1],  weight*lw)
            h_sig4_node2.Fill(outScores[1], weight*lw)
            h_sig_node3.Fill(outScores[2],  weight*lw)
            h_sig4_node3.Fill(outScores[2], weight*lw)
            y_true_roc3.append(trueclass)
            y_pred_roc3.append(outScores)
            correlation.Fill(rhoUse,outScores[0], weight*lw)
            correlation_2bin.Fill(rhoUse,outScores[0], weight)
        elif (process==4):
            if(np.argmax(outScores)==0):
                h_ttbkg_node1_max.Fill(outScores[0], weight*lw)
            h_ttbkg_node1.Fill(outScores[0], weight*lw)
            h_ttbkg_ratioObservable.Fill(outScores[0]/(outScores[0]+outScores[1]), weight*lw)
            h_ttbkg_node2.Fill(outScores[1], weight*lw)
            h_ttbkg_node3.Fill(outScores[2], weight*lw)
            y_true_roc0.append(trueclass)
            y_pred_roc0.append(outScores)
            y_true_roc1.append(trueclass)
            y_pred_roc1.append(outScores)
            y_true_roc2.append(trueclass)
            y_pred_roc2.append(outScores)
            y_true_roc3.append(trueclass)
            y_pred_roc3.append(outScores)
        elif (process==5):
            if(np.argmax(outScores)==0):
                h_dy_node1_max.Fill(outScores[0], weight*lw)
            h_dy_node1.Fill(outScores[0], weight*lw)
            h_dy_ratioObservable.Fill(outScores[0]/(outScores[0]+outScores[1]), weight*lw)
            h_dy_node2.Fill(outScores[1], weight*lw)
            h_dy_node3.Fill(outScores[2], weight*lw)
            y_true_roc0.append(trueclass)
            y_pred_roc0.append(outScores)
            y_true_roc1.append(trueclass)
            y_pred_roc1.append(outScores)
            y_true_roc2.append(trueclass)
            y_pred_roc2.append(outScores)
            y_true_roc3.append(trueclass)
            y_pred_roc3.append(outScores)
        else:
            print (process, "SHOULD NOT BE HERE!")
        confmatrix.Fill(np.argmax(outScores), trueclass, weight*lw)


    y_pred_roc0=np.array(y_pred_roc0)
    y_true_roc0=np.array(y_true_roc0)
    y_pred_roc1=np.array(y_pred_roc1)
    y_true_roc1=np.array(y_true_roc1)
    y_pred_roc2=np.array(y_pred_roc2)
    y_true_roc2=np.array(y_true_roc2)
    y_pred_roc3=np.array(y_pred_roc3)
    y_true_roc3=np.array(y_true_roc3)

    for binY in range(1,correlation.GetNbinsY()+1):
        intY = correlation.Integral(-1,-1,binY,binY)
        if intY>0.:
            for binX in range(1,correlation.GetNbinsX()+1):
                content = correlation.GetBinContent(binX, binY)
                correlation.SetBinContent(binX, binY, content/intY)

    for binY in range(1,correlation_2bin.GetNbinsY()+1):
        intY = correlation_2bin.Integral(-1,-1,binY,binY)
        if intY>0.:
            for binX in range(1,correlation_2bin.GetNbinsX()+1):
                content = correlation_2bin.GetBinContent(binX, binY)
                correlation_2bin.SetBinContent(binX, binY, content/intY)

    print ("ROC score 0: ",roc_auc_score(y_true_roc0, y_pred_roc0, average="weighted", multi_class="ovr"))
    skplt.metrics.plot_roc_curve(y_true_roc0, y_pred_roc0)
    matplotlib.pyplot.savefig(outDir+"ROC_0.pdf")
    print ("ROC score 1: ",roc_auc_score(y_true_roc1,y_pred_roc1, average="weighted", multi_class="ovr"))
    skplt.metrics.plot_roc_curve(y_true_roc1, y_pred_roc1)
    matplotlib.pyplot.savefig(outDir+"ROC_1.pdf")
    print ("ROC score 2: ",roc_auc_score(y_true_roc2,y_pred_roc2, average="weighted", multi_class="ovr"))
    skplt.metrics.plot_roc_curve(y_true_roc2, y_pred_roc2)
    matplotlib.pyplot.savefig(outDir+"ROC_2.pdf")
    print ("ROC score 3: ",roc_auc_score(y_true_roc3,y_pred_roc3, average="weighted", multi_class="ovr"))
    skplt.metrics.plot_roc_curve(y_true_roc3, y_pred_roc3)
    matplotlib.pyplot.savefig(outDir+"ROC_3.pdf")


    # normalize everything
    for h in [
              h_sig1_node1,h_sig2_node1,h_sig3_node1,h_sig4_node1,h_ttbkg_node1,h_dy_node1,
              h_sig1_ratioObservable,h_sig2_ratioObservable,h_sig3_ratioObservable,h_sig4_ratioObservable,h_ttbkg_ratioObservable,h_dy_ratioObservable,
              h_sig1_node1_max, h_sig2_node1_max, h_sig3_node1_max,h_sig4_node1_max, h_ttbkg_node1_max, h_dy_node1_max,
              h_sig1_node2,h_sig2_node2,h_sig3_node2,h_sig4_node2,h_ttbkg_node2,h_dy_node2,
              h_sig1_node3,h_sig2_node3,h_sig3_node3,h_sig4_node3,h_ttbkg_node3,h_dy_node3,
             ]:
        # print (h.GetName())
        h.Scale(1./h.Integral())
        h.SetLineWidth(2)



    # setLog = True
    setLog = False

    style.style1d()
    s = style.style1d()
    # c=ROOT.TCanvas("c1","c1",800,800)
    c=ROOT.TCanvas()
    c.SetLogy(setLog)
    maximum = max(h_sig1_node1.GetMaximum(),h_sig2_node1.GetMaximum(),h_sig3_node1.GetMaximum(),h_sig4_node1.GetMaximum(),h_ttbkg_node1.GetMaximum(),h_dy_node1.GetMaximum())
    # for h_ in [
    #           h_sig1_node1, h_sig2_node1, h_sig3_node1, h_sig4_node1, h_ttbkg_node1, h_dy_node1,
    #           h_sig1_node2, h_sig2_node2, h_sig3_node2, h_sig4_node2, h_ttbkg_node2, h_dy_node2,
    #           h_sig1_node3, h_sig2_node3, h_sig3_node3, h_sig4_node3, h_ttbkg_node3, h_dy_node3,
    #           ]:
    #     h_.SetLineWidth(2)
    if(setLog):
        maximum =maximum*100.
    h_sig_node1.SetMaximum(maximum*1.3)
    h_sig1_node1.SetMaximum(maximum*1.3)
    h_sig2_node1.SetMaximum(maximum*1.3)
    h_sig3_node1.SetMaximum(maximum*1.3)
    h_sig4_node1.SetMaximum(maximum*1.3)
    h_ttbkg_node1.SetMaximum(maximum*1.3)
    h_dy_node1.SetMaximum(maximum*1.3)
    h_sig1_node1.Draw("histe")
    h_sig2_node1.Draw("histe same")
    h_sig3_node1.Draw("histe same")
    h_sig4_node1.Draw("histe same")
    h_ttbkg_node1.Draw("histe same")
    h_dy_node1.Draw("histe same")
    # Adding a legend
    leg = ROOT.TLegend(.56, .75, .94, .9)
    leg.AddEntry(h_sig1_node1, "t#bar{t}+jet #rho#in[0.7,1.]")
    leg.AddEntry(h_sig2_node1, "t#bar{t}+jet #rho#in[0.45,0.7]")
    leg.AddEntry(h_sig3_node1, "t#bar{t}+jet #rho#in[0.3,0.45]")
    leg.AddEntry(h_sig4_node1, "t#bar{t}+jet #rho#in[0,0.3]")
    leg.AddEntry(h_ttbkg_node1, "t#bar{t} bkg")
    leg.AddEntry(h_dy_node1, "DY")
    leg.Draw()
    c.SaveAs(outDir+"classifierOutputNode0.pdf")

    c.Clear()
    maximum = max(h_sig1_node1_max.GetMaximum(),h_sig2_node1_max.GetMaximum(),h_sig3_node1_max.GetMaximum(),h_sig4_node1_max.GetMaximum(),h_ttbkg_node1_max.GetMaximum(),h_dy_node1_max.GetMaximum())
    if(setLog):
        maximum =maximum*100.
    h_sig_node1_max.SetMaximum(maximum*1.3)
    h_sig1_node1_max.SetMaximum(maximum*1.3)
    h_sig2_node1_max.SetMaximum(maximum*1.3)
    h_sig3_node1_max.SetMaximum(maximum*1.3)
    h_sig4_node1_max.SetMaximum(maximum*1.3)
    h_ttbkg_node1_max.SetMaximum(maximum*1.3)
    h_dy_node1_max.SetMaximum(maximum*1.3)
    h_sig1_node1_max.Draw("histe")
    h_sig2_node1_max.Draw("histe same")
    h_sig3_node1_max.Draw("histe same")
    h_sig4_node1_max.Draw("histe same")
    h_ttbkg_node1_max.Draw("hist same")
    h_dy_node1_max.Draw("hist same")
    # Adding a legend
    leg = ROOT.TLegend(.56, .75, .94, .9)
    leg.AddEntry(h_sig1_node1_max, "t#bar{t}+jet #rho#in[0.7,1.]")
    leg.AddEntry(h_sig2_node1_max, "t#bar{t}+jet #rho#in[0.45,0.7]")
    leg.AddEntry(h_sig3_node1_max, "t#bar{t}+jet #rho#in[0.3,0.45]")
    leg.AddEntry(h_sig4_node1_max, "t#bar{t}+jet #rho#in[0,0.3]")
    leg.AddEntry(h_ttbkg_node1_max, "t#bar{t} bkg")
    leg.AddEntry(h_dy_node1_max, "DY")
    leg.Draw()
    c.SaveAs(outDir+"classifierOutputNode0_max.pdf")

    c.Clear()
    maximum = max(h_sig1_ratioObservable.GetMaximum(),h_sig2_ratioObservable.GetMaximum(),h_sig3_ratioObservable.GetMaximum(),h_sig4_ratioObservable.GetMaximum(),h_ttbkg_ratioObservable.GetMaximum(),h_dy_ratioObservable.GetMaximum())
    if(setLog):
        maximum =maximum*100.
    h_sig_ratioObservable.SetMaximum(maximum*1.3)
    h_sig1_ratioObservable.SetMaximum(maximum*1.3)
    h_sig2_ratioObservable.SetMaximum(maximum*1.3)
    h_sig3_ratioObservable.SetMaximum(maximum*1.3)
    h_sig4_ratioObservable.SetMaximum(maximum*1.3)
    h_ttbkg_ratioObservable.SetMaximum(maximum*1.3)
    h_dy_ratioObservable.SetMaximum(maximum*1.3)
    h_sig1_ratioObservable.Draw("histe")
    h_sig2_ratioObservable.Draw("histe same")
    h_sig3_ratioObservable.Draw("histe same")
    h_sig4_ratioObservable.Draw("histe same")
    h_ttbkg_ratioObservable.Draw("hist same")
    h_dy_ratioObservable.Draw("hist same")
    # Adding a legend
    leg = ROOT.TLegend(.56, .75, .94, .9)
    leg.AddEntry(h_sig1_ratioObservable, "t#bar{t}+jet #rho#in[0.7,1.]")
    leg.AddEntry(h_sig2_ratioObservable, "t#bar{t}+jet #rho#in[0.45,0.7]")
    leg.AddEntry(h_sig3_ratioObservable, "t#bar{t}+jet #rho#in[0.3,0.45]")
    leg.AddEntry(h_sig4_ratioObservable, "t#bar{t}+jet #rho#in[0,0.3]")
    leg.AddEntry(h_ttbkg_ratioObservable, "t#bar{t} bkg")
    leg.AddEntry(h_dy_ratioObservable, "DY")
    leg.Draw()
    c.SaveAs(outDir+"classifierOutput_ratioObservable.pdf")

    c.Clear()
    maximum = max(h_sig1_node1.GetMaximum(),h_sig2_node2.GetMaximum(),h_sig3_node2.GetMaximum(),h_sig4_node2.GetMaximum(),h_ttbkg_node2.GetMaximum(),h_dy_node2.GetMaximum())
    if(setLog):
        maximum =maximum*100.
    h_sig_node2.SetMaximum(maximum*1.3)
    h_sig1_node2.SetMaximum(maximum*1.3)
    h_sig2_node2.SetMaximum(maximum*1.3)
    h_sig3_node2.SetMaximum(maximum*1.3)
    h_sig4_node2.SetMaximum(maximum*1.3)
    h_ttbkg_node2.SetMaximum(maximum*1.3)
    h_dy_node2.SetMaximum(maximum*1.3)
    h_sig1_node2.Draw("histe")
    h_sig1_node2.Draw("histe same")
    h_sig2_node2.Draw("histe same")
    h_sig3_node2.Draw("histe same")
    h_sig4_node2.Draw("histe same")
    h_ttbkg_node2.Draw("histe same")
    h_dy_node2.Draw("histe same")
    # Adding a legend
    leg = ROOT.TLegend(.56, .75, .94, .9)
    leg.AddEntry(h_sig1_node2, "t#bar{t}+jet #rho#in[0.7,1.]")
    leg.AddEntry(h_sig2_node2, "t#bar{t}+jet #rho#in[0.45,0.7]")
    leg.AddEntry(h_sig3_node2, "t#bar{t}+jet #rho#in[0.3,0.45]")
    leg.AddEntry(h_sig4_node2, "t#bar{t}+jet #rho#in[0,0.3]")
    leg.AddEntry(h_ttbkg_node2, "t#bar{t} bkg")
    leg.AddEntry(h_dy_node2, "DY")
    leg.Draw()
    c.SaveAs(outDir+"classifierOutputNode1.pdf")

    c.Clear()
    maximum = max(h_sig1_node3.GetMaximum(),h_sig2_node3.GetMaximum(),h_sig3_node3.GetMaximum(),h_sig4_node3.GetMaximum(),h_ttbkg_node3.GetMaximum(),h_dy_node3.GetMaximum())
    if(setLog):
        maximum =maximum*100.
    h_sig1_node3.SetMaximum(maximum*1.3)
    h_sig2_node3.SetMaximum(maximum*1.3)
    h_sig3_node3.SetMaximum(maximum*1.3)
    h_sig4_node3.SetMaximum(maximum*1.3)
    h_sig_node3.SetMaximum(maximum*1.3)
    h_ttbkg_node3.SetMaximum(maximum*1.3)
    h_dy_node3.SetMaximum(maximum*1.3)
    h_sig1_node3.Draw("histe")
    h_sig2_node3.Draw("histe same")
    h_sig3_node3.Draw("histe same")
    h_sig4_node3.Draw("histe same")
    h_ttbkg_node3.Draw("histe same")
    h_dy_node3.Draw("histe same")
    # Adding a legend
    leg = ROOT.TLegend(.56, .75, .94, .9)
    leg.AddEntry(h_sig1_node3, "t#bar{t}+jet #rho#in[0.7,1.]")
    leg.AddEntry(h_sig2_node3, "t#bar{t}+jet #rho#in[0.45,0.7]")
    leg.AddEntry(h_sig3_node3, "t#bar{t}+jet #rho#in[0.3,0.45]")
    leg.AddEntry(h_sig4_node3, "t#bar{t}+jet #rho#in[0,0.3]")
    leg.AddEntry(h_ttbkg_node3, "t#bar{t} bkg")
    leg.AddEntry(h_dy_node3, "DY")
    leg.Draw()
    c.SaveAs(outDir+"classifierOutputNode2.pdf")

    # c.Clear()
    # c.SetLogy(False)
    # # cut value
    # nbins = h_sig_node1.GetNbinsX()
    # h_cutValue = h_sig_node1.Clone()
    # maxiV = 0.
    # for binX in range(1,nbins+1):
    #     intSB = 0.
    #     for index in range(binX,nbins+1):
    #         s = h_sig_node1.GetBinContent(index)
    #         b = h_ttbkg_node1.GetBinContent(index)+h_dy_node1.GetBinContent(index)
    #         thisBin = s/np.sqrt(s+b)
    #         intSB = intSB + (thisBin*thisBin)
    #     h_cutValue.SetBinContent(binX, np.sqrt(intSB))
    #     if np.sqrt(intSB)> maxiV:
    #         maxiV = np.sqrt(intSB)
    #
    # h_cutValue.SetMaximum(maxiV*1.3)
    # h_cutValue.SetMinimum(0.)
    # h_cutValue.Draw("hist")
    # c.SaveAs(outDir+"sOverB.pdf")

    # c.Clear()
    # c.SetLogy(False)
    # h_cutValue2 = h_sig_node1.Clone()
    # h_cutValue2_sig1 = h_sig_node1.Clone()
    # h_cutValue2_sig2 = h_sig_node1.Clone()
    # h_cutValue2_sig3 = h_sig_node1.Clone()
    # nbins = h_cutValue2.GetNbinsX()
    # maxiV_ = 0.
    # for binX_ in range(1,nbins+1):
    #     s_ = h_sig_node1.Integral(binX_,nbins+1)
    #     s_1 = h_sig1_node1.Integral(binX_,nbins+1)
    #     s_2 = h_sig2_node1.Integral(binX_,nbins+1)
    #     s_3 = h_sig3_node1.Integral(binX_,nbins+1)
    #     b_ = h_ttbkg_node1.Integral(binX_,nbins+1)+h_dy_node1.Integral(binX_,nbins+1)
    #     sb = s_/np.sqrt(s_+b_)
    #     sb1 = s_/np.sqrt(s_1+b_)
    #     sb2 = s_/np.sqrt(s_2+b_)
    #     sb3 = s_/np.sqrt(s_3+b_)
    #     h_cutValue2.SetBinContent(binX_, sb)
    #     h_cutValue2_sig1.SetBinContent(binX_, sb1)
    #     h_cutValue2_sig2.SetBinContent(binX_, sb2)
    #     h_cutValue2_sig3.SetBinContent(binX_, sb3)
    #     if sb > maxiV_:
    #         maxiV_ = (sb*1.001)
    #
    # h_cutValue2.SetMaximum(maxiV_*1.3)
    # h_cutValue2_sig1.SetMaximum(maxiV_*1.3)
    # h_cutValue2_sig2.SetMaximum(maxiV_*1.3)
    # h_cutValue2_sig3.SetMaximum(maxiV_*1.3)
    # h_cutValue2.SetMinimum(0.)
    # h_cutValue2_sig1.SetLineColor(ROOT.kRed)
    # h_cutValue2_sig2.SetLineColor(ROOT.kRed+1)
    # h_cutValue2_sig3.SetLineColor(ROOT.kRed+2)
    # h_cutValue2.Draw("hist")
    # h_cutValue2_sig1.Draw("hist same")
    # h_cutValue2_sig2.Draw("hist same")
    # h_cutValue2_sig3.Draw("hist same")
    # c.SaveAs(outDir+"sbSum.pdf")

    # c.Clear()
    # c.SetLogy(False)
    # h_cutValue2 = h_sig_node1.Clone()
    # nbins = h_cutValue2.GetNbinsX()
    # maxiV_ = 0.
    # for binX_ in range(1,nbins+1):
    #     s_ = h_sig_node1.GetBinContent(binX_)
    #     b_ = h_ttbkg_node1.GetBinContent(binX_)+h_dy_node1.GetBinContent(binX_)
    #     sb = s_/b_
    #     h_cutValue2.SetBinContent(binX_, sb)
    #     if sb > maxiV_:
    #         maxiV_ = (sb*1.001)
    #
    # h_cutValue2.SetMaximum(maxiV_*1.3)
    # h_cutValue2.SetMinimum(0.)
    # h_cutValue2.Draw("hist")
    # # c.SaveAs(outDir+"sb.pdf")
    # c.Clear()
    # c.SetLogy(False)
    # h_cutValue3 = h_sig_node1.Clone()
    # nbins = h_cutValue3.GetNbinsX()
    # maxiV_ = 0.
    # for binX_ in range(1,nbins+1):
    #     s_ = h_sig_node1.GetBinContent(binX_)
    #     b_ = h_ttbkg_node1.GetBinContent(binX_)
    #     sb = s_/b_
    #     h_cutValue3.SetBinContent(binX_, sb)
    #     if sb > maxiV_:
    #         maxiV_ = (sb*1.001)
    #
    # h_cutValue3.SetMaximum(maxiV_*1.3)
    # h_cutValue3.SetMinimum(0.)
    # h_cutValue3.Draw("hist")
    # c.SaveAs(outDir+"sb_ttonly.pdf")
    # confusion matrix
    c.Clear()
    c.SetLogx(False)
    c.SetLogy(False)
    style.style2d()
    s = style.style2d()
    c=ROOT.TCanvas()
    confmatrix.Draw("colz text")
    ROOT.gStyle.SetPaintTextFormat("2.1f")
    confmatrix.SetMarkerSize(0.7)
    confmatrix.GetZaxis().SetTitle("Events")
    ROOT.gStyle.SetPalette(ROOT.kThermometer)
    c.SaveAs(outDir+"confmatrix.pdf")

    # correlation
    c.Clear()
    s.SetPadRightMargin(0.2)
    c.SetLogx(False)
    c.SetLogy(False)
    correlation.Draw("colz")
    ROOT.gStyle.SetPalette(ROOT.kThermometer)
    c.SaveAs(outDir+"correlation.pdf")

    c.Clear()
    s.SetPadRightMargin(0.2)
    c.SetLogx(False)
    c.SetLogy(False)
    correlation_2bin.Draw("colz")
    ROOT.gStyle.SetPalette(ROOT.kThermometer)
    c.SaveAs(outDir+"correlation_2bin.pdf")

    h_upper = correlation_2bin.ProjectionX("up", 1, 1)
    h_lower = correlation_2bin.ProjectionX("low", 2, 2)
    h_upper.Scale(1./h_upper.Integral())
    hsum = correlation_2bin.ProjectionX("sum", 1, 2)
    h_lower.Scale(1./h_lower.Integral())

    # hsum = h_upper.Clone()
    # hsum.Add(h_lower)
    hsum.Scale(1./hsum.Integral())

    c.Clear()
    hsum.Draw("hist")
    h_upper.Draw("hist same")
    h_lower.SetLineColor(ROOT.kRed)
    h_lower.Draw("hist same")
    hsum.SetLineColor(ROOT.kGreen)
    c.SaveAs(outDir+"correlation_shape.pdf")

    c.Clear()
    confmatrixNorm = confmatrix.Clone()
    for binX in range(1, confmatrixNorm.GetNbinsX()+1):
        for binY in range(1, confmatrixNorm.GetNbinsY()+1):
            intTrues = confmatrix.Integral(1, confmatrixNorm.GetNbinsX()+1, binY, binY)
            nomVal = confmatrix.GetBinContent(binX, binY)/intTrues
            confmatrixNorm.SetBinContent(binX, binY,nomVal)
    ROOT.gStyle.SetPaintTextFormat("1.3f")
    confmatrixNorm.Draw("colz text")
    c.SaveAs(outDir+"confmatrixNorm.pdf")

    outRootFile = ROOT.TFile(outDir+"/AllPlots.root", "RECREATE")
    outRootFile.cd()
    confmatrix.Write("confmatrix")
    confmatrixNorm.Write("confmatrixNormed")
    correlation.Write("correlation")
    correlation_2bin.Write("correlation_2bin")
    h_sig_node1_max.Write("node1max_sig")
    h_sig1_node1_max.Write("node1max_sig1")
    h_sig2_node1_max.Write("node1max_sig2")
    h_sig3_node1_max.Write("node1max_sig3")
    h_sig4_node1_max.Write("node1max_sig4")
    h_ttbkg_node1_max.Write("node1max_ttbkg")
    h_dy_node1_max.Write("node1max_dy")
    h_sig_node1.Write("node1_sig")
    h_sig1_node1.Write("node1_sig1")
    h_sig2_node1.Write("node1_sig2")
    h_sig3_node1.Write("node1_sig3")
    h_sig4_node1.Write("node1_sig4")
    h_ttbkg_node1.Write("node1_ttbkg")
    h_sig_ratioObservable.Write("ratioObservable_sig")
    h_sig1_ratioObservable.Write("ratioObservable_sig1")
    h_sig2_ratioObservable.Write("ratioObservable_sig2")
    h_sig3_ratioObservable.Write("ratioObservable_sig3")
    h_sig4_ratioObservable.Write("ratioObservable_sig4")
    h_ttbkg_ratioObservable.Write("ratioObservable_ttbkg")
    h_dy_node1.Write("node1_dy")
    h_sig_node2.Write("node2_sig")
    h_sig1_node2.Write("node2_sig1")
    h_sig2_node2.Write("node2_sig2")
    h_sig3_node2.Write("node2_sig3")
    h_sig4_node2.Write("node2_sig4")
    h_ttbkg_node2.Write("node2_ttbkg")
    h_dy_node2.Write("node2_dy")
    h_sig_node3.Write("node3_sig")
    h_sig1_node3.Write("node3_sig1")
    h_sig2_node3.Write("node3_sig2")
    h_sig3_node3.Write("node3_sig3")
    h_sig4_node3.Write("node3_sig4")
    h_ttbkg_node3.Write("node3_ttbkg")
    h_dy_node3.Write("node3_dy")
    outRootFile.Close()

def doFitForBaysian(inX_train, outY_train, weights_train, inX_test, outY_test, weights_test, learningRate = 0.001, batchSize = 4096, indexActivation = 0, dropout = 0.05, nDense = 3, nNodes = 500, regRate = 1e-7):
    nDense = int(nDense)
    nNodes = int(nNodes)
    batchSize = int(batchSize)
    indexActivation = int(indexActivation)

    activations = ["sigmoid", "relu", "softmax", "selu", "softplus"]
    activation = activations[indexActivation]

    print ("batch size",batchSize,"nDense",nDense,"nNodes",nNodes,"indexActivation",indexActivation,"activation",activation,"dropout",dropout,"regRate",regRate,"learningRate",learningRate)

    model = models.getClassificationModel(outputNodes =3 ,regRate = regRate, activation = activation, dropout = dropout, nDense = nDense,
                                      nNodes = nNodes, inputDim = inX_train.shape[1], outputActivation = 'softmax',
                                      printSummary=False)
    optimizer = tf.keras.optimizers.Adam(lr = learningRate)

    if int(nNodes*0.5**(nDense-1))<3:
        return 0.

    print ("compiling model")
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['accuracy',])

    print ("N parameter: ",model.count_params())
    if (model.count_params()> 1500000):
        return 0.

    callbacks=[]
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', patience=3)
    earlyStop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=70, restore_best_weights=True)

    # callbacks.append(reduce_lr)
    callbacks.append(earlyStop)

    print ("fitting model")

    fit = model.fit(
            inX_train,
            outY_train,
            sample_weight = weights_train,
            validation_split = 0.3,
            batch_size = batchSize,
            epochs = 250,
            shuffle = False,
            callbacks = callbacks,
            verbose = 0)
            # verbose = 1)

    score = model.evaluate(
                        inX_test,
                        outY_test,
                        sample_weight = weights_test,
                        batch_size = batchSize,
                        verbose = 0)
    y_predicted = model.predict(inX_test)

    print('Test loss:', score[0])
    print('Test Accuracy:', score[1])

    y_predictedTrue = [np.argmax(entry) for entry in outY_test]
    y_predictedTrue = np.array(y_predictedTrue)

    rocs = roc_auc_score(y_predictedTrue,y_predicted, average="weighted", multi_class="ovr")
    print ("ROC score: ",rocs)
    return rocs

def doBaysianOptim(inPathFolder, additionalName, year, tokeep = None, outFolder = "BaysianOptim/"):
    inX_train, inX_test, outY_train, outY_test, weights_train, weights_test, lumW_train, lumW_test, scaler = loadData(
        inPathFolder = inPathFolder, year=year, additionalName = additionalName, testFraction = 0.3,
        overwrite = False, withBTag = True, withCharge = False,
        maxEvents = None)

    feature_names, inX_train, inX_test = helpers.getReducedFeatureNamesAndInputs(inX_train, inX_test, tokeep=tokeep)
    weights_train_original = weights_train
    weights_test_original = weights_test
    outY_test, outY_train, weights_test, weights_train, proc_test, proc_train = reweight(outY_test, outY_train, weights_test, weights_train)



    fit_with_partial = partial(doFitForBaysian, inX_train, outY_train, weights_train, inX_test, outY_test, weights_test)

    # Bounded region of parameter space
    pbounds = {'batchSize': (1000, 10000),
                'regRate': (1e-7, 1e-4),
                'learningRate': (1e-4, 1e-2),
                'dropout': (1e-4, 5e-1),
                'nDense': (2, 10),
                'nNodes': (50, 1000),
                'indexActivation': (0, 4)
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

    if not os.path.exists(outFolder):
        os.makedirs(outFolder)
    if os.path.exists(outFolder+"/logs.json"):
        print("Loading logs from",outFolder+"/logs.json")
        load_logs(optimizer, logs=[outFolder+"/logs.json"])
    else:
        print("Create new logs in",outFolder+"/logs.json")
    logger = JSONLogger(path = outFolder+"/logs.json", reset=False)

    optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
    optimizer.maximize(init_points=50, n_iter=10,kappa=8, alpha=1e-3, n_restarts_optimizer=5)
    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))

    print(optimizer.max)

def doTrainingAndEvaluation(inPathFolder, additionalName, year, tokeep = None, tokeepReg=None, outFolder="outFolder_DEFAULTNAME/", modelName = "classModel_preUL04"):
    hp_lambda = 0.75
    # hp_lambda = 0.3
    # hp_lambda = 10000.


    learningRate = 0.0001
    # learningRate = 0.001
    # batchSize = 5563
    # batchSize = 15563
    batchSize = 512
    # dropout =  0.488
    dropout =  0.288
    nDense = int(4)
    # nNodes = int(104)
    nNodes = int(204)
    regRate = 7.835e-05
    activation = 'selu'
    outputActivation = 'softmax'

    if not os.path.exists("tmpDIR/inX_train.npy"):

        inX_train, inX_test, outY_train, outY_test, weights_train, weights_test, lumW_train, lumW_test, scaler = loadData(
                                        inPathFolder = inPathFolder, year=year, additionalName = additionalName,
                                        testFraction = 0.3, overwrite = False, withBTag = True, withCharge = False,
                                        maxEvents = None)
                                        # maxEvents = 1000000)

        os.makedirs("tmpDIR/")
        feature_names, inX_train, inX_test, targetRho, testRho = helpers.getReducedFeatureNamesAndInputsWithSecondSet(inX_train, inX_test, tokeep=tokeep, tokeep2=tokeepReg,
            # regDNNPath ="/nfs/dust/cms/user/sewuchte/TopRhoNetwork/preUL04_11_02_21/rhoReg_madgraph/rhoRegModel_preUL04_11_02_21.h5")
            # regDNNPath ="/nfs/dust/cms/user/sewuchte/TopRhoNetwork/preUL04_11_02_21_ghost/rhoReg_madgraph_ghost_reweight_38/rhoRegModel_preUL04_11_02_21_ghost.h5")
            regDNNPath ="/nfs/dust/cms/user/sewuchte/TopRhoNetwork/preUL05_19_01_22_3/rhoReg_madgraph/preUL05_rhoRegModel_19_01_22.h5")

        targetRho = np.array(targetRho)

        weights_train_original = weights_train
        weights_test_original = weights_test
        outY_test, outY_train, weights_test, weights_train, proc_test, proc_train = reweight(outY_test, outY_train, weights_test, weights_train)

        np.save("tmpDIR/inX_train.npy",inX_train)
        np.save("tmpDIR/inX_test.npy",inX_test)
        np.save("tmpDIR/targetRho.npy",targetRho)
        np.save("tmpDIR/testRho.npy",testRho)
        np.save("tmpDIR/outY_test.npy",outY_test)
        np.save("tmpDIR/outY_train.npy",outY_train)
        np.save("tmpDIR/weights_test.npy",weights_test)
        np.save("tmpDIR/weights_train.npy",weights_train)
        np.save("tmpDIR/weights_train_original.npy",weights_train_original)
        np.save("tmpDIR/weights_test_original.npy",weights_test_original)
        np.save("tmpDIR/proc_test.npy",proc_test)
        np.save("tmpDIR/proc_train.npy",proc_train)
        np.save("tmpDIR/lumW_train.npy",lumW_train)
        np.save("tmpDIR/lumW_test.npy",lumW_test)

    else:

        inX_train = np.load("tmpDIR/inX_train.npy", allow_pickle=True)
        inX_test = np.load("tmpDIR/inX_test.npy", allow_pickle=True)
        targetRho = np.load("tmpDIR/targetRho.npy", allow_pickle=True)
        testRho = np.load("tmpDIR/testRho.npy", allow_pickle=True)
        outY_test = np.load("tmpDIR/outY_test.npy", allow_pickle=True)
        outY_train = np.load("tmpDIR/outY_train.npy", allow_pickle=True)
        weights_test = np.load("tmpDIR/weights_test.npy", allow_pickle=True)
        weights_train = np.load("tmpDIR/weights_train.npy", allow_pickle=True)
        weights_train_original = np.load("tmpDIR/weights_train_original.npy", allow_pickle=True)
        weights_test_original = np.load("tmpDIR/weights_test_original.npy", allow_pickle=True)
        proc_test = np.load("tmpDIR/proc_test.npy", allow_pickle=True)
        proc_train = np.load("tmpDIR/proc_train.npy", allow_pickle=True)
        lumW_train = np.load("tmpDIR/lumW_train.npy", allow_pickle=True)
        lumW_test = np.load("tmpDIR/lumW_test.npy", allow_pickle=True)
        feature_names = helpers.getReducedFeatureNames(tokeep=tokeep)

        print (outY_train)
        counts=[0. for i in range(4)]
        weights=[]
        for i in range(len(outY_train)):
            if (proc_train[i][0]<4):
                counts[proc_train[i][0]] = counts[proc_train[i][0]] +weights_train[i]
        print ("signal sums:",counts)


    print ("getting model")
    model = models.getClassificationModelWithInvariance(outputNodes =3 ,regRate = regRate, activation = activation, dropout = dropout, nDense = nDense,
                                      nNodes = nNodes, inputDim = inX_train.shape[1], outputActivation = outputActivation, hp_lambda=hp_lambda)
    optimizer = tf.keras.optimizers.Adam(lr = learningRate)


    print ("compiling model")
    # model.compile(optimizer=optimizer, loss=['categorical_crossentropy','mean_squared_error'],metrics={'output': 'accuracy', 'gradRev_output': 'mae'}, loss_weights=[1., 5.])
    # model.compile(optimizer=optimizer, loss=['categorical_crossentropy','mean_squared_error'],metrics={'output': 'accuracy', 'gradRev_output': 'mae'}, loss_weights=[1., 3.])
    # model.compile(optimizer=optimizer, loss=['categorical_crossentropy','mean_squared_error'],metrics={'output': 'accuracy', 'gradRev_output': 'mae'}, loss_weights=[1., 0.001])
    model.compile(optimizer=optimizer, loss=['categorical_crossentropy','mean_squared_error'],metrics={'output': 'accuracy', 'gradRev_output': 'mae'}, loss_weights=[1., 0.1])

    callbacks=[]
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', patience=3)
    # earlyStop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=50, restore_best_weights=True)
    earlyStop = tf.keras.callbacks.EarlyStopping(monitor = 'val_output_loss', patience=250, restore_best_weights=True)
    # earlyStop = tf.keras.callbacks.EarlyStopping(monitor = 'val_output_loss', patience=450, restore_best_weights=True)
    modelCheckpoint = tf.keras.callbacks.ModelCheckpoint(outFolder + '/model_best.h5', monitor='val_output_loss', save_best_only=True)


    print (inX_train[0])
    print (outY_train[0])
    print (targetRho[0])


    # callbacks.append(reduce_lr)
    callbacks.append(earlyStop)
    callbacks.append(modelCheckpoint)

    print ("fitting model")

    fit = model.fit(
            x = inX_train,
            # outY_train,
            y = [outY_train,targetRho],
            # sample_weight = weights_train,
            sample_weight = {'output': weights_train,
                   # 'gradRev_output': np.ones(len(weights_train))},
                   # 'gradRev_output': weights_train_original},
                   'gradRev_output': weights_train},

            validation_split = 0.25,
            batch_size = batchSize,
            epochs = 5000,
            # epochs = 3000,
            # epochs = 150,
            # epochs = 50,
            # epochs = 5,
            shuffle = False,
            callbacks = callbacks,
            verbose = 1)

    # ####################
    # # --- save model ---
    # ####################
    if not os.path.exists(outFolder+"/"+year):
        os.makedirs(outFolder+"/"+year)
    model.save(outFolder + '/model_last.h5')

    # --- load weights in classifier for evaluation
    model = models.getClassificationModel(outputNodes =3 ,regRate = regRate, activation = activation, dropout = dropout, nDense = nDense,
                                      nNodes = nNodes, inputDim = inX_train.shape[1], outputActivation = outputActivation)

    # model.load_weights(outFolder.replace("_last","") + '/model_last.h5', by_name=True)
    model.load_weights(outFolder + '/model_last.h5', by_name=True)
    # model.load_weights(outFolder + '/model_best.h5', by_name=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
                      loss=tf.keras.losses.categorical_crossentropy,
                      metrics=['accuracy'])
    model.save(outFolder+'/model_last_simple.h5')


    y_predicted_train = model.predict(inX_train)
    y_predicted = model.predict(inX_test)

    max_display = inX_test.shape[1]

    if not os.path.exists(outFolder+"/"+year):
        os.makedirs(outFolder+"/"+year)
    matplotlib.pyplot.figure(0)
    matplotlib.pyplot.plot(fit.history['output_accuracy'])
    matplotlib.pyplot.plot(fit.history['val_output_accuracy'])
    matplotlib.pyplot.title('model accuracy')
    matplotlib.pyplot.ylabel('accuracy')
    matplotlib.pyplot.xlabel('epoch')
    # matplotlib.pyplot.ylim(ymax = min(fit.history['gradRev_output_loss'])*1.4, ymin = min(fit.history['gradRev_output_loss'])*0.9)
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
    matplotlib.pyplot.figure(2)
    matplotlib.pyplot.plot(fit.history['output_loss'])
    matplotlib.pyplot.plot(fit.history['val_output_loss'])
    matplotlib.pyplot.title('model output loss')
    matplotlib.pyplot.ylabel('output loss')
    matplotlib.pyplot.xlabel('epoch')
    matplotlib.pyplot.ylim(ymax = min(fit.history['output_loss'])*1.4, ymin = min(fit.history['output_loss'])*0.9)
    matplotlib.pyplot.legend(['train', 'validation'], loc='upper right')
    matplotlib.pyplot.savefig(outFolder+"/"+year+"/outputloss.pdf")
    matplotlib.pyplot.figure(3)
    matplotlib.pyplot.plot(fit.history['gradRev_output_loss'])
    matplotlib.pyplot.plot(fit.history['val_gradRev_output_loss'])
    matplotlib.pyplot.title('model gradRev output loss')
    matplotlib.pyplot.ylabel('gradRev output loss')
    matplotlib.pyplot.ylim(ymax = min(fit.history['gradRev_output_loss'])*1.4, ymin = min(fit.history['gradRev_output_loss'])*0.9)
    matplotlib.pyplot.xlabel('epoch')
    matplotlib.pyplot.legend(['train', 'validation'], loc='upper right')
    matplotlib.pyplot.savefig(outFolder+"/"+year+"/gradRevoutputloss.pdf")

    doPlots(outY_train, y_predicted_train ,np.array(targetRho), weights_train_original, lumW_train, proc_train, year = year, outFolder = outFolder+"/train/")
    doPlots(outY_test, y_predicted ,np.array(testRho), weights_test_original, lumW_test, proc_test, year = year, outFolder = outFolder+"/test/")

    class_names=["tt-signal", "tt-bkg", "DY"]

    # for explainer, name  in ((shap.DeepExplainer(model,inX_test[:1000]),"DeepExplainer"), (shap.GradientExplainer(model,inX_test[:1000]),"GradientExplainer")):
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

    saveModel = True
    # saveModel = False
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

def justEvaluate(inPathFolder, additionalName, year, tokeep = None, tokeepReg=None, modelDir = "classOutput/", modelName = "classModel_DEFAULTNAME", outFolder="outFolder_DEFAULTNAME/"):
    inX_train, inX_test, outY_train, outY_test, weights_train, weights_test, lumW_train, lumW_test, scaler = loadData(
        inPathFolder = inPathFolder,
        year=year,
        additionalName = additionalName, testFraction = 0.99,
        overwrite = False, withBTag = True, withCharge = False,
        maxEvents = None)
        # maxEvents = 500000)

    # for i in range(len(feature_names)):
    #     feature_names[i]=feature_names[i].replace(feature_names[i],str(i)+"_"+feature_names[i])
    #
    # feature_names_original = feature_names.copy()
    # feature_names_new = []
    #
    # to_remove = []
    # if tokeep == None:
    #     to_keep = [i for i in range(len(feature_names))] #all
    # else:
    #     to_keep = tokeep
    #
    # for i in range(len(feature_names_original)):
    #     if i in to_keep:
    #         feature_names_new.append(feature_names_original[i])
    #     else:
    #         to_remove.append(i)
    #
    # feature_names = feature_names_new
    #
    # inX_train=np.delete(inX_train,  to_remove,1)
    # inX_test=np.delete(inX_test,  to_remove,1)
    #
    # proc_test = outY_test
    # outY_test_ = []
    # for i in range(len(outY_test)):
    #     if(outY_test[i][0]==0):
    #         outY_test_.append([1,0,0])
    #     elif(outY_test[i][0]==1):
    #         outY_test_.append([1,0,0])
    #     elif(outY_test[i][0]==2):
    #         outY_test_.append([1,0,0])
    #     elif(outY_test[i][0]==3):
    #         outY_test_.append([1,0,0])
    #     elif(outY_test[i][0]==4):
    #         outY_test_.append([0,1,0])
    #     elif(outY_test[i][0]==5):
    #         outY_test_.append([0,0,1])
    #     else:
    #         print (outY_test[i][0],"SHOULD NOT BE HERE!!")
    # outY_test = np.array(outY_test_)

    # feature_names, inX_train, inX_test = helpers.getReducedFeatureNamesAndInputs(inX_train, inX_test, tokeep=tokeep)
    feature_names, inX_train, inX_test, targetRho, testRho = helpers.getReducedFeatureNamesAndInputsWithSecondSet(inX_train, inX_test, tokeep=tokeep, tokeep2=tokeepReg,
        # regDNNPath ="/nfs/dust/cms/user/sewuchte/TopRhoNetwork/preUL04_11_02_21/rhoReg_madgraph/rhoRegModel_preUL04_11_02_21.h5")
        regDNNPath ="/nfs/dust/cms/user/sewuchte/TopRhoNetwork/preUL04_11_02_21_ghost/rhoReg_madgraph_ghost_reweight_38/rhoRegModel_preUL04_11_02_21_ghost.h5")

    testRho = np.array(testRho)

    weights_train_original = weights_train
    weights_test_original = weights_test
    outY_test, outY_train, weights_test, weights_train, proc_test, proc_train = reweight(outY_test, outY_train, weights_test, weights_train)

    max_display = inX_test.shape[1]

    # model = tf.keras.models.load_model("classOutput_Pt30_madgraph/"+"ClassifierFirstTry_preULv04_Pt30_madgraph"+".h5")
    print ("loading model:",modelDir+"/"+modelName+".h5")
    model = tf.keras.models.load_model(modelDir+"/"+modelName+".h5")
    print('inputs: ', [input.op.name for input in model.inputs])
    print('outputs: ', [output.op.name for output in model.outputs])

    y_predicted = model.predict(inX_test)

    doPlots(outY_test, y_predicted, np.array(testRho), weights_test_original, lumW_test, proc_test, year = year, outFolder = outFolder)

def justEvaluateByHand(modelDir = "classOutput/", modelName = "classModel_DEFAULTNAME"):
    Input0 = 40.034
    Input1 = 144.581
    Input2 = 109.198
    Input3 = 0.104598
    Input4 = 0.10565
    Input5 = 106.865
    Input6 = 0.458017
    Input7 = 0
    Input8 = 0
    Input9 = 4

    inX_test=[
        Input0,
        Input1,
        Input2,
        Input3,
        Input4,
        Input5,
        Input6,
        Input7,
        Input8,
        Input9
    ]
    inX_test=np.array([inX_test])
    model = tf.keras.models.load_model(modelDir+"/"+modelName+".h5")
    y_predicted = model.predict(inX_test)
    print (y_predicted)

def main():
    # keep =[10,51,30,39,34,38,0,91,27,41,1,31,13,62,4,3,35,6,58,32,36,52,14,9,17]
    # keep =[10,30,0,39,34,38,35,27,4,17,9,13,3,31]
    # keep =[10,30,0,39,34,38,35,27,4,9]
    # keep =[10,30,0,39,34,38,35,27]
    # keep =[10,53,30,39,34,38,93,31,0,27]
    keep =[10,53,30,39,34,38,93,31,0,27, 102, 103]
    # keep =None
    # doBaysianOptim(inPathFolder = "classInput/madgraph/", additionalName = "_preUL04_11_02_21", year ="FR2", tokeep = [10,53,30,41,39,42,34,38,93,31], outFolder="preUL04_11_02_21/class_madgraph_optim/")
    # doTrainingAndEvaluation(inPathFolder = "classInput/madgraph/", additionalName = "_preUL04_11_02_21", year ="FR2", tokeep = [10,53,30,41,39,42,34,38,93,31], tokeepReg=[41,42,35,60,84,64,30,76,39,88],
    # doTrainingAndEvaluation(inPathFolder = "classInput/madgraph/", additionalName = "_preUL04_11_02_21", year ="FR2", tokeep = keep, tokeepReg=[41,42,35,60,84,64,30,76,39,88],
    #                         modelName = "classInvarianceModel_preUL04_11_02_21", outFolder="preUL04_11_02_21/classInvariance_madgraph/")
    # doTrainingAndEvaluation(inPathFolder = "classInput/madgraph/", additionalName = "_preUL04_11_02_21", year ="FR2", tokeep = keep, tokeepReg=[41,42,35,60,84,64,30,76,39,88],
    #                         modelName = "classInvarianceModel_preUL04_11_02_21", outFolder="preUL04_11_02_21/classInvariance_madgraph_newHybrid/")
    # doTrainingAndEvaluation(inPathFolder = "classInput/madgraph_ghost/", additionalName = "_preUL04_11_02_21", year ="FR2", tokeep = keep, tokeepReg=[41,42,35,60,84,64,30,76,39,88],
    #                         modelName = "classInvarianceModel_preUL04_11_02_21", outFolder="preUL04_11_02_21_ghost/classInvariance_madgraph_ghost_3/")
    # justEvaluate(inPathFolder="classInput/madgraph/", additionalName="_preUL04_moreInputs_Pt30_madgraph_cutSel", year="FR2", tokeep = keep,
    #             modelDir = "classOutput_preUL04_moreInputs_Pt30_madgraph_cutSel/", modelName = "rhoRegModel_preUL04_moreInputs_Pt30_madgraph_cutSel", outFolder="February_final_comparison/rho_madgraph/")
    # justEvaluate(inPathFolder="classInput/powheg/", additionalName="_preUL04_11_02_21", year="FR2", tokeep = keep,tokeepReg=[41,42,35,60,84,64,30,76,39,88],
    #             modelDir = "preUL04_11_02_21/classInvariance_madgraph/", modelName = "classInvarianceModel_preUL04_11_02_21", outFolder="preUL04_11_02_21/classInvariance_powheg/")
    # justEvaluate(inPathFolder="classInput/powheg/", additionalName="_preUL04_11_02_21", year="FR2", tokeep = keep, tokeepReg=[41,42,35,60,84,64,30,76,39,88],
    #             modelDir = "preUL04_11_02_21/classInvariance_madgraph_newHybrid/", modelName = "classInvarianceModel_preUL04_11_02_21", outFolder="preUL04_11_02_21/classInvariance_powheg_newHybrid/")
    # justEvaluate(inPathFolder="classInput/madgraph_ghost/", additionalName="_preUL04_11_02_21", year="FR2", tokeep = keep, tokeepReg=[41,42,35,60,84,64,30,76,39,88],
    #             modelDir = "preUL04_11_02_21_ghost/classInvariance_madgraph_ghost/", modelName = "classInvarianceModel_preUL04_11_02_21", outFolder="preUL04_11_02_21/classInvariance_madgraph_ghost_ANplots/")
    # justEvaluate(inPathFolder="classInput/powheg_ghost/", additionalName="_preUL04_11_02_21", year="FR2", tokeep = keep, tokeepReg=[41,42,35,60,84,64,30,76,39,88],
    #             modelDir = "preUL04_11_02_21_ghost/classInvariance_madgraph_ghost/", modelName = "classInvarianceModel_preUL04_11_02_21", outFolder="preUL04_11_02_21/classInvariance_powheg_ghost_ANplots/")
    # justEvaluate(inPathFolder="classInput/powheg/", additionalName="_preUL04_11_02_21", year="FR2", tokeep = [10,53,30,41,39,42,34,38,93,31],tokeepReg=[41,42,35,60,84,64,30,76,39,88],
    #             modelDir = "preUL04_11_02_21/class_madgraph/", modelName = "rhoRegModel_preUL04_11_02_21", outFolder="preUL04_11_02_21/classInvariance_powheg_nonGradRev/")
    # justEvaluate(inPathFolder="classInput/madgraph/", additionalName="_preUL04_11_02_21", year="FR2", tokeep = [10,53,30,41,39,42,34,38,93,31],tokeepReg=[41,42,35,60,84,64,30,76,39,88],
    #             modelDir = "preUL04_11_02_21/class_madgraph/", modelName = "rhoRegModel_preUL04_11_02_21", outFolder="preUL04_11_02_21/classInvariance_madgraph_nonGradRev/")
    # justEvaluateByHand(modelPath = "classOutput_Pt30_madgraph" , modelName = "ClassifierFirstTry_preULv04_Pt30_madgraph")


    doTrainingAndEvaluation(inPathFolder = "classInput/madgraph/", additionalName = "_preUL05_02_02_22", year ="FR2", tokeep = keep, tokeepReg=[41,42,35,60,84,64,30,76,39,88, 102, 103],
                            # modelName = "classInvarianceModel_preUL05_02_02_22", outFolder="preUL05_02_02_22/classInvariance_madgraph/")
                            # modelName = "classInvarianceModel_preUL05_02_02_22", outFolder="preUL05_02_02_22_2/classInvariance_madgraph/")
                            modelName = "classInvarianceModel_preUL05_02_02_22", outFolder="preUL05_02_02_22_3/classInvariance_madgraph/")
                            # modelName = "classInvarianceModel_preUL05_02_02_22", outFolder="preUL05_02_02_22_4/classInvariance_madgraph/")

if __name__ == "__main__":
    main()
