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
import matplotlib
matplotlib.use('Agg') # Save the plot to a file instead of displaying it on the screen.
os.environ['QT_QPA_PLATFORM']='offscreen'
import shap
import smogn
import pandas
from sklearn import preprocessing
from scipy.stats.stats import pearsonr
from scipy.stats.mstats import gmean
from scipy.stats import chisquare
from array import array
from scipy.spatial.distance import jensenshannon
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if(len(physical_devices)>0):
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.compat.v1.disable_eager_execution()

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


# stupid way to handle bayesian opt
# global bayesian_base_outFolder
bayesian_base_outFolder= ""
# global bayesian_current_iteration
bayesian_current_iteration = 0

import ROOT

@jit
def compute_probs(data, n=10):
    h, e = np.histogram(data, n)
    p = h/data.shape[0]
    return e, p

@jit
def support_intersection(p, q):
    sup_int = (
        list(
            filter(
                lambda x: (x[0]!=0) & (x[1]!=0), zip(p, q)
            )
        )
    )
    return sup_int

@jit
def get_probs(list_of_tuples):
    p = np.array([p[0] for p in list_of_tuples])
    q = np.array([p[1] for p in list_of_tuples])
    return p, q

@jit
def kl_divergence(p, q):
    return np.sum(p*np.log(p/q))

@jit
def js_divergence(p, q):
    m = (1./2.)*(p + q)
    return (1./2.)*kl_divergence(p, m) + (1./2.)*kl_divergence(q, m)

@jit
def compute_kl_divergence(train_sample, test_sample, n_bins=10):
    """
    Computes the KL Divergence using the support
    intersection between two different samples
    """
    e, p = compute_probs(train_sample, n=n_bins)
    _, q = compute_probs(test_sample, n=e)

    list_of_tuples = support_intersection(p, q)
    p, q = get_probs(list_of_tuples)

    return kl_divergence(p, q)

@jit
def compute_js_divergence(train_sample, test_sample, n_bins=10):
    """
    Computes the JS Divergence using the support
    intersection between two different samples
    """
    e, p = compute_probs(train_sample, n=n_bins)
    _, q = compute_probs(test_sample, n=e)

    list_of_tuples = support_intersection(p,q)
    p, q = get_probs(list_of_tuples)

    return js_divergence(p, q)

ar_bins = array('d',[340, 400, 500, 650, 1500])
nbin = len(ar_bins)-1
logbins = array('d', np.concatenate( ( np.linspace(340, 500, 32, endpoint=False), [500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 625, 650, 675, 750, 800, 900, 1000, 1500])))  #used for 2d histograms
nlogbin = len(logbins)-1
neg_bins = array('d', np.concatenate((np.concatenate((np.linspace(-1500, -750, 50, endpoint=False), np.linspace(-750, -650, 50, endpoint=False))), np.linspace(-650, -340, 311)) ))  #used for 2d histograms
#logbins_diff = np.concatenate(([-200, -100, -50],np.concatenate((np.linspace(-20, 20, 40), [50, 100, 200]))))
logbins_diff = np.linspace(-100, 100, 200)
nlogbin_diff =len(logbins_diff)-1


def loadData(inPathFolder = "rhoInput/years", year = "2016", additionalName = "_3JetKinPlusRecoSolRight", testFraction = 0.4, overwrite = False, withBTag = True, pTEtaPhiMode=True, maxEvents = None):
    '''
    Load data from saved numpy arrays or create them if not available (using loadRegressionData)
    '''
    print ("loadData for year "+year+"/"+" from "+inPathFolder+year+"/flat_[..]"+additionalName+".npy")
    print ("\t overwrite = "+str(overwrite))
    print ("\t testFraction = "+str(testFraction))
    print ("\t withBTag = "+str(withBTag))
    print ("\t pTEtaPhiMode = "+str(pTEtaPhiMode))
    print ("\t maxEvents = "+str(maxEvents))

    loadData = False
    if not os.path.exists(inPathFolder+year+"/flat_inX"+additionalName+".npy"):
        loadData = True
        print("*** Not found the data in the right directory ***")
    else:
        print("\nData found in the directory :"+inPathFolder+year)
        loadData = overwrite

    if loadData:
        nJets=2
        print ("\t need to load from root file with settings: nJets >= "+str(nJets)+"; max Events = "+str(maxEvents))
        inX, outY, weights, lkrM, krM = helpers.loadRegressionData(inPathFolder+year+"/", "miniTree", nJets = 2, maxEvents = 0 if maxEvents==None else maxEvents, withBTag=withBTag, pTEtaPhiMode=pTEtaPhiMode)
        inX=np.array(inX)
        outY=np.array(outY)
        weights=np.array(weights)
        lkrM=np.array(lkrM)
        krM=np.array(krM)
        np.save(inPathFolder+year+"/flat_inX"+additionalName+".npy", inX)
        np.save(inPathFolder+year+"/flat_outY"+additionalName+".npy", outY)
        np.save(inPathFolder+year+"/flat_weights"+additionalName+".npy", weights)
        np.save(inPathFolder+year+"/flat_lkrM"+additionalName+".npy", lkrM)
        np.save(inPathFolder+year+"/flat_krM"+additionalName+".npy", krM)

    print("*** inX, outY, weights, lkrM, krM loading")
    inX = np.load(inPathFolder+year+"/flat_inX"+additionalName+".npy")
    outY = np.load(inPathFolder+year+"/flat_outY"+additionalName+".npy")
    weights = np.load(inPathFolder+year+"/flat_weights"+additionalName+".npy")
    lkrM = np.load(inPathFolder+year+"/flat_lkrM"+additionalName+".npy")
    krM = np.load(inPathFolder+year+"/flat_krM"+additionalName+".npy")
    print("*** inX, outY, weights, lkrM, krM loaded")

    if maxEvents is not None:
        inX = inX[:maxEvents]
        outY = outY[:maxEvents]
        weights = weights[:maxEvents]
        krM = krM[:maxEvents]

    #outY = outY.reshape((outY.shape[0], 1))
    print('\n\nShapes of all the data at my disposal:\nInput  \t',inX.shape,'\nOutput\t', outY.shape,'\nWeights\t', weights.shape,'\nLoose M\t', lkrM.shape,'\nFull M\t', krM.shape)
    # transform y to uniform
# Q? What's the scaler
    print("Computing the scaler of the output")
    scaler = preprocessing.StandardScaler().fit(outY)

    inX_train, inX_test, outY_train, outY_test, weights_train, weights_test, lkrM_train, lkrM_test, krM_train, krM_test = train_test_split(inX, outY, weights, lkrM, krM, test_size = testFraction, random_state=1999)
# Add random_state to replicate the output. Shuffle is true by default. random_state is a seed
# Weights, Input and Output are splitted keeping the correspondences. The same applies for training and testing

    print ("\tData splitted succesfully")
    print("Number training events :", inX_train.shape[0], len(inX_train))
    print("Number of features     :", inX_train.shape[1])
    print("loadData ended returning inX_train and so on...")

    return inX_train, inX_test, outY_train, outY_test, weights_train, weights_test,lkrM_train, lkrM_test, krM_train, krM_test, scaler


def doEvaluationPlots(yTest, yPredicted, weightTest, lkrM, krM, year = "", outFolder = "MOutput/"):

    outDir = outFolder+"/"+year+"/"
    if not os.path.exists(outDir):
        os.makedirs(outDir)

    print("Invariant Mass plot: Predicted vs True")
    f = matplotlib.pyplot.figure()
    values, bins, patches = matplotlib.pyplot.hist(yTest, bins=100, range=(340,1500), label="True", alpha=0.5)
    values2, bins2, patches2 = matplotlib.pyplot.hist(yPredicted, bins=100, range=(340,1500), label="Reco", alpha=0.5)
    matplotlib.pyplot.legend(loc = "best")
    matplotlib.pyplot.ylabel('Events')
    matplotlib.pyplot.xlabel('m_{tt} (GeV/c)')
    f.savefig(outDir+"m_tt.pdf")

    #ROOT.gStyle.SetPalette(ROOT.kThermometer)

# from utils style: style.style1d(): divisions, labels offsets, margins ...
    style.style1d()
    s = style.style1d()
# 10 histograms defined for shapes.pdf output
    
   
    histoDNN  = ROOT.TH1F( "reco shape", ";m_{tt}", 50, 340, 1500)
    histoTrue = ROOT.TH1F( "true", ";m_{tt}", 50, 340, 1500)
    histoLoose = ROOT.TH1F( "lkr", ";m_{tt}", 50, 340, 1500)
    histoKinReco = ROOT.TH1F( "kr", ";m_{tt}", 50, 340, 1500)
    histoShape5 = ROOT.TH1F( "true shape no kinReco", ";m_{tt}", 50, 340, 1500)
    histoShape6 = ROOT.TH1F( "reg shape no kinReco", ";m_{tt}", 50, 340, 1500)
    histoShape8 = ROOT.TH1F( "new hybrid shape", ";m_{tt}", 50, 340, 1500)
    histoShape9 = ROOT.TH1F( "new hybrid2 shape", ";m(tt)", 50, 340, 1500)

    for Mtrue, Mpred, weight, lkrMass, krMass in zip(yTest, yPredicted, weightTest, lkrM, krM):
        #print("prova")
        if lkrMass>0.:
            histoLoose.Fill(lkrMass, weight)
        if krMass >0.:
            histoKinReco.Fill(krMass, weight)
        else:
            histoShape6.Fill(Mpred, weight)
            histoShape5.Fill(Mtrue, weight)
        histoDNN.Fill(Mpred, weight)
        mAverage = []
        mAverage.append(Mpred)
        if lkrMass>0.:
            mAverage.append(lkrMass)
        if krMass>0.:
            mAverage.append(krMass)
        mAv = np.mean(mAverage)
        mAverage = np.array(mAverage, dtype=float)
        mAv2 = gmean(mAverage)
        histoShape8.Fill(mAv, weight)
        histoShape9.Fill(mAv2, weight)
        histoTrue.Fill(Mtrue, weight)


    histoShape7 = histoShape6.Clone()
    histoShape7.Add(histoKinReco)

    max_ = 1.4*max(histoDNN.GetMaximum(),histoTrue.GetMaximum(),histoLoose.GetMaximum(),histoKinReco.GetMaximum(),histoShape5.GetMaximum(),histoShape6.GetMaximum(),histoShape7.GetMaximum())
    histoDNN.SetMaximum(max_)
    histoTrue.SetMaximum(max_)
    histoLoose.SetMaximum(max_)
    histoKinReco.SetMaximum(max_)
    histoShape5.SetMaximum(max_)
    histoShape6.SetMaximum(max_)
    histoShape7.SetMaximum(max_)
    histoShape8.SetMaximum(max_)
    histoShape9.SetMaximum(max_)

    histoDNN.SetLineWidth(2)
    histoTrue.SetLineWidth(2)
    histoLoose.SetLineWidth(2)
    histoKinReco.SetLineWidth(2)
    histoShape5.SetLineWidth(2)
    histoShape6.SetLineWidth(2)
    histoShape7.SetLineWidth(2)
    histoShape8.SetLineWidth(2)
    histoShape9.SetLineWidth(2)

    c0=ROOT.TCanvas("c0","c0",800,800)
    histoDNN.SetLineColor(ROOT.kRed)
    histoDNN.Draw("h0")
    histoTrue.Draw("h0same")
    histoLoose.SetLineColor(ROOT.kOrange+1)
    histoLoose.Draw("h0same")
    histoKinReco.SetLineColor(ROOT.kGreen+1)
    histoKinReco.Draw("h0same")
    histoShape5.SetLineColor(ROOT.kRed)
    histoShape5.SetLineStyle(ROOT.kDashed)
    # histoShape5.Draw("h0same")
    histoShape6.SetLineColor(ROOT.kBlack)
    histoShape6.SetLineStyle(ROOT.kDashed)
    histoShape6.Draw("h0same")
    histoShape7.SetLineColor(ROOT.kBlue)
    histoShape7.SetLineStyle(ROOT.kDashed)
    histoShape7.Draw("h0same")
    histoShape8.SetLineColor(ROOT.kOrange)
    histoShape8.SetLineStyle(ROOT.kDashed)
    histoShape8.Draw("h0same")
    histoShape9.SetLineColor(ROOT.kCyan)
    histoShape9.SetLineStyle(ROOT.kDashed)
    histoShape9.Draw("h0same")
    l = ROOT.TLegend(0.7,0.7,0.9,0.9)
    l.AddEntry(histoDNN, "DNN")
    l.AddEntry(histoTrue, "truth")
    l.AddEntry(histoLoose, "LKR")
    l.AddEntry(histoKinReco, "KR")
    # l.AddEntry(histoShape5, "DNN(noKR)")
    l.AddEntry(histoShape6, "truth(noKR)")
    l.AddEntry(histoShape7, "KR + reg (noKR)")
    l.AddEntry(histoShape8, "newHybrid")
    l.AddEntry(histoShape9, "newHybrid2")
    l.Draw()
    c0.SaveAs(outDir+"shapes.pdf")
    
    histoDNN.Write("shape_reg")
    histoTrue.Write("shape_true")
    histoLoose.Write("shape_lkr")
    histoKinReco.Write("shape_kr")
    histoShape5.Write("shape_trueNoKR")
    histoShape6.Write("shape_regNoKR")
    histoShape7.Write("shape_KRPlusRegNoKR")
    histoShape8.Write("shape_newHybrid")
    histoShape9.Write("shape_newHybrid2")
    
    style.style2d()
    s = style.style2d()
    
    histo = ROOT.TH2F( "histo", 		"DNNMinusTrue_vs_True;		m_{tt}^{true};		m_{tt}^{DNN} - m_{tt}^{true}"	  	,nlogbin, logbins, nlogbin_diff, logbins_diff )	
    histo_kr = ROOT.TH2F( "histo_kr", 		"krMinusTrue_vs_True;		m_{tt}^{true};		m_{tt}^{kinReco} - m_{tt}^{true}"	,nlogbin, logbins, nlogbin_diff,logbins_diff )		
    histo_krReg = ROOT.TH2F( "histo_krReg", 	"krRegMinusTrue_vs_True;	m_{tt}^{true};		m_{tt}^{kinReco} - m_{tt}^{true}"	,nlogbin, logbins, nlogbin_diff, logbins_diff )
    histo_lkr = ROOT.TH2F( "histo_lkr", 	"lrMinusTrue_vs_True;		m_{tt}^{true};		m_{tt}^{LooseReco} - m_{tt}^{true}"	,nlogbin, logbins, nlogbin_diff, logbins_diff )
    histTrue = ROOT.TH1F( "histTrue", 		"m_{tt}^{true} distribution;	m_{tt}^{true}", nlogbin, logbins)
# two histos I remove
#    histo_newHybrid = ROOT.TH2F( "histo_newHyb","newHybrid m_{tt} diff;	m_{tt}^{kinReco};	m_{tt}^{true} - m_{tt}^{kinReco}"	,nlogbin, logbins, nlogbin_diff, logbins_diff )
#    histo_newHybrid2 = ROOT.TH2F( "newHybrid2", "Title;				m_{tt}^{kinReco};	m_{tt}^{true} - m_{tt}^{kinReco}"	,nlogbin, logbins, nlogbin_diff, logbins_diff )
# Q? histo?krReg is the same of histo_kr when kr works optherwise replaced by NN solution, what's the point?
# Why not the same for LooseKinReco


	# Detach the object from the file. When the files is closed the object is not deleted
    histos = [histo, histo_kr, histTrue, histo_krReg, histo_lkr]# histo_newHybrid2, histo_newHybrid]
    for i in histos:
        i.SetDirectory(0)
    

    histoRecoGen = ROOT.TH2F( "histoRecoGen", "			;m_{tt} reco;		m_{tt}^{true}",nlogbin, logbins, nlogbin, logbins )
    histoRecoGen_kr = ROOT.TH2F( "histoRecoGen_kr", "		;m_{tt}^{kinReco};	m_{tt}^{true}",nlogbin, logbins, nlogbin, logbins )
    histoRecoGen_krReg = ROOT.TH2F( "histoRecoGen_krReg", "	;m_{tt}^{kinReco};	m_{tt}^{true}",nlogbin, logbins, nlogbin, logbins )
    histoRecoGen_lkr = ROOT.TH2F( "histoRecoGen_lkr", "		;m_{tt}^{kinReco};	m_{tt}^{true}",nlogbin, logbins, nlogbin, logbins )
#    histoRecoGen_newHybrid = ROOT.TH2F( "newHybrid m(tt)2d", ";m(tt) reco;m_{tt}^{true}",nlogbin, logbins, nlogbin, logbins )
#    histoRecoGen_newHybrid2 = ROOT.TH2F( "newHybrid2 m(tt)2d", ";m(tt) reco;m_{tt}^{true}",nlogbin, logbins, nlogbin, logbins )
    histoRecoGen2 = ROOT.TH2F( "reg resp", ";reco bin		;True Bin", nlogbin, logbins, nlogbin, logbins)
    histoRecoGen2_kr = ROOT.TH2F( "kr resp", ";reco bin		;True Bin", nlogbin, logbins, nlogbin, logbins)
    histoRecoGen2_krReg = ROOT.TH2F( "kr+reg resp", ";reco bin	;True Bin", nlogbin, logbins, nlogbin, logbins)
    histoRecoGen2_lkr = ROOT.TH2F( "lkr resp", ";reco bin	;True Bin", nlogbin, logbins, nlogbin, logbins)
#    histoRecoGen2_newHybrid = ROOT.TH2F( "newHybrid resp", ";reco bin;True Bin", 20 ,340, 1500, 20 ,340, 1500)
#    histoRecoGen2_newHybrid2 = ROOT.TH2F( "newHybrid2 resp", ";reco bin;True Bin", 20 ,340, 1500, 20 ,340, 1500)
    
    histoRecoGenList  = [histoRecoGen, histoRecoGen_kr, histoRecoGen_krReg, histoRecoGen_lkr] 
    histoRecoGenList2 = [histoRecoGen2, histoRecoGen2_kr, histoRecoGen2_krReg, histoRecoGen2_lkr]
    for i in histoRecoGenList:
        i.SetDirectory(0)
    for i in histoRecoGenList2:
        i.SetDirectory(0)


# HYBRID
#    histoRecoGen2_newHybrid.SetDirectory(0)
#    histoRecoGen2_newHybrid2.SetDirectory(0)
    
    histoResp = ROOT.TH2F( "resp class", ";m_{tt}^{true};m_{tt}^{DNN}", nbin, ar_bins, nbin, ar_bins )
#    histo2dbinned_newHybrid = ROOT.TH2F( "resp class newHybrid", ";m_{tt}^{true};m_{tt} reco", nbin, ar_bins, nbin, ar_bins )
    histTrueBinned = ROOT.TH1F( "true contents binned", ";m_{tt}", nbin, ar_bins)
    histTrueBinned.SetDirectory(0)


    

# ***************************************
# *					*
# *    Filling all the histograms     	*	
# *					*
# ***************************************
    print("-------------------------------------\n|    Filling all the histograms     |\n-------------------------------------")

    for Mtrue, Mpred, weight, lkrMass, krMass in zip(yTest, yPredicted, weightTest, lkrM, krM):
        diff = Mpred-Mtrue
        histo.Fill(Mtrue, diff, weight)
        histoRecoGen.Fill( Mpred, Mtrue, weight)
        histoRecoGen2.Fill(Mpred, Mtrue, weight)
                # histo2dbinned.Fill(Mpred, Mtrue, weight)
        histoResp.Fill(Mtrue, Mpred, weight)
        histTrue.Fill(Mtrue, weight)
# Fill histograms of kinReco
        if krMass>0.:
            diff_kr = krMass - Mtrue
            histo_kr.Fill(Mtrue, diff_kr, weight)
            histo_krReg.Fill(Mtrue, diff_kr, weight)
            histoRecoGen_kr.Fill(krMass, Mtrue, weight)
            histoRecoGen2_kr.Fill(krMass, Mtrue, weight)
            histoRecoGen_krReg.Fill(krMass, Mtrue, weight)
            histoRecoGen2_krReg.Fill(krMass, Mtrue, weight)
        else:
            histo_krReg.Fill(Mtrue, diff, weight)
            histoRecoGen_krReg.Fill(Mpred, Mtrue, weight)
            histoRecoGen2_krReg.Fill(Mpred, Mtrue, weight)
        if lkrMass>0.:
            diff_lkr = lkrMass-Mtrue
            histo_lkr.Fill(Mtrue, diff_lkr, weight)
            histoRecoGen_lkr.Fill(lkrMass, Mtrue, weight)
            histoRecoGen2_lkr.Fill(lkrMass, Mtrue, weight)
# HYBRID
#        mAverage = []
#        mAverage.append(Mpred)
#        if lkrMass>0.:
#            mAverage.append(lkrMass)
#        if krMass>0.:
#            mAverage.append(krMass)
#        mAv = np.mean(mAverage)			# average of krMass (when available), lkrmass (when available), mpred
# Q? is this legit? To have an average of things that are correlated since they start from the same variables
#        mAverage = np.array(mAverage, dtype=float)
#        histo2dbinned_newHybrid.Fill(Mtrue, mAv, weight)
#        mAv2 = gmean(mAverage)			# geometric mean
#        diff_av = Mtrue-mAv
#        diff_av2 = Mtrue-mAv2
#        histo_newHybrid.Fill(Mtrue, diff_av, weight)
#        histoRecoGen_newHybrid.Fill(mAv, Mtrue, weight)
#        histoRecoGen2_newHybrid.Fill(mAv, Mtrue, weight)
#        histo_newHybrid2.Fill(Mtrue, diff_av2, weight)
#        histoRecoGen_newHybrid2.Fill(mAv2, Mtrue, weight)
#        histoRecoGen2_newHybrid2.Fill(mAv2, Mtrue, weight)
    #print("******************\n\n\n**********************\n\n\nEntries: \t",histoRecoGen2.GetEntries(),"\n\n\n***********************\n\n\n")
    hResp = histoRecoGen2.Clone()
    hResp_kr = histoRecoGen2_kr.Clone()
    hResp_lkr = histoRecoGen2_lkr.Clone()
    hResp_krReg = histoRecoGen2_krReg.Clone()
#    hResp_newHybrid = histoRecoGen2_newHybrid.Clone()
#    hResp_newHybrid2 = histoRecoGen2_newHybrid2.Clone()

    for i in histos:
        i = NormalizeBinContent(i)
    for i in histoRecoGenList:
        i = NormalizeBinContent(i)
# histoRecoGenList2 must not be normalized by the area because they give transition probabilites between bins. it is reasonable that larger bins will ahve larger proabhailtiy(?)


# ---------------------------------------
# |    Draw and Save the histograms    	|	
# ---------------------------------------
    c=ROOT.TCanvas("c1","c1",800,800)
    histo.SetStats(0)
    histo.Draw("colz")
    c.SetLogx()
    c.SaveAs(outDir+"reg_GenRecoDiff2d.pdf")
    c.Clear()
    histo_kr.SetStats(0)
    histo_kr.Draw("colz")
    c.SaveAs(outDir+"kr_GenRecoDiff2d.pdf")
    c.Clear()
    histo_krReg.SetStats(0)
    histo_krReg.Draw("colz")
    c.SaveAs(outDir+"krReg_GenRecoDiff2d.pdf")
    c.Clear()
    histo_lkr.SetStats(0)
    histo_lkr.Draw("colz")
    c.SaveAs(outDir+"lkr_GenRecoDiff2d.pdf")
    c.Clear()
#    histo_newHybrid.SetStats(0)
#    histo_newHybrid.Draw("colz")
#    c.SaveAs(outDir+"newHybrid_GenRecoDiff2d.pdf")
#    c.Clear()
#    histo_newHybrid2.SetStats(0)
#    histo_newHybrid2.Draw("colz")
#    c.SaveAs(outDir+"newHybrid2_GenRecoDiff2d.pdf")
#    c.Clear()

    histoRecoGen.Draw("colz")
    c.SetLogy()
    corrLatex = ROOT.TLatex()
    corrLatex.SetTextSize(0.65 * corrLatex.GetTextSize())
    corrLatex.DrawLatexNDC(0.65, 0.85, str(np.round(histoRecoGen.GetCorrelationFactor(),3)))
    c.SaveAs(outDir+"reg_GenReco2d.pdf")
    c.Clear()
    
    histoRecoGen_kr.Draw("colz")
    corrLatex_kr = ROOT.TLatex()
    corrLatex_kr.SetTextSize(0.65 * corrLatex_kr.GetTextSize())
    corrLatex_kr.DrawLatexNDC(0.65, 0.85, str(np.round(histoRecoGen_kr.GetCorrelationFactor(),3)))
    c.SaveAs(outDir+"kr_GenReco2d.pdf")
    c.Clear()
    c.SetLogy(False)
    histoRecoGen_krReg.Draw("colz")
    corrLatex_krReg = ROOT.TLatex()
    corrLatex_krReg.SetTextSize(0.65 * corrLatex_kr.GetTextSize())
    corrLatex_krReg.DrawLatexNDC(0.65, 0.85, str(np.round(histoRecoGen_kr.GetCorrelationFactor(),3)))
    c.SaveAs(outDir+"krReg_GenReco2d.pdf")
    c.Clear()
    histoRecoGen_lkr.Draw("colz")
    corrLatex_lkr = ROOT.TLatex()
    corrLatex_lkr.SetTextSize(0.65 * corrLatex_lkr.GetTextSize())
    corrLatex_lkr.DrawLatexNDC(0.65, 0.85, str(np.round(histoRecoGen_lkr.GetCorrelationFactor(),3)))
    c.SaveAs(outDir+"lkr_GenReco2d.pdf")
    c.Clear()
#    histoRecoGen_newHybrid.Draw("colz")
#    corrLatex_newHybrid = ROOT.TLatex()
#    corrLatex_newHybrid.SetTextSize(0.65 * corrLatex_newHybrid.GetTextSize())
#    corrLatex_newHybrid.DrawLatexNDC(0.65, 0.85, str(np.round(histoRecoGen_newHybrid.GetCorrelationFactor(),3)))
#    c.SaveAs(outDir+"newHybrid_GenReco2d.pdf")
#    c.Clear()
#    histoRecoGen_newHybrid2.Draw("colz")
#    corrLatex_newHybrid2 = ROOT.TLatex()
#    corrLatex_newHybrid2.SetTextSize(0.65 * corrLatex_newHybrid2.GetTextSize())
#    corrLatex_newHybrid2.DrawLatexNDC(0.65, 0.85, str(np.round(histoRecoGen_newHybrid2.GetCorrelationFactor(),3)))
#    c.SaveAs(outDir+"newHybrid2_GenReco2d.pdf")
#    c.Clear()
    
    histoResp.SetStats(0)
    hRespBinned = histoResp.Clone()
    histoResp.Scale(1./(histoResp.Integral()))
    print ("binned correlation", histoResp.GetCorrelationFactor())

    # from array import array
    nbinsx = histoResp.GetNbinsX()
    nbinsy = histoResp.GetNbinsY()

    ma = np.empty((nbinsx,nbinsy))
    ma2 = np.empty((nbinsx,nbinsy))
    for iBinX in range(1,nbinsx+1):
        for iBinY in range(1,nbinsy+1):
            ma[iBinX-1,iBinY-1]=histoResp.GetBinContent(iBinX,iBinY)
            ma2[iBinY-1,iBinX-1]=histoResp.GetBinContent(iBinX,iBinY)
    print ("Condition number = ",np.linalg.cond(ma))

    histoResp.Scale(100.)


    ROOT.gStyle.SetPaintTextFormat("2.2f")
    histoResp.SetMarkerSize(0.7)
    histoResp.GetZaxis().SetTitle("Transition Probability [%]")
    # style.setPalette("bird")
    #ROOT.gStyle.SetPalette(ROOT.kThermometer)
    # ROOT.gStyle.SetPaintTextFormat("1.2f");
    histoResp.Draw("colz text")
    c.SetLogy()
    c.SaveAs(outDir+"reg_response.pdf")

    c.Clear()

#HYBRID
    '''histo2dbinned_newHybrid.SetStats(0)
    hRespBinned_newHybrid = histo2dbinned_newHybrid.Clone()

    histo2dbinned_newHybrid.Scale(1./(histo2dbinned_newHybrid.Integral()))

    # from array import array
    nbinsx = histo2dbinned_newHybrid.GetNbinsX()
    nbinsy = histo2dbinned_newHybrid.GetNbinsY()

    ma = np.empty((nbinsx,nbinsy))
    ma2 = np.empty((nbinsx,nbinsy))
    for iBinX in range(1,nbinsx+1):
        for iBinY in range(1,nbinsy+1):
            ma[iBinX-1,iBinY-1]=histo2dbinned_newHybrid.GetBinContent(iBinX,iBinY)
            ma2[iBinY-1,iBinX-1]=histo2dbinned_newHybrid.GetBinContent(iBinX,iBinY)
    print ("neHybrid Condition number = ",np.linalg.cond(ma))


    histo2dbinned_newHybrid.Scale(100.)

    ROOT.gStyle.SetPaintTextFormat("2.2f")
    histo2dbinned_newHybrid.SetMarkerSize(0.7)
    histo2dbinned_newHybrid.GetZaxis().SetTitle("Transition Probability [%]")
    #ROOT.gStyle.SetPalette(ROOT.kThermometer)
    # ROOT.gStyle.SetPaintTextFormat("1.2f");
    histo2dbinned_newHybrid.Draw("colz text")
    c.SaveAs(outDir+"newHybrid_response.pdf")

    c.Clear()'''

    histoRecoGen2.Scale(1./histoRecoGen2.Integral())
    # histoRecoGen2.Scale(100.)
    ROOT.gStyle.SetPaintTextFormat("1.2f");
    histoRecoGen2.SetStats(0)
    histoRecoGen2.SetMarkerSize(1)
    histoRecoGen2.Draw("colz text")
    c.SaveAs(outDir+"resp.pdf")
    c.Clear()
    histoRecoGen2_kr.Scale(1./histoRecoGen2_kr.Integral())
    ROOT.gStyle.SetPaintTextFormat("1.2f");
    histoRecoGen2_kr.SetStats(0)
    histoRecoGen2_kr.SetMarkerSize(1)
    histoRecoGen2_kr.Draw("colz text")
    c.SaveAs(outDir+"kr_resp.pdf")
    c.Clear()
    histoRecoGen2_krReg.Scale(1./histoRecoGen2_krReg.Integral())
    ROOT.gStyle.SetPaintTextFormat("1.2f");
    histoRecoGen2_krReg.SetStats(0)
    histoRecoGen2_krReg.SetMarkerSize(1)
    histoRecoGen2_krReg.Draw("colz text")
    c.SaveAs(outDir+"krReg_resp.pdf")
    c.Clear()
    histoRecoGen2_lkr.Scale(1./histoRecoGen2_lkr.Integral())
    ROOT.gStyle.SetPaintTextFormat("1.2f");
    histoRecoGen2_lkr.SetStats(0)
    histoRecoGen2_lkr.SetMarkerSize(1)
    histoRecoGen2_lkr.Draw("colz text")
    c.SaveAs(outDir+"lkr_resp.pdf")
    c.Clear()
# HYBRID
#    histoRecoGen2_newHybrid.Scale(1./histoRecoGen2_newHybrid.Integral())
    # histoRecoGen2_lkr.Scale(100.)
#    ROOT.gStyle.SetPaintTextFormat("1.2f");
    # ROOT.gStyle.SetPaintTextFormat("1.0f");
#    histoRecoGen2_newHybrid.SetStats(0)
#    histoRecoGen2_newHybrid.SetMarkerSize(1)
#    histoRecoGen2_newHybrid.Draw("colz text")
#    c.SaveAs(outDir+"newHybrid_resp.pdf")
#    c.Clear()

# Q? RMS and mean based on histograms if you cut data you must know it
    rms_reg, mean_reg, respRMS_reg = doRMSandMean(histo, "reg", outDir)
    rms_kr, mean_kr, respRMS_kr = doRMSandMean(histo_kr, "kr", outDir)
    rms_krReg, mean_krReg, respRMS_krReg = doRMSandMean(histo_krReg, "krReg", outDir)
    rms_lkr, mean_lkr, respRMS_lkr = doRMSandMean(histo_lkr, "lkr", outDir)
#    rms_newHybrid, mean_newHybrid, respRMS_newHybrid = doRMSandMean(histo_newHybrid, "newHybrid", outDir)
#    rms_newHybrid2, mean_newHybrid2, respRMS_newHybrid2 = doRMSandMean(histo_newHybrid2, "newHybrid2", outDir)


#GC new plot for all the means together
    style.style1d()
    s = style.style1d()
    c.SetLogx(False)
    c.SetLogy(False)
    c.Clear()
    mean_reg.SetStats(0)
    mean_kr.SetStats(0)
    mean_lkr.SetStats(0)
    mean_krReg.SetStats(0)
    mean_reg.Draw("histe")
    mean_kr.Draw("same")
    mean_lkr.Draw("same")
    mean_krReg.Draw("same")
    c.SaveAs(outDir+"allMeans.pdf")
    

    
    
    purity_kr, stability_kr, eff_kr = doPSE(hResp_kr, histTrue, "kr", outDir)
# Q? why not for lkr?

# put above
#    style.style1d()
#    s = style.style1d()

#    c.SetLogx(False)
#    c.SetLogy(False)
    c.Clear()

    g_purity = helpers.purityStabilityGraph(hRespBinned, 0)
    g_stability = helpers.purityStabilityGraph(hRespBinned, 1)

    # Styling axis
    histo_ = hResp.ProjectionX(hResp.GetName()+"_px")
    histo_.SetAxisRange(0., 1.1, "Y")
    histo_.GetYaxis().SetTitle("a.u.")
    ROOT.gPad.SetRightMargin(0.05)
    histo_.Draw("AXIS")

    # Styling graphs
    setGraphStyle(g_purity,       1, 1, 2, 21, 1, 1)
    setGraphStyle(g_stability,    1, 2, 2, 20, 2, 1)

    # Drawing graphs
    g_purity.Draw("P0same")
    g_stability.Draw("P0same")
    ROOT.gPad.Update()

    # Adding a legend
    leg = ROOT.TLegend(.66, .85, .94, .9)
    leg.AddEntry(g_purity, "Purity", "p")
    leg.AddEntry(g_stability, "Stability", "p")
    leg.Draw()

    c.SaveAs(outDir+"reg_pse.pdf")

    outRootFile = ROOT.TFile(outDir+"/AllPlots.root","RECREATE")
    outRootFile.cd()
    
    histo.Write("true_vs_reg")
    histo_kr.Write("true_vs_kr")
    histo_lkr.Write("true_vs_lkr")
    histo_krReg.Write("true_vs_krReg")
#    histo_newHybrid.Write("true_vs_newHybrid")
#    histo_newHybrid2.Write("true_vs_newHybrid2")
    histTrue.Write("true")
    histoRecoGen.Write("recoGen_reg")
    histoRecoGen2.Write("recoGen2_reg")
    histoRecoGen_kr.Write("recoGen_kr")
    histoRecoGen_lkr.Write("recoGen_lkr")
    histoRecoGen_krReg.Write("recoGen_krReg")
#    histoRecoGen_newHybrid.Write("recoGen_newHybrid")
    histoRecoGen2_krReg.Write("recoGen2_krReg")
    histoRecoGen2_lkr.Write("recoGen2_lkr")
    histoRecoGen2_kr.Write("recoGen2_kr")
#    histoRecoGen2_newHybrid.Write("recoGen2_newHybrid")
#    histoRecoGen2_newHybrid2.Write("recoGen2_newHybrid2")
    rms_reg.Write("rms_reg")
    respRMS_reg.Write("respRMS_reg")
    mean_reg.Write("mean_reg")
    rms_kr.Write("rms_kr")
    respRMS_kr.Write("respRMS_kr")
    mean_kr.Write("mean_kr")
    rms_krReg.Write("rms_krReg")
    respRMS_krReg.Write("respRMS_krReg")
    mean_krReg.Write("mean_krReg")
    rms_lkr.Write("rms_lkr")
    respRMS_lkr.Write("respRMS_lkr")
    mean_lkr.Write("mean_lkr")
#    rms_newHybrid.Write("rms_newHybrid")
#    respRMS_newHybrid.Write("respRMS_newHybrid")
#    mean_newHybrid.Write("mean_newHybrid")
#    rms_newHybrid2.Write("rms_newHybrid2")
#    respRMS_newHybrid2.Write("respRMS_newHybrid2")
#    mean_newHybrid2.Write("mean_newHybrid2")
    purity_kr.Write("purity_kr")
    stability_kr.Write("stability_kr")
    eff_kr.Write("efficiency_kr")
    outRootFile.Close()

''' called in main
doTrainingAndEvaluation(inPathFolder = "/nfs/dust/cms/user/celottog/TopRhoNetwork/rhoInput/powheg/", additionalName = "_preUL05_28_02_22_ttbar", year ="2016", tokeep = keep,
				modelName = "preUL05_rhoRegModel_28_02_22", outFolder="preUL05_28_02_22/rhoReg_madgraph/")'''
def doTrainingAndEvaluation(inPathFolder, additionalName, year, tokeep = None, outFolder="outFolder_DEFAULTNAME/", modelName = "rhoRegModel_preUL04_moreInputs_Pt30_madgraph_cutSel"):
    
    inX_train, inX_test, outY_train, outY_test, weights_train, weights_test,lkrM_train, lkrM_test, krM_train, krM_test, scaler = loadData(
                                    inPathFolder = inPathFolder, year = year,
                                    additionalName = additionalName, testFraction = 0.4,
                                    overwrite = False, withBTag = True, pTEtaPhiMode=True,
                                    maxEvents = None)
                                    # maxEvents = 10000)
#Now data are loaded and splitted in training and testing with corresponding weights. Also the output of the classical approaches are saved
# Q? what are weights test
# Q? What's the scaler
    print ("Input events with \t",inX_train.shape[1], "features")
# Reduce to most useful features (to be kept)
    feature_names, inX_train, inX_test = helpers.getReducedFeatureNamesAndInputs(inX_train, inX_test, tokeep=tokeep)
    print (len(feature_names), " Reduced features labels")
 
    weightHisto = ROOT.TH1F("weightHisto","weightHisto",nbin, ar_bins)
    m_tt_arr = np.array([])
    print("Filling histo with m_tt")
    for m,weight in zip(outY_train, weights_train):
        weightHisto.Fill(m,abs(weight))

    weightHisto.Scale(1./weightHisto.Integral())
    maximumBin = weightHisto.GetMaximumBin()
    maximumBinContent = weightHisto.GetBinContent(maximumBin)

    canvas = ROOT.TCanvas()
    weightHisto.Draw("histe")
    weightHisto.SetTitle("Normalized and weighted m_{tt} distibutions. Training")
    weightHisto.SetYTitle('Normalized Counts')
    canvas.SaveAs("mNormalizedWeighted.pdf")

    for binX in range(1,weightHisto.GetNbinsX()+1):
        c = weightHisto.GetBinContent(binX)
        if c>0:
            weightHisto.SetBinContent(binX, maximumBinContent/c)
        else:
            weightHisto.SetBinContent(binX, 1.)
    weightHisto.SetBinContent(0, weightHisto.GetBinContent(1))
    weightHisto.SetBinContent(weightHisto.GetNbinsX()+1, weightHisto.GetBinContent(weightHisto.GetNbinsX()))

    canvas.Clear()
    weightHisto.Draw("histe")
    weightHisto.SetTitle("Weights distributions")
    canvas.SetLogy(1)
    canvas.SaveAs("weights.pdf")

# Weightsd in the zip were determined by teh mlb distributions(?). This ones are used for the training in order to give more importance to regions with few events
# now weighhisto has the weights to be used for the training

    weights_train_original = weights_train
    weights_train_=[]
    for w,m in zip(weights_train,outY_train):
        weightBin = weightHisto.FindBin(m)
        addW = weightHisto.GetBinContent(weightBin)	# weights for importance
        weights_train_.append(abs(w)*addW)           	# weights of the training are the product of the original weights and the weights used to give more importance to regions with few events
    weights_train = np.array(weights_train_)		# final weights_train is np array

    weights_train = 1./np.mean(weights_train)*weights_train	# normalized by the mean

# hyperparameters
    learningRate = 0.0001
    batchSize = 512
    dropout = 0.360
    nDense = 2
    nNodes = 512
    regRate = 7.9857e-4
    activation = 'selu'
    outputActivation = 'linear'

    print ("getting model")
    model = models.getMRegModelFlat(regRate = regRate, activation = activation, dropout = dropout, nDense = nDense,
                                      nNodes = nNodes, inputDim = inX_train.shape[1], outputActivation = outputActivation)
# Q? Question in models: why no batch normalization in second layer?

    optimizer = tf.keras.optimizers.Adam(lr = learningRate)

    print ("compiling model")
    model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanAbsolutePercentageError(), metrics=['mean_absolute_error','mean_squared_error'])
# Q? MeanAbsolutePercentageError was chosen as loss function because gave best results in terms of ?


    callbacks=[]
# three Keras callbacks that are used to monitor and control the training of a neural network model:
# Learning rate scheduler. It reduces the learning rate by a factor of 0.1 if the validation loss does not improve for 3 epochs. This helps the model to converge faster and avoid overshooting the optimal solution.
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', patience=3)
# This callback monitors the validation loss and stops training if the validation loss does not improve for 300 epochs
    earlyStop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=300, restore_best_weights=True)
# This callback saves the weights of the best performing model during training, based on the validation loss. The saved model is stored in the specified output folder with the name "model_best.h5"
    modelCheckpoint = tf.keras.callbacks.ModelCheckpoint(outFolder + '/model_best.h5', monitor='val_loss', save_best_only=True)

# Not using the reduce_lr Q? why?
    callbacks.append(earlyStop)
    callbacks.append(modelCheckpoint)

    print ("fitting model")
    fit = model.fit(
            inX_train,
            outY_train,
            sample_weight = weights_train,
            validation_split = 0.25,
            batch_size = batchSize,
            epochs = 3500,
            shuffle = False,
            callbacks = callbacks,
            verbose = 1)
# Real prediction based on training
    y_predicted = model.predict(inX_test)
    y_predicted_train = model.predict(inX_train)

    if not os.path.exists(outFolder+"/"+year):
        os.makedirs(outFolder+"/"+year)

    #  "mean_absolute_error"
    matplotlib.pyplot.figure(0)
    matplotlib.pyplot.plot(fit.history['mean_absolute_error'])
    matplotlib.pyplot.plot(fit.history['val_mean_absolute_error'])
    matplotlib.pyplot.title('model mean_absolute_error')
    matplotlib.pyplot.ylabel('mean_absolute_error')
    matplotlib.pyplot.xlabel('epoch')
    # matplotlib.pyplot.yscale('log')
    matplotlib.pyplot.ylim(ymax = min(fit.history['mean_absolute_error'])*1.4, ymin = min(fit.history['mean_absolute_error'])*0.9)
    matplotlib.pyplot.legend(['train', 'validation'], loc='upper right')
    matplotlib.pyplot.savefig(outFolder+"/"+year+"/mean_absolute_error.pdf")
    #  "mean_absolute_error"
    matplotlib.pyplot.figure(1)
    matplotlib.pyplot.plot(fit.history['mean_squared_error'])
    matplotlib.pyplot.plot(fit.history['val_mean_squared_error'])
    matplotlib.pyplot.title('model mean_squared_error')
    matplotlib.pyplot.ylabel('mean_squared_error')
    matplotlib.pyplot.xlabel('epoch')
    # matplotlib.pyplot.yscale('log')
    matplotlib.pyplot.ylim(ymax = min(fit.history['mean_squared_error'])*1.4, ymin = min(fit.history['mean_squared_error'])*0.9)
    matplotlib.pyplot.legend(['train', 'validation'], loc='upper right')
    matplotlib.pyplot.savefig(outFolder+"/"+year+"/mean_squared_error.pdf")
    # "Loss"
    matplotlib.pyplot.figure(2)
    matplotlib.pyplot.plot(fit.history['loss'])
    matplotlib.pyplot.plot(fit.history['val_loss'])
    matplotlib.pyplot.title('model loss')
    matplotlib.pyplot.ylabel('loss')
    matplotlib.pyplot.xlabel('epoch')
    # matplotlib.pyplot.yscale('log')
    matplotlib.pyplot.ylim(ymax = min(fit.history['loss'])*1.4, ymin = min(fit.history['loss'])*0.9)
    matplotlib.pyplot.legend(['train', 'validation'], loc='upper right')
    matplotlib.pyplot.savefig(outFolder+"/"+year+"/loss.pdf")

    corr = pearsonr(outY_test.reshape(outY_test.shape[0]), y_predicted.reshape(y_predicted.shape[0]))
    print("correlation:",corr[0])
# statistica distance: measures how one probab distr P is different from a second Q. In our case predicted output and expected one
    kl = compute_kl_divergence(y_predicted, outY_test, n_bins=100)
    print ("KL", kl)
# another way to say how similar are two distributions
    js = compute_js_divergence(y_predicted, outY_test, n_bins=100)
    print ("JS", js)
    print('Do Evaluation Plots')
    doEvaluationPlots(outY_train, y_predicted_train, weights_train_original, lkrM_train, krM_train, year = year, outFolder = outFolder+"/train/")
    doEvaluationPlots(outY_test, y_predicted, weights_test, lkrM_test, krM_test, year = year, outFolder = outFolder+"/test/")
# weights test are just the ones from mlb distributions (or the original ones)

    print('Before defining class \'rho\'')
    class_names=["rho"]
    print('After defining class \'rho\'')
    max_display = inX_test.shape[1]
# Commented till here
    for explainer, name  in [(shap.GradientExplainer(model,inX_test[:1000]),"GradientExplainer"),]:
        shap.initjs()
        print("... {0}: explainer.shap_values(X)".format(name))
        shap_values = explainer.shap_values(inX_test[:1000])
        print("... shap.summary_plot")
        matplotlib.pyplot.clf()
        shap.summary_plot(shap_values, inX_test[:1000], plot_type="bar",
            feature_names=feature_names,
            class_names=class_names,
            max_display=max_display,plot_size=[15.0,0.4*max_display+1.5],show=False)
        matplotlib.pyplot.savefig(outFolder+"/"+year+"/"+"/shap_summary_{0}.pdf".format(name))


    saveModel = True
    if saveModel:
        model.save(outFolder+modelName+".h5")
        tf.keras.backend.clear_session()
        tf.keras.backend.set_learning_phase(0)
        model = tf.keras.models.load_model(outFolder+modelName+".h5")
        print('inputs: ', [input.op.name for input in model.inputs])
        print('outputs: ', [output.op.name for output in model.outputs])
        frozen_graph = helpers.freeze_session(tf.compat.v1.keras.backend.get_session(), output_names=[out.op.name for out in model.outputs])
        tf.compat.v1.train.write_graph(frozen_graph, outFolder+'/', modelName+'.pbtxt', as_text=True)
        tf.compat.v1.train.write_graph(frozen_graph, outFolder+'/', modelName+'.pb', as_text=False)
        print ("Saved model to",outFolder+"/"+year+'/'+modelName+'.pbtxt/.pb/.h5')




def justEvaluate(inPathFolder, additionalName, year, tokeep = None, modelDir = "rhoOutput/", modelName = "rhoRegModel_DEFAULTNAME", outFolder="outFolder_DEFAULTNAME/"):
    inX_train, inX_test, outY_train, outY_test, weights_train, weights_test,lkrM_train, lkrM_test, krM_train, krM_test, scaler = loadData(
                                    inPathFolder = inPathFolder, year = year,
                                    additionalName = additionalName, testFraction = 0.99,
                                    overwrite = False, withBTag = True, pTEtaPhiMode=True,
                                    maxEvents = None)
                                    # maxEvents = 50000)

    feature_names = [i for i in feature_names_all]
    for i in range(len(feature_names_all)):
        feature_names[i]=feature_names_all[i].replace(feature_names_all[i],str(i)+"_"+feature_names_all[i])

    feature_names_original = feature_names_all.copy()
    feature_names_new = []

    to_remove = []
    if tokeep == None:
        to_keep = [i for i in range(len(feature_names_all))] #all
    else:
        to_keep = tokeep

    for i in range(len(feature_names_original)):
        if i in to_keep:
            feature_names_new.append(feature_names_original[i])
        else:
            to_remove.append(i)

    feature_names = feature_names_new

    inX_test=np.delete(inX_test,  to_remove,1)

    model = tf.keras.models.load_model(modelDir+"/"+modelName+".h5")
    y_predicted = model.predict(inX_test)

    doEvaluationPlots(outY_test, y_predicted, weights_test, lkrM_test, krM_test, year = year, outFolder = outFolder)

def main():
    keep = None 
	#[41,42,35,60,84,64,30,76,39,88,103]
    print("********************************************\n*					   *\n*        Main function started             *\n*					   *\n********************************************")
    # doBaysianOptim(inPathFolder = "rhoInput/madgraph_ghost/", additionalName = "_preUL04_11_02_21_ghost", year ="FR2",
    #                 tokeep = keep, outFolder="preUL04_11_02_21_ghost/rho_madgraph_optim_ghost_perIterationSave/")
    # justEvaluate(inPathFolder="rhoInput/powheg/", additionalName="_preUL04_11_02_21", year="FR2", tokeep = [41,42,35,60,84,64,30,76,39,88],
    #             modelDir = "preUL04_11_02_21/rhoReg_madgraph/", modelName = "rhoRegModel_preUL04_11_02_21", outFolder="preUL04_11_02_21/rhoReg_powheg_newHybrid/")
    doEvaluate = False
    if (doEvaluate):
    # doBaysianOptim(inPathFolder = "rhoInput/madgraph_ghost/", additionalName = "_preUL04_11_02_21_ghost", year ="FR2",
    #                 tokeep = keep, outFolder="preUL04_11_02_21_ghost/rho_madgraph_optim_ghost_perIterationSave/")
    
        print("Calling justEvaluate ...")
        justEvaluate(inPathFolder="/nfs/dust/cms/user/celottog/TopRhoNetwork/rhoInput/powheg/", additionalName="_preUL05_28_02_22_ttbar", year="2016", tokeep = keep,
        	         modelDir = "preUL05_28_02_22/rhoReg_madgraph/", modelName = "preUL05_rhoRegModel_28_02_22", outFolder="preUL05_28_02_22_justevaluated/rhoReg_madgraph/")
    else:
        print("Calling doTrainingAndEvaluation ...")
        doTrainingAndEvaluation(inPathFolder = "/nfs/dust/cms/user/celottog/TopRhoNetwork/rhoInput/powheg/", additionalName = "_preUL05_28_02_22_ttbar", year ="2016", tokeep = keep,
				modelName = "preUL05_rhoRegModel_28_02_22", outFolder="preUL05_28_02_22/rhoReg_madgraph/")

if __name__ == "__main__":
    main()
