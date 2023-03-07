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
matplotlib.use('Agg')
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
#bins = np.arange(340, 1500, 50)


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
        lkrM = lkrM[:maxEvents]
        krM = krM[:maxEvents]

    #outY = outY.reshape((outY.shape[0], 1))
    print(inX.shape, outY.shape, weights.shape, lkrM.shape, krM.shape)
    # transform y to uniform
    print("*** Defining scaler")
    scaler = preprocessing.StandardScaler().fit(outY)

    inX_train, inX_test, outY_train, outY_test, weights_train, weights_test, lkrM_train, lkrM_test, krM_train, krM_test = train_test_split(inX, outY, weights, lkrM, krM, test_size = testFraction)

    print ("\t data splitted succesfully")
    print ("loadData end")
    print("Number training events :", inX_train.shape[0], len(inX_train))
    print("Number of features     :", inX_train.shape[1])
    print("loadData ended returning inX_train and so on...")

    return inX_train, inX_test, outY_train, outY_test, weights_train, weights_test,lkrM_train, lkrM_test, krM_train, krM_test, scaler


def doEvaluationPlots(yTest, yPredicted, weightTest, lkrM, krM, year = "", outFolder = "MOutput/"):

    outDir = outFolder+"/"+year+"/"
    if not os.path.exists(outDir):
        os.makedirs(outDir)

    f = matplotlib.pyplot.figure()
    values, bins, patches = matplotlib.pyplot.hist(yTest, bins=100, range=(340,3000), label="true", alpha=0.5)
    values2, bins2, patches2 = matplotlib.pyplot.hist(yPredicted, bins=100, range=(340,3000), label="reco", alpha=0.5)
    matplotlib.pyplot.legend(loc = "best")
    matplotlib.pyplot.ylabel('Events')
    matplotlib.pyplot.xlabel('m(tt)')
    f.savefig(outDir+"eval_m(t).pdf")

    ROOT.gStyle.SetPalette(ROOT.kThermometer)

    style.style1d()
    s = style.style1d()
    histoShape = ROOT.TH1F( "reco shape", ";m(t#overline{t})", 50, 340, 1500)
    histoShape2 = ROOT.TH1F( "true shape2", ";m(t#overline{t})", 50, 340, 1500)
    histoShape3 = ROOT.TH1F( "lkr shape2", ";m(t#overline{t})", 50, 340, 1500)
    histoShape4 = ROOT.TH1F( "kr shape2", ";m(t#overline{t})", 50, 340, 1500)
    histoShape5 = ROOT.TH1F( "true shape no kinReco", ";m(t#overline{t})", 50, 340, 1500)
    histoShape6 = ROOT.TH1F( "reg shape no kinReco", ";m(t#overline{t})", 50, 340, 1500)
    histoShape8 = ROOT.TH1F( "new hybrid shape", ";m(t#overline{t})", 50, 340, 1500)
    histoShape9 = ROOT.TH1F( "new hybrid2 shape", ";m(t#overline{t})", 50, 340, 1500)

    print(yTest, "\n\n\n***\n\n\n",yPredicted, "\n\n\n***\n\n\n", weightTest, "\n\n\n***\n\n\n",lkrM, "\n\n\n***\n\n\n", krM)
    print(len(yTest), "   ***   ",len(yPredicted), "   ***    ", len(weightTest), "   ***    ",len(lkrM), "   ***   ", len(krM))
    for Mtrue, Mpred, weight, lkrMass, krMass in zip(yTest, yPredicted, weightTest, lkrM, krM):
        #print("prova")
        if lkrMass>0.:
            histoShape3.Fill(lkrMass, weight)
        if krMass >0.:
            histoShape4.Fill(krMass, weight)
        else:
            histoShape6.Fill(Mpred, weight)
            histoShape5.Fill(Mtrue, weight)
        histoShape.Fill(Mpred, weight)
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
        histoShape2.Fill(Mtrue, weight)
    print('Triallllllllllllll\n\n\n\nTriallllll')
    histoShape7 = histoShape6.Clone()
    histoShape7.Add(histoShape4)

    max_ = 1.4*max(histoShape.GetMaximum(),histoShape2.GetMaximum(),histoShape3.GetMaximum(),histoShape4.GetMaximum(),histoShape5.GetMaximum(),histoShape6.GetMaximum(),histoShape7.GetMaximum())
    histoShape.SetMaximum(max_)
    histoShape2.SetMaximum(max_)
    histoShape3.SetMaximum(max_)
    histoShape4.SetMaximum(max_)
    histoShape5.SetMaximum(max_)
    histoShape6.SetMaximum(max_)
    histoShape7.SetMaximum(max_)
    histoShape8.SetMaximum(max_)
    histoShape9.SetMaximum(max_)

    histoShape.SetLineWidth(2)
    histoShape2.SetLineWidth(2)
    histoShape3.SetLineWidth(2)
    histoShape4.SetLineWidth(2)
    histoShape5.SetLineWidth(2)
    histoShape6.SetLineWidth(2)
    histoShape7.SetLineWidth(2)
    histoShape8.SetLineWidth(2)
    histoShape9.SetLineWidth(2)

    c0=ROOT.TCanvas("c0","c0",800,800)
    histoShape.SetLineColor(ROOT.kRed)
    histoShape.Draw("h0")
    histoShape2.Draw("h0same")
    histoShape3.SetLineColor(ROOT.kOrange+1)
    histoShape3.Draw("h0same")
    histoShape4.SetLineColor(ROOT.kGreen+1)
    histoShape4.Draw("h0same")
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
    l = ROOT.TLegend(0.2,0.7,0.4,0.9)
    l.AddEntry(histoShape, "DNN")
    l.AddEntry(histoShape2, "truth")
    l.AddEntry(histoShape3, "LKR")
    l.AddEntry(histoShape4, "KR")
    # l.AddEntry(histoShape5, "DNN(noKR)")
    l.AddEntry(histoShape6, "truth(noKR)")
    l.AddEntry(histoShape7, "KR + reg (noKR)")
    l.AddEntry(histoShape8, "newHybrid")
    l.AddEntry(histoShape9, "newHybrid2")
    l.Draw()
    c0.SaveAs(outDir+"shapes.pdf")

    style.style2d()
    s = style.style2d()
    histo = ROOT.TH2F( "reg rho diff", ";rho reco;m(t#overline{t}) true - m(t#overline{t}) reco",100, 340, 1500, 200, -1500, 1500 )
    histo.SetDirectory(0)
    histo_kr = ROOT.TH2F( "kr rho diff", ";rho reco;m(t#overline{t}) true - m(t#overline{t}) reco",100, 340, 1500, 200, -1500, 1500 )
    histo_kr.SetDirectory(0)
    histo_krReg = ROOT.TH2F( "kr+Reg rho diff", ";rho reco;m(t#overline{t}) true - m(t#overline{t}) reco",100, 340, 1500, 200, -1500, 1500 )
    histo_krReg.SetDirectory(0)
    histo_lkr = ROOT.TH2F( "lkr rho diff", ";rho reco;m(t#overline{t}) true - m(t#overline{t}) reco",100, 340, 1500, 200, -1500, 1500 )
    histo_lkr.SetDirectory(0)
    histo_newHybrid = ROOT.TH2F( "newHybrid rho diff", ";rho reco;m(t#overline{t}) true - m(t#overline{t}) reco",100, 340, 1500, 200, -1500, 1500 )
    histo_newHybrid.SetDirectory(0)
    histo_newHybrid2 = ROOT.TH2F( "newHybrid2 rho diff", ";rho reco;m(t#overline{t}) true - m(t#overline{t}) reco",100, 340, 1500, 200, -1500, 1500 )
    histo_newHybrid2.SetDirectory(0)
    histTrue = ROOT.TH1F( "true contents", ";rho", 20, 0, 1)
    histTrue.SetDirectory(0)

    histoRecoGen = ROOT.TH2F( "reg rho2d", ";m(t#overline{t}) reco;m(t#overline{t}) true",100, 340, 1500, 100, 340, 1500 )
    histoRecoGen.SetDirectory(0)
    histoRecoGen_kr = ROOT.TH2F( "kr rho2d", ";m(t#overline{t}) reco;m(t#overline{t}) true",100, 340, 1500, 100, 340, 1500 )
    histoRecoGen_kr.SetDirectory(0)
    histoRecoGen_krReg = ROOT.TH2F( "kr+reg rho2d", ";m(t#overline{t}) reco;m(t#overline{t}) true",100, 340, 1500, 100, 340, 1500 )
    histoRecoGen_krReg.SetDirectory(0)
    histoRecoGen_lkr = ROOT.TH2F( "lkr rho2d", ";m(t#overline{t}) reco;m(t#overline{t}) true",100, 340, 1500, 100, 340, 1500 )
    histoRecoGen_lkr.SetDirectory(0)
    histoRecoGen_newHybrid = ROOT.TH2F( "newHybrid rho2d", ";m(t#overline{t}) reco;m(t#overline{t}) true",100, 340, 1500, 100, 340, 1500 )
    histoRecoGen_newHybrid.SetDirectory(0)
    histoRecoGen_newHybrid2 = ROOT.TH2F( "newHybrid2 rho2d", ";m(t#overline{t}) reco;m(t#overline{t}) true",100, 340, 1500, 100, 340, 1500 )
    histoRecoGen_newHybrid2.SetDirectory(0)
    histoRecoGen2 = ROOT.TH2F( "reg resp", ";reco bin;true bin", 20 ,340, 1500, 20 ,340, 1500)
    histoRecoGen2.SetDirectory(0)
    histoRecoGen2_kr = ROOT.TH2F( "kr resp", ";reco bin;true bin", 20 ,340, 1500, 20 ,340, 1500)
    histoRecoGen2_kr.SetDirectory(0)
    histoRecoGen2_krReg = ROOT.TH2F( "kr+reg resp", ";reco bin;true bin", 20 ,340, 1500, 20 ,340, 1500)
    histoRecoGen2_krReg.SetDirectory(0)
    histoRecoGen2_lkr = ROOT.TH2F( "lkr resp", ";reco bin;true bin", 20 ,340, 1500, 20 ,340, 1500)
    histoRecoGen2_lkr.SetDirectory(0)
    histoRecoGen2_newHybrid = ROOT.TH2F( "newHybrid resp", ";reco bin;true bin", 20 ,340, 1500, 20 ,340, 1500)
    histoRecoGen2_newHybrid.SetDirectory(0)
    histoRecoGen2_newHybrid2 = ROOT.TH2F( "newHybrid2 resp", ";reco bin;true bin", 20 ,340, 1500, 20 ,340, 1500)
    histoRecoGen2_newHybrid2.SetDirectory(0)

    binning = range(0, 1400, 50)
    ar_bins = array("d", binning)
    nbin = len(binning)-1

    histo2dbinned = ROOT.TH2F( "resp class", ";rho true;#rho reco", nbin, ar_bins, nbin, ar_bins )
    histo2dbinned_newHybrid = ROOT.TH2F( "resp class newHybrid", ";rho true;#rho reco", nbin, ar_bins, nbin, ar_bins )
    histTrueBinned = ROOT.TH1F( "true contents binned", ";rho", nbin, ar_bins)
    histTrueBinned.SetDirectory(0)
    print("Before entering in the loopo \n\n\n\n*****************************************\n\n\n\n ")
    print(yTest.mean(), yTest.max(), yTest.min(), yPredicted.mean(), yPredicted.min(), yPredicted.max())
    for Mtrue, Mpred, weight, lkrMass, krMass in zip(yTest, yPredicted, weightTest, lkrM, krM):

        diff = Mtrue-Mpred
        histo.Fill(Mtrue, diff, weight)
        histoRecoGen.Fill(Mpred, Mtrue, weight)
        histoRecoGen2.Fill(Mpred, Mtrue, weight)
        # histo2dbinned.Fill(Mpred, Mtrue, weight)
        histo2dbinned.Fill(Mtrue, Mpred, weight)
        histTrue.Fill(Mtrue, weight)
        if krMass>0.:
            diff_kr = Mtrue -krMass
            histo_kr.Fill(Mtrue, diff_kr, weight)
            histoRecoGen_kr.Fill(krMass, Mtrue, weight)
            histoRecoGen2_kr.Fill(krMass, Mtrue, weight)
            histo_krReg.Fill(Mtrue, diff_kr, weight)
            histoRecoGen_krReg.Fill(krMass, Mtrue, weight)
            histoRecoGen2_krReg.Fill(krMass, Mtrue, weight)
        else:
            histo_krReg.Fill(Mtrue, diff, weight)
            histoRecoGen_krReg.Fill(Mpred, Mtrue, weight)
            histoRecoGen2_krReg.Fill(Mpred, Mtrue, weight)
        if lkrMass>0.:
            diff_lkr = Mtrue-lkrMass
            histo_lkr.Fill(Mtrue, diff_lkr, weight)
            histoRecoGen_lkr.Fill(lkrMass, Mtrue, weight)
            histoRecoGen2_lkr.Fill(lkrMass, Mtrue, weight)
        mAverage = []
        mAverage.append(Mpred)
        if lkrMass>0.:
            mAverage.append(lkrMass)
        if krMass>0.:
            mAverage.append(krMass)
        mAv = np.mean(mAverage)
        mAverage = np.array(mAverage, dtype=float)
        histo2dbinned_newHybrid.Fill(Mtrue, mAv, weight)
        mAv2 = gmean(mAverage)
        diff_av = Mtrue-mAv
        diff_av2 = Mtrue-mAv2
        histo_newHybrid.Fill(Mtrue, diff_av, weight)
        histoRecoGen_newHybrid.Fill(mAv, Mtrue, weight)
        histoRecoGen2_newHybrid.Fill(mAv, Mtrue, weight)
        histo_newHybrid2.Fill(Mtrue, diff_av2, weight)
        histoRecoGen_newHybrid2.Fill(mAv2, Mtrue, weight)
        histoRecoGen2_newHybrid2.Fill(mAv2, Mtrue, weight)
    print("******************\n\n\n**********************\n\n\nEntries: \t",histoRecoGen2.GetEntries(),"\n\n\n***********************\n\n\n")
    hResp = histoRecoGen2.Clone()
    hResp_kr = histoRecoGen2_kr.Clone()
    hResp_lkr = histoRecoGen2_lkr.Clone()
    hResp_krReg = histoRecoGen2_krReg.Clone()
    hResp_newHybrid = histoRecoGen2_newHybrid.Clone()
    hResp_newHybrid2 = histoRecoGen2_newHybrid2.Clone()

    c=ROOT.TCanvas("c1","c1",800,800)

    histo.GetXaxis().SetRangeUser(350,1500)
    histo.GetYaxis().SetRangeUser(-1200,1200)
    histo.SetStats(0)
    histo.Draw("colz")
    c.SaveAs(outDir+"reg_GenRecoDiff2d.pdf")
    c.Clear()
    histo_kr.GetXaxis().SetRangeUser(350,1500)
    histo_kr.GetYaxis().SetRangeUser(-1200,1200)
    histo_kr.SetStats(0)
    histo_kr.Draw("colz")
    c.SaveAs(outDir+"kr_GenRecoDiff2d.pdf")
    c.Clear()
    histo_krReg.GetXaxis().SetRangeUser(350,1500)
    histo_krReg.GetYaxis().SetRangeUser(-1200,1200)
    histo_krReg.SetStats(0)
    histo_krReg.Draw("colz")
    c.SaveAs(outDir+"krReg_GenRecoDiff2d.pdf")
    c.Clear()
    histo_lkr.GetXaxis().SetRangeUser(350,1500)
    histo_lkr.GetYaxis().SetRangeUser(-1200,1200)
    histo_lkr.SetStats(0)
    histo_lkr.Draw("colz")
    c.SaveAs(outDir+"lkr_GenRecoDiff2d.pdf")
    c.Clear()
    histo_newHybrid.GetXaxis().SetRangeUser(350,1500)
    histo_newHybrid.GetYaxis().SetRangeUser(-1200,1200)
    histo_newHybrid.SetStats(0)
    histo_newHybrid.Draw("colz")
    c.SaveAs(outDir+"newHybrid_GenRecoDiff2d.pdf")
    c.Clear()
    histo_newHybrid2.GetXaxis().SetRangeUser(350,1500)
    histo_newHybrid2.GetYaxis().SetRangeUser(-1200,1200)
    histo_newHybrid2.SetStats(0)
    histo_newHybrid2.Draw("colz")
    c.SaveAs(outDir+"newHybrid2_GenRecoDiff2d.pdf")
    c.Clear()

    histoRecoGen.Draw("colz")
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
    histoRecoGen_newHybrid.Draw("colz")
    corrLatex_newHybrid = ROOT.TLatex()
    corrLatex_newHybrid.SetTextSize(0.65 * corrLatex_newHybrid.GetTextSize())
    corrLatex_newHybrid.DrawLatexNDC(0.65, 0.85, str(np.round(histoRecoGen_newHybrid.GetCorrelationFactor(),3)))
    c.SaveAs(outDir+"newHybrid_GenReco2d.pdf")
    c.Clear()
    histoRecoGen_newHybrid2.Draw("colz")
    corrLatex_newHybrid2 = ROOT.TLatex()
    corrLatex_newHybrid2.SetTextSize(0.65 * corrLatex_newHybrid2.GetTextSize())
    corrLatex_newHybrid2.DrawLatexNDC(0.65, 0.85, str(np.round(histoRecoGen_newHybrid2.GetCorrelationFactor(),3)))
    c.SaveAs(outDir+"newHybrid2_GenReco2d.pdf")
    c.Clear()

    histo2dbinned.SetStats(0)
    hRespBinned = histo2dbinned.Clone()
    histo2dbinned.Scale(1./(histo2dbinned.Integral()))
    print ("binned correlation", histo2dbinned.GetCorrelationFactor())

    # from array import array
    nbinsx = histo2dbinned.GetNbinsX()
    nbinsy = histo2dbinned.GetNbinsY()

    ma = np.empty((nbinsx,nbinsy))
    ma2 = np.empty((nbinsx,nbinsy))
    for iBinX in range(1,nbinsx+1):
        for iBinY in range(1,nbinsy+1):
            ma[iBinX-1,iBinY-1]=histo2dbinned.GetBinContent(iBinX,iBinY)
            ma2[iBinY-1,iBinX-1]=histo2dbinned.GetBinContent(iBinX,iBinY)
    print ("Condition number = ",np.linalg.cond(ma))

    histo2dbinned.Scale(100.)


    ROOT.gStyle.SetPaintTextFormat("2.2f")
    histo2dbinned.SetMarkerSize(0.7)
    histo2dbinned.GetZaxis().SetTitle("Transition Probability [%]")
    # style.setPalette("bird")
    ROOT.gStyle.SetPalette(ROOT.kThermometer)
    # ROOT.gStyle.SetPaintTextFormat("1.2f");
    histo2dbinned.Draw("colz text")
    c.SaveAs(outDir+"reg_response.pdf")

    c.Clear()

    histo2dbinned_newHybrid.SetStats(0)
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
    ROOT.gStyle.SetPalette(ROOT.kThermometer)
    # ROOT.gStyle.SetPaintTextFormat("1.2f");
    histo2dbinned_newHybrid.Draw("colz text")
    c.SaveAs(outDir+"newHybrid_response.pdf")

    c.Clear()

    histoRecoGen2.Scale(1./histoRecoGen2.Integral())
    # histoRecoGen2.Scale(100.)
    ROOT.gStyle.SetPaintTextFormat("1.2f");
    # ROOT.gStyle.SetPaintTextFormat("1.0f");
    histoRecoGen2.SetStats(0)
    histoRecoGen2.SetMarkerSize(1)
    histoRecoGen2.Draw("colz text")
    c.SaveAs(outDir+"resp.pdf")
    c.Clear()
    histoRecoGen2_kr.Scale(1./histoRecoGen2_kr.Integral())
    # histoRecoGen2_kr.Scale(100.)
    ROOT.gStyle.SetPaintTextFormat("1.2f");
    # ROOT.gStyle.SetPaintTextFormat("1.0f");
    histoRecoGen2_kr.SetStats(0)
    histoRecoGen2_kr.SetMarkerSize(1)
    histoRecoGen2_kr.Draw("colz text")
    c.SaveAs(outDir+"kr_resp.pdf")
    c.Clear()
    histoRecoGen2_krReg.Scale(1./histoRecoGen2_krReg.Integral())
    # histoRecoGen2_krReg.Scale(100.)
    ROOT.gStyle.SetPaintTextFormat("1.2f");
    # ROOT.gStyle.SetPaintTextFormat("1.0f");
    histoRecoGen2_krReg.SetStats(0)
    histoRecoGen2_krReg.SetMarkerSize(1)
    histoRecoGen2_krReg.Draw("colz text")
    c.SaveAs(outDir+"krReg_resp.pdf")
    c.Clear()
    histoRecoGen2_lkr.Scale(1./histoRecoGen2_lkr.Integral())
    # histoRecoGen2_lkr.Scale(100.)
    ROOT.gStyle.SetPaintTextFormat("1.2f");
    # ROOT.gStyle.SetPaintTextFormat("1.0f");
    histoRecoGen2_lkr.SetStats(0)
    histoRecoGen2_lkr.SetMarkerSize(1)
    histoRecoGen2_lkr.Draw("colz text")
    c.SaveAs(outDir+"lkr_resp.pdf")
    c.Clear()
    histoRecoGen2_newHybrid.Scale(1./histoRecoGen2_newHybrid.Integral())
    # histoRecoGen2_lkr.Scale(100.)
    ROOT.gStyle.SetPaintTextFormat("1.2f");
    # ROOT.gStyle.SetPaintTextFormat("1.0f");
    histoRecoGen2_newHybrid.SetStats(0)
    histoRecoGen2_newHybrid.SetMarkerSize(1)
    histoRecoGen2_newHybrid.Draw("colz text")
    c.SaveAs(outDir+"newHybrid_resp.pdf")
    c.Clear()

    rms_reg, mean_reg, respRMS_reg = doRMSandMean(histo, "reg", outDir)
    rms_kr, mean_kr, respRMS_kr = doRMSandMean(histo_kr, "kr", outDir)
    rms_krReg, mean_krReg, respRMS_krReg = doRMSandMean(histo_krReg, "krReg", outDir)
    rms_lkr, mean_lkr, respRMS_lkr = doRMSandMean(histo_lkr, "lkr", outDir)
    rms_newHybrid, mean_newHybrid, respRMS_newHybrid = doRMSandMean(histo_newHybrid, "newHybrid", outDir)
    rms_newHybrid2, mean_newHybrid2, respRMS_newHybrid2 = doRMSandMean(histo_newHybrid2, "newHybrid2", outDir)

    purity_kr, stability_kr, eff_kr = doPSE(hResp_kr, histTrue, "kr", outDir)

    style.style1d()
    s = style.style1d()

    c.SetLogx(False)
    c.SetLogy(False)
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
    histoShape.Write("shape_reg")
    histoShape2.Write("shape_true")
    histoShape3.Write("shape_lkr")
    histoShape4.Write("shape_kr")
    histoShape5.Write("shape_trueNoKR")
    histoShape6.Write("shape_regNoKR")
    histoShape7.Write("shape_KRPlusRegNoKR")
    histoShape8.Write("shape_newHybrid")
    histoShape9.Write("shape_newHybrid2")
    histo.Write("true_vs_reg")
    histo_kr.Write("true_vs_kr")
    histo_lkr.Write("true_vs_lkr")
    histo_krReg.Write("true_vs_krReg")
    histo_newHybrid.Write("true_vs_newHybrid")
    histo_newHybrid2.Write("true_vs_newHybrid2")
    histTrue.Write("true")
    histoRecoGen.Write("recoGen_reg")
    histoRecoGen2.Write("recoGen2_reg")
    histoRecoGen_kr.Write("recoGen_kr")
    histoRecoGen_lkr.Write("recoGen_lkr")
    histoRecoGen_krReg.Write("recoGen_krReg")
    histoRecoGen_newHybrid.Write("recoGen_newHybrid")
    histoRecoGen2_krReg.Write("recoGen2_krReg")
    histoRecoGen2_lkr.Write("recoGen2_lkr")
    histoRecoGen2_kr.Write("recoGen2_kr")
    histoRecoGen2_newHybrid.Write("recoGen2_newHybrid")
    histoRecoGen2_newHybrid2.Write("recoGen2_newHybrid2")
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
    rms_newHybrid.Write("rms_newHybrid")
    respRMS_newHybrid.Write("respRMS_newHybrid")
    mean_newHybrid.Write("mean_newHybrid")
    rms_newHybrid2.Write("rms_newHybrid2")
    respRMS_newHybrid2.Write("respRMS_newHybrid2")
    mean_newHybrid2.Write("mean_newHybrid2")
    purity_kr.Write("purity_kr")
    stability_kr.Write("stability_kr")
    eff_kr.Write("efficiency_kr")
    outRootFile.Close()


def doFitForBaysian(inX_train, outY_train, weights_train, inX_test, outY_test, weights_test, lkrM_test, krM_test, learningRate = 0.001, batchSize = 4096, dropout = 0.05, nDense = 3, nNodes = 500, indexActivation = 1, regRate = 1e-7):
    nDense = int(nDense)
    nNodes = int(nNodes)
    batchSize = int(batchSize)
    # indexActivation = int(indexActivation)
    # activations = ["sigmoid", "relu", "softmax", "selu", "softplus"]
    # activation = activations[indexActivation]
    activation = "selu"
    regRate = 7.9857e-5

    model = models.getRhoRegModelFlat(regRate = regRate, activation = activation, dropout = dropout, nDense = nDense,
                                      nNodes = nNodes, inputDim = inX_train.shape[1], outputActivation = 'linear', printSummary=False)
    optimizer = tf.keras.optimizers.Adam(lr = learningRate)

    print ("batch size",batchSize,"nDense",nDense,"nNodes",nNodes,"dropout",dropout,"regRate",regRate,"learningRate",learningRate)

    model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanAbsolutePercentageError(),metrics=['mean_absolute_error','mean_squared_error'])

    print ("N parameter: ",model.count_params())
    if (model.count_params()> 1500000):
        return 0.

    callbacks=[]
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', patience=3)
    earlyStop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=40, restore_best_weights=True)
    callbacks.append(earlyStop)

    fit = model.fit(
            inX_train,
            outY_train,
            sample_weight = weights_train,
            validation_split = 0.25,
            batch_size = batchSize,
            epochs = 300,
            shuffle = False,
            callbacks = callbacks,
            verbose = 0)

    score = model.evaluate(
                        inX_test,
                        outY_test,
                        sample_weight = weights_test,
                        batch_size = batchSize,
                        verbose = 0)
    y_predicted = model.predict(inX_test)
    y_predicted = y_predicted.reshape(y_predicted.shape[0])
    outY_test = outY_test.reshape(outY_test.shape[0])
    print('Test loss:', score[0])
    print('Test MSE:', score[1])

    corr = pearsonr(outY_test, y_predicted)
    print("corr",corr[0])

    kl = compute_kl_divergence(y_predicted, outY_test, n_bins=200)
    print ("KL", kl)
    js = compute_js_divergence(y_predicted, outY_test, n_bins=200)
    print ("JS", js)

    binning = [0.0, 0.45, 0.75, 1.0]
    ar_bins = array("d", binning)
    histo2dbinned = ROOT.TH2F( "resp class", ";rho true;#rho reco", 3, ar_bins, 3, ar_bins )
    for Mtrue, Mpred, weight in zip(outY_test, y_predicted, weights_test):
        histo2dbinned.Fill(Mtrue, Mpred, weight)
    histo2dbinned.Scale(1./(histo2dbinned.Integral()))
    print ("binned correlation", histo2dbinned.GetCorrelationFactor())
    nbinsx = histo2dbinned.GetNbinsX()
    nbinsy = histo2dbinned.GetNbinsY()
    ma = np.empty((nbinsx,nbinsy))
    ma2 = np.empty((nbinsx,nbinsy))
    for iBinX in range(1,nbinsx+1):
        for iBinY in range(1,nbinsy+1):
            ma[iBinX-1,iBinY-1]=histo2dbinned.GetBinContent(iBinX,iBinY)
            ma2[iBinY-1,iBinX-1]=histo2dbinned.GetBinContent(iBinX,iBinY)
    print ("Condition number = ",np.linalg.cond(ma))

    global bayesian_base_outFolder
    global bayesian_current_iteration
    outFolder = bayesian_base_outFolder+"_"+str(bayesian_current_iteration)+"/"
    modelName = "optimModel_"+str(bayesian_current_iteration)

    doEvaluationPlots(outY_test, y_predicted, weights_test, lkrM_test, krM_test, year = "FR2", outFolder = outFolder+"/test/")

    saveModel=True
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
        print ("Saved model to",outFolder+"/"+"FR2"+'/'+modelName+'.pbtxt/.pb/.h5')

    bayesian_current_iteration = bayesian_current_iteration + 1

    if np.isnan(corr[0]):
        return 0.
    if np.isnan(kl):
        return 0.
    if np.isnan(js):
        return 0.
    else:
        return (1.-js)


def doBaysianOptim(inPathFolder, additionalName, year, tokeep = None, outFolder = "BaysianOptim/"):
    inX_train, inX_test, outY_train, outY_test, weights_train, weights_test,lkrM_train, lkrM_test, krM_train, krM_test, scaler = loadData(
                                    inPathFolder = inPathFolder, year = year, additionalName = additionalName, testFraction = 0.4, overwrite = False, withBTag = True, pTEtaPhiMode=True, maxEvents = None)
    global bayesian_base_outFolder
    bayesian_base_outFolder = outFolder

    feature_names, inX_train, inX_test = helpers.getReducedFeatureNamesAndInputs(inX_train, inX_test, tokeep=tokeep)

    weightHisto = ROOT.TH1F("weightHisto","weightHisto",100,340,1500)

    for rho,weight in zip(outY_train, weights_train):
        weightHisto.Fill(rho, abs(weight))

    weightHisto.Scale(1./weightHisto.Integral())
    maximumBin = weightHisto.GetMaximumBin()
    maximumBinContent = weightHisto.GetBinContent(maximumBin)

    for binX in range(1,weightHisto.GetNbinsX()+1):
        c = weightHisto.GetBinContent(binX)
        if c>0:
            weightHisto.SetBinContent(binX, maximumBinContent/c)
        else:
            weightHisto.SetBinContent(binX, 1.)
    weightHisto.SetBinContent(0, weightHisto.GetBinContent(1))
    weightHisto.SetBinContent(weightHisto.GetNbinsX()+1, weightHisto.GetBinContent(weightHisto.GetNbinsX()))

    weights_train_original = weights_train
    weights_train_=[]
    for w,rho in zip(weights_train,outY_train):
        weightBin = weightHisto.FindBin(rho)
        addW = weightHisto.GetBinContent(weightBin)
        weights_train_.append(abs(w)*addW)
    weights_train = np.array(weights_train_)

    weights_train = 1./np.mean(weights_train)*weights_train

    fit_with_partial = partial(doFitForBaysian, inX_train, outY_train, weights_train, inX_test, outY_test, weights_test, lkrM_test, krM_test)

    # Bounded region of parameter space
    pbounds = {'batchSize': (1000, 30000),
                # 'regRate': (1e-7, 1e-4),
                'learningRate': (1e-4, 1e-2),
                'dropout': (1e-4, 5e-1),
                'nDense': (2, 10),
                # 'indexActivation': (0, 4),
                'nNodes': (50, 3000)
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
    optimizer.maximize(init_points=30, n_iter=30, kappa=8, alpha=1e-3, n_restarts_optimizer=5)
    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))
    print(optimizer.max)


def doTrainingAndEvaluation(inPathFolder, additionalName, year, tokeep = None, outFolder="outFolder_DEFAULTNAME/", modelName = "rhoRegModel_preUL04_moreInputs_Pt30_madgraph_cutSel"):
    
    inX_train, inX_test, outY_train, outY_test, weights_train, weights_test,lkrM_train, lkrM_test, krM_train, krM_test, scaler = loadData(
                                    inPathFolder = inPathFolder, year = year,
                                    additionalName = additionalName, testFraction = 0.4,
                                    overwrite = False, withBTag = True, pTEtaPhiMode=True,
                                    maxEvents = None)
                                    # maxEvents = 10000)

    print ("Each events has \t",inX_train.shape[1], "features")
    feature_names, inX_train, inX_test = helpers.getReducedFeatureNamesAndInputs(inX_train, inX_test, tokeep=tokeep)

    print (len(feature_names), " Reduced features labels")
 
    weightHisto = ROOT.TH1F("weightHisto","weightHisto",100,340,1500)
    m_tt_arr = np.array([])
    print("Filling histo with 1./m_tt")
    for m,weight in zip(outY_train, weights_train):
        weightHisto.Fill(m,abs(weight))

    weightHisto.Scale(1./weightHisto.Integral())
    maximumBin = weightHisto.GetMaximumBin()
    maximumBinContent = weightHisto.GetBinContent(maximumBin)

    canvas = ROOT.TCanvas()
    weightHisto.Draw("histe")
    canvas.SaveAs("weights_m.pdf")

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
    canvas.SetLogy(1)
    canvas.SaveAs("weights.pdf")

    weights_train_original = weights_train
    weights_train_=[]
    for w,m in zip(weights_train,outY_train):
        weightBin = weightHisto.FindBin(m)
        addW = weightHisto.GetBinContent(weightBin)
        weights_train_.append(abs(w)*addW)
    weights_train = np.array(weights_train_)

    weights_train = 1./np.mean(weights_train)*weights_train

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

    optimizer = tf.keras.optimizers.Adam(lr = learningRate)

    print ("compiling model")
    model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanAbsolutePercentageError(),metrics=['mean_absolute_error','mean_squared_error'])

    callbacks=[]
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', patience=3)
    earlyStop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=300, restore_best_weights=True)
    modelCheckpoint = tf.keras.callbacks.ModelCheckpoint(outFolder + '/model_best.h5', monitor='val_loss', save_best_only=True)

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

    kl = compute_kl_divergence(y_predicted, outY_test, n_bins=100)
    print ("KL", kl)
    js = compute_js_divergence(y_predicted, outY_test, n_bins=100)
    print ("JS", js)

    doEvaluationPlots(outY_train, y_predicted_train, weights_train_original, lkrM_train, krM_train, year = year, outFolder = outFolder+"/train/")
    doEvaluationPlots(outY_test, y_predicted, weights_test, lkrM_test, krM_test, year = year, outFolder = outFolder+"/test/")


    class_names=["rho"]
    max_display = inX_test.shape[1]
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
                                    inPathFolder = inPathFolder, year = "FR2",
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
    keep = None #[41,42,35,60,84,64,30,76,39,88,103]
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
        	         modelDir = "preUL05_28_02_22/rhoReg_madgraph/", modelName = "preUL05_rhoRegModel_28_02_22", outFolder="preUL05_28_02_22/rhoReg_madgraph/")
    else:
        print("Calling doTrainingAndEvaluation ...")
        doTrainingAndEvaluation(inPathFolder = "/nfs/dust/cms/user/celottog/TopRhoNetwork/rhoInput/powheg/", additionalName = "_preUL05_28_02_22_ttbar", year ="2016", tokeep = keep,
				modelName = "preUL05_rhoRegModel_28_02_22", outFolder="preUL05_28_02_22/rhoReg_madgraph/")

if __name__ == "__main__":
    main()
