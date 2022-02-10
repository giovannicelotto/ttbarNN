from utils import helpers, models,style
from sklearn.model_selection import train_test_split

import os

import numpy as np
import tensorflow as tf
from keras import optimizers,losses
import keras.backend as K
# K.tensorflow_backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))))
# K.tensorflow_backend.set_session(tf.compat.v1.Session()(config=tf.compat.v1.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))))
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tensorflow_smearing import network as smearLayer

def loadData(inPathFolder = "rhoInput/years", year = "2017", additionalName = "_3JetKinPlusRecoSolRight", testFraction = 0.2, overwrite = False,
            withBTag = False, withCharge = False, maxEvents = None):

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
        print ("\t need to load from root file with settings: nJets = "+str(3)+"; max Events = all")
        # inX, outY, weights = helpers.loadRhoDataFlat(inPathFolder+"/total"+year+".root", "miniTree", nJets = 3, maxEvents = 0,
        inX, outY, weights, lkrRho, krRho = helpers.loadRhoDataFlatPFN(inPathFolder+"/total"+year+".root", "miniTree", nJets = 3, maxEvents = 0 if maxEvents==None else maxEvents,
                                                    withBTag=withBTag, withCharge=withCharge)

        inX=np.array(inX)
        outY=np.array(outY)
        weights=np.array(weights)
        lkrRho=np.array(lkrRho)
        krRho=np.array(krRho)

        np.save(inPathFolder+"/flat_inX"+additionalName+".npy", inX)
        np.save(inPathFolder+"/flat_outY"+additionalName+".npy", outY)
        np.save(inPathFolder+"/flat_weights"+additionalName+".npy", weights)
        np.save(inPathFolder+"/flat_lkrRho"+additionalName+".npy", lkrRho)
        np.save(inPathFolder+"/flat_krRho"+additionalName+".npy", krRho)

    inX = np.load(inPathFolder+"/flat_inX"+additionalName+".npy")
    outY = np.load(inPathFolder+"/flat_outY"+additionalName+".npy")
    weights = np.load(inPathFolder+"/flat_weights"+additionalName+".npy")
    lkrRho = np.load(inPathFolder+"/flat_lkrRho"+additionalName+".npy")
    krRho = np.load(inPathFolder+"/flat_krRho"+additionalName+".npy")

    if maxEvents is not None:
        inX = inX[:maxEvents]
        outY = outY[:maxEvents]
        weights = weights[:maxEvents]
        lkrRho = lkrRho[:maxEvents]
        krRho = krRho[:maxEvents]

    inX = inX.reshape((inX.shape[0],10,4))

    # from sklearn.preprocessing import *
    outY = outY.reshape((outY.shape[0], 1))

    # transform y to uniform
    from sklearn import preprocessing
    scaler = preprocessing.QuantileTransformer(n_quantiles = 100000, output_distribution='uniform', subsample = int(1e6), random_state = int(1)).fit(outY)
    # outY = scaler.transform(outY)

    # inX_train, inX_test, outY_train, outY_test, weights_train, weights_test= train_test_split(inX, outY, weights, test_size = testFraction, shuffle=True)
    inX_train, inX_test, outY_train, outY_test, weights_train, weights_test, lkrRho_train, lkrRho_test, krRho_train, krRho_test = train_test_split(inX, outY, weights, lkrRho, krRho, test_size = testFraction)

    print ("\t data splitted succesfully")
    print ("loadData end")

    print("N training events: ",inX_train.shape[0])
    print(inX_train.shape)
    # print(weights_train.shape)
    # print(inX_test.shape)
    # print(outY_test.shape)
    # print(weights_test.shape)

    return inX_train, inX_test, outY_train, outY_test, weights_train, weights_test,lkrRho_train, lkrRho_test, krRho_train, krRho_test, scaler
    # return inX_train, inX_test, outY_train, outY_test, weights_train, weights_test,lkrRho,krRho, scaler


def doEvaluationPlots(yTest, yPredicted, weightTest, lkrRho, krRho, year = "", outFolder = "rhoOutput/"):

    import ROOT
    import matplotlib.pyplot as plt

    outDir = outFolder+"/"+year+"/"
    if not os.path.exists(outDir):
        os.makedirs(outDir)

    f = plt.figure()
    # values, bins, patches = plt.hist(yTest, bins=10, range=(0,1), label="true", alpha=0.5)
    # values2, bins2, patches2 = plt.hist(yPredicted, bins=10, range=(0,1), label="reco", alpha=0.5)
    values, bins, patches = plt.hist(yTest, bins=100, range=(0,1), label="true", alpha=0.5)
    values2, bins2, patches2 = plt.hist(yPredicted, bins=100, range=(0,1), label="reco", alpha=0.5)
    plt.legend(loc = "best")
    plt.ylabel('Events')
    plt.xlabel('rho')
    f.savefig(outDir+"eval_rho.pdf")

    style.style1d()
    s = style.style1d()
    histoShape = ROOT.TH1F( "reco shape", ";rho", 50, 0, 1)
    histoShape2 = ROOT.TH1F( "true shape2", ";rho", 50, 0, 1)
    histoShape3 = ROOT.TH1F( "lkr shape2", ";rho", 50, 0, 1)
    histoShape4 = ROOT.TH1F( "kr shape2", ";rho", 50, 0, 1)
    histoShape5 = ROOT.TH1F( "true shape no kinReco", ";rho", 50, 0, 1)
    histoShape6 = ROOT.TH1F( "reg shape no kinReco", ";rho", 50, 0, 1)

    for rhotrue, rhopred, weight, lkrR, krR in zip(yTest, yPredicted, weightTest,lkrRho, krRho):
        if lkrR>0.:
            histoShape3.Fill(lkrR, weight)
        if krR >0.:
            histoShape4.Fill(krR, weight)
            # histoShape.Fill(krR+rhopred, weight)
        else:
            histoShape6.Fill(rhopred, weight)
            histoShape5.Fill(rhotrue, weight)
        histoShape.Fill(rhopred, weight)
        histoShape2.Fill(rhotrue, weight)

    histoShape7 = histoShape6.Clone()
    histoShape7.Add(histoShape4)

    max_ = max(histoShape.GetMaximum(),histoShape2.GetMaximum(),histoShape3.GetMaximum(),histoShape4.GetMaximum(),histoShape5.GetMaximum(),histoShape6.GetMaximum(),histoShape7.GetMaximum())
    histoShape.SetMaximum(max_)
    histoShape2.SetMaximum(max_)
    histoShape3.SetMaximum(max_)
    histoShape4.SetMaximum(max_)
    histoShape5.SetMaximum(max_)
    histoShape6.SetMaximum(max_)
    histoShape7.SetMaximum(max_)

    c0=ROOT.TCanvas("c0","c0",800,800)
    histoShape.SetLineColor(ROOT.kRed)
    # histoShape.Scale(1./histoShape.Integral())
    histoShape.Draw("h0")
    # histoShape2.Scale(1./histoShape2.Integral())
    histoShape2.Draw("h0same")
    # histoShape3.Scale(1./histoShape3.Integral())
    histoShape3.SetLineColor(ROOT.kBlue)
    histoShape3.Draw("h0same")
    # histoShape4.Scale(1./histoShape4.Integral())
    histoShape4.SetLineColor(ROOT.kGreen+1)
    histoShape4.Draw("h0same")
    # histoShape5.Scale(1./histoShape5.Integral())
    histoShape5.SetLineColor(ROOT.kRed)
    histoShape5.SetLineStyle(ROOT.kDashed)
    # histoShape5.Draw("h0same")
    # histoShape6.Scale(1./histoShape6.Integral())
    histoShape6.SetLineColor(ROOT.kBlack)
    histoShape6.SetLineStyle(ROOT.kDashed)
    # histoShape6.Draw("h0same")
    # histoShape7.Scale(1./histoShape7.Integral())
    histoShape7.SetLineColor(ROOT.kGreen+1)
    histoShape7.SetLineStyle(ROOT.kDashed)
    histoShape7.Draw("h0same")
    l = ROOT.TLegend(0.2,0.7,0.4,0.9)
    l.AddEntry(histoShape, "DNN")
    l.AddEntry(histoShape2, "truth")
    l.AddEntry(histoShape3, "LKR")
    l.AddEntry(histoShape4, "KR")
    # l.AddEntry(histoShape5, "DNN(noKR)")
    # l.AddEntry(histoShape6, "truth(noKR)")
    l.AddEntry(histoShape7, "KR +reg(noKR)")
    l.Draw()
    c0.SaveAs(outDir+"shapes.pdf")

    style.style2d()
    s = style.style2d()
    histo = ROOT.TH2F( "reg rho diff", ";rho reco;#rho true - #rho reco", 500, 0, 1, 1000, -1, 1 )
    histo.SetDirectory(0)
    histo_kr = ROOT.TH2F( "kr rho diff", ";rho reco;#rho true - #rho reco", 500, 0, 1, 1000, -1, 1 )
    histo_kr.SetDirectory(0)
    histo_krReg = ROOT.TH2F( "kr+Reg rho diff", ";rho reco;#rho true - #rho reco", 500, 0, 1, 1000, -1, 1 )
    histo_krReg.SetDirectory(0)
    histo_lkr = ROOT.TH2F( "lkr rho diff", ";rho reco;#rho true - #rho reco", 500, 0, 1, 1000, -1, 1 )
    histo_lkr.SetDirectory(0)
    histTrue = ROOT.TH1F( "true contents", ";rho", 20, 0, 1)
    histTrue.SetDirectory(0)
    histoRecoGen = ROOT.TH2F( "reg rho2d", ";#rho reco;#rho true", 500, 0, 1, 500, 0, 1 )
    histoRecoGen.SetDirectory(0)
    histoRecoGen_kr = ROOT.TH2F( "kr rho2d", ";#rho reco;#rho true", 500, 0, 1, 500, 0, 1 )
    histoRecoGen_kr.SetDirectory(0)
    histoRecoGen_krReg = ROOT.TH2F( "kr+reg rho2d", ";#rho reco;#rho true", 500, 0, 1, 500, 0, 1 )
    histoRecoGen_krReg.SetDirectory(0)
    histoRecoGen_lkr = ROOT.TH2F( "lkr rho2d", ";#rho reco;#rho true", 500, 0, 1, 500, 0, 1 )
    histoRecoGen_lkr.SetDirectory(0)
    histoRecoGen2 = ROOT.TH2F( "reg resp", ";reco bin;true bin", 20, 0, 1, 20, 0, 1 )
    histoRecoGen2.SetDirectory(0)
    histoRecoGen2_kr = ROOT.TH2F( "kr resp", ";reco bin;true bin", 20, 0, 1, 20, 0, 1 )
    histoRecoGen2_kr.SetDirectory(0)
    histoRecoGen2_krReg = ROOT.TH2F( "kr+reg resp", ";reco bin;true bin", 20, 0, 1, 20, 0, 1 )
    histoRecoGen2_krReg.SetDirectory(0)
    histoRecoGen2_lkr = ROOT.TH2F( "lkr resp", ";reco bin;true bin", 20, 0, 1, 20, 0, 1 )
    histoRecoGen2_lkr.SetDirectory(0)


    # print (yTest.shape, yPredicted.shape)
    # for rhotrue, rhopred, weight in zip(yTest, yPredicted, weightTest):
    for rhotrue, rhopred, weight, lkrR, krR in zip(yTest, yPredicted, weightTest,lkrRho, krRho):
        diff = rhotrue-rhopred
        histo.Fill(rhotrue, diff, weight)
        histoRecoGen.Fill(rhopred, rhotrue, weight)
        histoRecoGen2.Fill(rhopred, rhotrue, weight)
        histTrue.Fill(rhotrue, weight)
        if krR>0.:
            diff_kr = rhotrue-krR
            histo_kr.Fill(rhotrue, diff_kr, weight)
            histoRecoGen_kr.Fill(krR, rhotrue, weight)
            histoRecoGen2_kr.Fill(krR, rhotrue, weight)
            # histo_krReg.Fill(rhotrue, diff_kr-rhopred, weight)
            # histoRecoGen_krReg.Fill(krR+rhopred, rhotrue, weight)
            # histoRecoGen2_krReg.Fill(krR+rhopred, rhotrue, weight)
            histo_krReg.Fill(rhotrue, diff_kr, weight)
            histoRecoGen_krReg.Fill(krR, rhotrue, weight)
            histoRecoGen2_krReg.Fill(krR, rhotrue, weight)
        else:
            histo_krReg.Fill(rhotrue, diff, weight)
            histoRecoGen_krReg.Fill(rhopred, rhotrue, weight)
            histoRecoGen2_krReg.Fill(rhopred, rhotrue, weight)
        if lkrR>0.:
            diff_lkr = rhotrue-lkrR
            histo_lkr.Fill(rhotrue, diff_lkr, weight)
            histoRecoGen_lkr.Fill(lkrR, rhotrue, weight)
            histoRecoGen2_lkr.Fill(lkrR, rhotrue, weight)

    hResp = histoRecoGen2.Clone()
    hResp_kr = histoRecoGen2_kr.Clone()
    hResp_lkr = histoRecoGen2_lkr.Clone()
    hResp_krReg = histoRecoGen2_krReg.Clone()

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

    def doRMSandMean(histo, name):
        style.style1d()
        s = style.style1d()
        c=ROOT.TCanvas()
        Xnb=20
        Xr1=0.
        Xr2=1.
        dXbin=(Xr2-Xr1)/((Xnb));
        titleRMSVsGen_ptTop_full ="; #rho_{true};RMS"
        titleMeanVsGen_ptTop_full ="; #rho_{true};Mean"
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
        # c.SaveAs(outDir+"reg_rms.pdf")
        c.SaveAs(outDir+name+"_rms.pdf")
        c.Clear()
        h_meanVsGen_.SetStats(0)
        h_meanVsGen_.Draw()
        # c.SaveAs(outDir+"reg_mean.pdf")
        c.SaveAs(outDir+name+"_mean.pdf")

    doRMSandMean(histo, "reg")
    doRMSandMean(histo_kr, "kr")
    doRMSandMean(histo_krReg, "krReg")
    doRMSandMean(histo_lkr, "lkr")

    def doPSE(hResp, hGen, name):
        style.style1d()
        s = style.style1d()
        c = ROOT.TCanvas()
        n = hResp.GetNbinsX()
        hp = ROOT.TH1D("", "", n, 0.0, n)
        hs = ROOT.TH1D("", "", n, 0.0, n)
        he = ROOT.TH1D("", "", n, 0.0, n)

        for b in range(1,n+1):
            GenInBinAndRecInBin = hResp.GetBinContent(b, b)
            RecInBin = hResp.Integral(b, b, 1, n)
            GenInBinAndRec = hResp.Integral(1, n, b, b)
            GenInBinAll = hGen.GetBinContent(b)
            hp.SetBinContent(b, GenInBinAndRecInBin / RecInBin)
            hs.SetBinContent(b, GenInBinAndRecInBin / GenInBinAndRec)
            he.SetBinContent(b, GenInBinAndRec / GenInBinAll)

        leg = ROOT.TLegend(0.15, 0.67, 0.40, 0.85)
        leg.SetTextFont(62)
        hr = ROOT.TH2D("", "", 1, he.GetBinLowEdge(1), he.GetBinLowEdge(n + 1), 1, 0.0, 1.0)
        hr.GetXaxis().SetTitle("Bin")
        hr.SetStats(0)
        hr.Draw()
        hp.SetMarkerColor(4)
        hp.SetMarkerStyle(23)
        hp.SetMarkerSize(markerSize)
        leg.AddEntry(hp, "Purity", "p")
        drawAsGraph(hp)
        hs.SetMarkerColor(2)
        hs.SetMarkerStyle(22)
        hs.SetMarkerSize(markerSize)
        leg.AddEntry(hs, "Stability", "p")
        drawAsGraph(hs)
        he.SetMarkerColor(8)
        he.SetMarkerStyle(20)
        he.SetMarkerSize(markerSize)
        leg.AddEntry(he, "Efficiency", "p")
        drawAsGraph(he)

        leg.Draw()
        c.SaveAs(outDir+name+"_pse.pdf")

    # doPSE(hResp_kr, histTrue, "kr")


def drawAsGraph(h):
    g = ROOT.TGraphAsymmErrors()
    for b in range(h.GetNbinsX()):
        x = h.GetBinLowEdge(b + 1) + offset * h.GetBinWidth(b + 1)
        y = h.GetBinContent(b + 1)
        uncLowY = h.GetBinError(b + 1)
        uncHighY = h.GetBinError(b + 1)
        g.SetPoint(b, x, y)
        g.SetPointError(b, 0.0, 0.0, uncLowY, uncHighY)
    g.SetLineColor(h.GetLineColor())
    g.SetMarkerColor(h.GetMarkerColor())
    g.SetMarkerStyle(h.GetMarkerStyle())
    g.SetLineStyle(h.GetLineStyle())
    g.SetMarkerSize(h.GetMarkerSize())
    drawOption = option
    # if(!flagUnc):
    #     drawOption += "X"
    g.Draw(drawOption)

    return g


def doTrainingAndEvaluation():
    # inX_train, inX_test, outY_train, outY_test, weights_train, weights_test,lkrRho, krRho, scaler = loadData(
    inX_train, inX_test, outY_train, outY_test, weights_train, weights_test,lkrRho_train, lkrRho_test, krRho_train, krRho_test, scaler = loadData(
                                    inPathFolder = "rhoInput/years/partonLVL", year = "2018",
                                    additionalName = "_2018_partonlvl_PtEtaPhiM_3JetKin_pfn", testFraction = 0.2,
                                    overwrite = False, withBTag = False, withCharge = False,
                                    maxEvents = None)
                                    # maxEvents = 10111)


    print (outY_train.shape)
    print (inX_train[0])

    inX_train=inX_train.reshape(inX_train.shape[0],40)
    inX_test=inX_test.reshape(inX_test.shape[0],40)
    inX_train=np.delete(inX_train,[24,25,26,27,   28,29,30,31,   32,33,34,35,   36,37,38,39],1)
    inX_test=np.delete(inX_test,  [24,25,26,27,   28,29,30,31,   32,33,34,35,   36,37,38,39],1)
    inX_train=inX_train.reshape(inX_train.shape[0],int(inX_test.shape[1]/4),4)
    inX_test=inX_test.reshape(inX_test.shape[0],int(inX_test.shape[1]/4),4)

    print (outY_train.shape)
    print (inX_train[0])

    learningRate = 0.01
    # batchSize = 2000
    batchSize = 200
    dropout = 0.05
    nDense = 5
    nNodes = 50
    regRate = 1e-7
    activation = 'relu'
    outputActivation = 'linear'



    model = models.getRhoRegModelFlat_new(n_particles = 6, regRate = regRate,batch_size=batchSize, activation = activation, dropout = dropout, nDense = nDense,
                                      nNodes = nNodes, inputDim = inX_train.shape[1], outputActivation = outputActivation)
    optimizer = tf.keras.optimizers.Adam(lr = learningRate)



    model.compile(optimizer=optimizer, loss='mean_squared_error',metrics=['mean_absolute_error','mean_squared_error'])

    callbacks_=[]
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', patience=3)
    earlyStop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=10)


    callbacks_.append(reduce_lr)
    callbacks_.append(earlyStop)


    fit = model.fit(
            inX_train,
            outY_train,
            # sample_weight = weights_train,
            # sample_weight = addWeights_train,
            validation_split = 0.25,
            batch_size = batchSize,
            epochs = 150,
            # epochs = 3,
            shuffle = False,
            callbacks = callbacks_,
            verbose = 1)

    # y_predicted = model.predict(inX_test,batch_size = batchSize)
    y_predicted = model.predict(inX_test)
    # print ("y_predicted.shape",y_predicted.shape)
    y_predicted = y_predicted.reshape(y_predicted.shape[0])
    outY_test = outY_test.reshape(outY_test.shape[0])
    krRho_test = krRho_test.reshape(krRho_test.shape[0])
    lkrRho_test = lkrRho_test.reshape(lkrRho_test.shape[0])



    from scipy.stats.stats import pearsonr
    print(pearsonr(outY_test, y_predicted)[0])

    outDir = "rhoOutput_pfn/"
    # outName = "rhoRegModel_optim_FR2lessWithNu"
    doEvaluationPlots(outY_test, y_predicted , weights_test, lkrRho_test, krRho_test, year = "2017", outFolder = outDir)
    # if not os.path.exists(outDir):
    #     os.makedirs(outDir)
    # model.save(outDir+outName+".h5")
    # tf.keras.backend.clear_session()
    # tf.keras.backend.set_learning_phase(0)
    # model = tf.keras.models.load_model(outDir+outName+".h5")
    # print('inputs: ', [input.op.name for input in model.inputs])
    # print('outputs: ', [output.op.name for output in model.outputs])

    # frozen_graph = helpers.freeze_session(tf.keras.backend.get_session(), output_names=[out.op.name for out in model.outputs])
    # tf.train.write_graph(frozen_graph, outDir+'/', outName+'.pbtxt', as_text=True)
    # tf.train.write_graph(frozen_graph, outDir+'/', outName+'.pb', as_text=False)


def main():
    doTrainingAndEvaluation()



if __name__ == "__main__":
    main()
