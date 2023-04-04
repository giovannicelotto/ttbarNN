import ROOT
import matplotlib.pyplot as plt
import os
#from style import *
import time
import matplotlib as mpl
#from utils.style import *
import numpy as np
from scipy.stats import moment
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tensorflow import keras
from fractions import Fraction
from decimal import Decimal, getcontext
ROOT.gROOT.SetBatch(True)

logbins =  np.concatenate((np.linspace(320, 500, 16, endpoint=False), [500, 520, 540, 560, 580, 600, 650, 750])) #]
#logbins = np.concatenate(([335], logbins))
#recoBin = np.array([1, 410, 500, 670, 1500]) #1500
recoBin = np.array([1, 380, 450, 550, 625, 812.5,  1200, 5000]) #1200
#recoBinPlusBin = np.array([1, 380, 450, 500, 625, 812.5,  1200, 5000])
nRecoBin = len(recoBin)-1
def GetRecoBin():
    return recoBin

def FillHisto(mtrue, mpred, histos):
    for i in range(0,4):
        if (mtrue>recoBin[i] and mtrue<recoBin[i+1]):
            histos[i].Fill(mpred-mtrue)

def ResponseMatrix(matrixName, y, yGen, outFolder, totGen,recoBinPlusBin = recoBin):
    '''create a response matrix with an additional bin used to count overflow event'''
    nRecoBinPlusBin = len(recoBinPlusBin)-1
    matrix = np.empty((nRecoBinPlusBin, nRecoBinPlusBin))
    for i in range(nRecoBinPlusBin):
        maskGen = (yGen >= recoBinPlusBin[i]) & (yGen < recoBinPlusBin[i+1])
       
        maskNorm = (totGen >= recoBinPlusBin[i]) & (totGen < recoBinPlusBin[i+1])
        normalization = len(maskNorm[maskNorm])
        for j in range(nRecoBinPlusBin):
            #print(f"Bin generated #{i}\tBin reconstructed #{j}\r", end="")
            maskRecoGen = (y >= recoBinPlusBin[j]) & (y < recoBinPlusBin[j+1]) & maskGen
            entries = len(maskRecoGen[maskRecoGen])
            #print("entries",entries)
            if normalization == 0:
                matrix[j, i] = 0
            else:
                matrix[j, i] = entries/normalization

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    im = ax.matshow(matrix[:nRecoBin,:nRecoBin], cmap=plt.cm.jet, alpha=0.7, norm=mpl.colors.LogNorm())
    ax.set_title("Generated (bin number)", fontsize=18)
    #ax.set_xlabel("Generated (bin number)", fontsize=18)
    ax.set_ylabel("Reconstructed (bin number)", fontsize=18)
    ax.tick_params(labelsize=18)
    ax.xaxis.set_ticks(np.arange(0, nRecoBinPlusBin, 1.0))
    ax.yaxis.set_ticks(np.arange(0, nRecoBinPlusBin, 1.0))
    for y in range(nRecoBinPlusBin):
        for x in range(nRecoBinPlusBin):
            plt.text(x , y , '%.3f' % matrix[y, x],
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=16
                    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.25)
    cbar = plt.colorbar(im, ax=ax, cax=cax)
    cbar.set_label(r'', fontsize=18)
    cbar.ax.tick_params(labelsize=18)
    fig.savefig(outFolder+"/"+matrixName+".pdf", bbox_inches='tight')
    plt.cla()
    return matrix


def covarianceMatrix(y, matrix, recoBin = recoBin):
    counts_reco, bins_reco = np.histogram(y, bins=recoBin, density=False)
    dy = np.diag(counts_reco)
    try:
        A_inv = np.linalg.inv(matrix)
        At = matrix.T
        cov = (A_inv.dot(dy)).dot(A_inv.T)
        return cov
    except:
        print("Singular Matrix\n\n\n\n\nCheck the matrix")



def doEvaluationPlots(yTest, yPredicted, weightTest, lkrM, krM, outFolder, totGen):
    if not os.path.exists(outFolder):
        os.makedirs(outFolder)
    
    yTest = yTest[:,0]
    yPredicted = yPredicted[:,0]
# *******************************
# *                             *
# *     Invariant Mass Plot     *
# *                             *
# ******************************* 
    print("Invariant Mass plot: Predicted vs True")
    f, ax = plt.subplots(1, 1)
    ax.hist(yPredicted,    bins = 80, range=(200,1500), label="DNN",   histtype=u'step', alpha=0.8) #bins 100 1500
    ax.hist(lkrM,          bins = 80, range=(200,1500), label="Loose",  histtype=u'step', alpha=0.5)
    ax.hist(krM,           bins = 80, range=(200,1500), label="kinReco",histtype=u'step', alpha=0.5)
    ax.hist(yTest,         bins = 80, range=(200,1500), label="True",   histtype=u'step', alpha=0.8)
    ax.vlines(x=340, ymin=0, ymax=50)
    ax.legend(loc = "best", fontsize=18)
    ax.tick_params(labelsize=16)
    ax.set_ylabel('Events', fontsize=18)
    ax.set_xlabel('m$_{tt}$ (GeV/c$^2$)', fontsize=18)
    f.savefig(outFolder+"/m_tt.pdf", bbox_inches='tight')
    plt.close()
    print("Invariant Mass plot saved in"+ outFolder+"/m_tt.pdf")



# *******************************
# *                             *
# * Purity Stability Efficinecy *
# *                             *
# ******************************* 
    print("Purity Stability Efficiency") 
    GenBinith = np.array([])
    RecoBinith = np.array([])
    GenRecoBinith = np.array([])
    for i in range(len(recoBin)-1):
        maskTotGen = np.array((totGen >= recoBin[i]) & (totGen<recoBin[i+1]))           # all the generated
        maskGen = np.array((yTest >= recoBin[i]) & (yTest<recoBin[i+1]))                # all the generated that satisfy the cuts
        maskReco = np.array((yPredicted >= recoBin[i]) & (yPredicted<recoBin[i+1]))     # all the reconstructed in bin
        maskGenReco = maskGen & maskReco
        GenBinith = np.append(GenBinith, len(maskTotGen[maskTotGen==True]))                              # Generated in bin i-th
        RecoBinith = np.append(RecoBinith, len(maskReco[maskReco==True]))                           # Reconstructed in bin ith
        GenRecoBinith = np.append(GenRecoBinith,  len(maskGenReco[maskGenReco==True]))                        # Generated and Reconstructed in bin ith
            
    
    fig, ax = plt.subplots(1, 1)
    dx_ = (recoBin[1:] - recoBin[:len(recoBin)-1])/2
    x_ = recoBin[:len(recoBin)-1] + dx_

    ax.errorbar(x_, GenRecoBinith/GenBinith, None, dx_, linestyle='None', label='Purity')
    ax.errorbar(x_, GenRecoBinith/RecoBinith, None, dx_, linestyle='None', label='Stability')
    ax.set_xlabel("m$_{tt}^{True}$")
    ax.legend(fontsize=18, loc='best',bbox_to_anchor=(1, 0.5))
    fig.savefig(outFolder+"/pse.pdf", bbox_inches='tight')
    plt.cla()



# *******************************
# *                             *
# *       Means and RMS         *
# *                             *
# *******************************  
    regMeanBinned = np.array([])
    LooseMeanBinned = np.array([])
    kinMeanBinned = np.array([])

    regSquaredBinned = np.array([])
    LooseSquaredBinned = np.array([])
    kinSquaredBinned = np.array([])
    
    regRmsBinned = np.array([])
    kinRmsBinned = np.array([])
    LooseRmsBinned = np.array([])
    
    regErrRmsBinned = np.array([])
    kinErrRmsBinned = np.array([])
    LooseErrRmsBinned = np.array([])
		
    regElementsPerBin = np.array([])
    kinElementsPerBin = np.array([])
    LooseElementsPerBin = np.array([])
# Makes sense to compare only when kin and loose works

    logbins_wu = np.concatenate(([1], logbins, [5000])) 
    #consider all the events below 340 in one bin.

    for i in range(len(logbins_wu)-1):
        mask = (yTest >= logbins_wu[i]) & (yTest < logbins_wu[i+1])
        kinMask =  (mask) 
        LooseMask =  (mask) 
        regMeanBinned       = np.append( regMeanBinned      ,   np.mean(yPredicted[mask]-yTest[mask]))
        kinMeanBinned       = np.append( kinMeanBinned      ,   np.mean(krM[kinMask]-yTest[kinMask]))
        LooseMeanBinned     = np.append( LooseMeanBinned    ,   np.mean(lkrM[LooseMask]-yTest[LooseMask]))

        regSquaredBinned       = np.append( regSquaredBinned      ,   np.mean((yPredicted[mask]-yTest[mask])**2))
        kinSquaredBinned       = np.append( kinSquaredBinned      ,   np.mean((krM[kinMask]-yTest[kinMask])**2))
        LooseSquaredBinned     = np.append( LooseSquaredBinned    ,   np.mean((lkrM[LooseMask]-yTest[LooseMask])**2))

        regRmsBinned        = np.append( regRmsBinned       ,   np.std(yPredicted[mask]-yTest[mask]))
        kinRmsBinned        = np.append( kinRmsBinned       ,   np.std(krM[kinMask]-yTest[kinMask]))
        LooseRmsBinned      = np.append( LooseRmsBinned     ,   np.std(lkrM[LooseMask]-yTest[LooseMask]))
        
        regElementsPerBin      = np.append( regElementsPerBin     ,  len(mask[mask==1]))
        kinElementsPerBin      = np.append( kinElementsPerBin     ,  len(kinMask[kinMask==1]))
        LooseElementsPerBin      = np.append( LooseElementsPerBin     ,  len(LooseMask[LooseMask==1]))

        regErrRmsBinned     = np.append( regErrRmsBinned    ,   np.sqrt((moment(yPredicted[mask]-yTest[mask], 4)-((regElementsPerBin[i]-3)/(regElementsPerBin[i]-1)*regRmsBinned**4)[0])/regElementsPerBin[i]))
        LooseErrRmsBinned   = np.append( LooseErrRmsBinned  ,   np.sqrt((moment(lkrM[LooseMask]-yTest[LooseMask], 4)-((LooseElementsPerBin[i]-3)/(LooseElementsPerBin[i]-1)*LooseRmsBinned**4)[0])/LooseElementsPerBin[i]))
        kinErrRmsBinned     = np.append( kinErrRmsBinned    ,   np.sqrt((moment(krM[kinMask]-yTest[kinMask], 4)-((kinElementsPerBin[i]-3)/(kinElementsPerBin[i]-1)*kinRmsBinned**4)[0])/kinElementsPerBin[i]))


    logbins_wu[0] = 320 # for visualization purpose the underflow is between 320, 340
    logbins_wu[-1] = 800

    errX = (logbins_wu[1:]-logbins_wu[:len(logbins_wu)-1])/2
    x = logbins_wu[:len(logbins_wu)-1] + errX
    ax.errorbar(x, regMeanBinned,  regRmsBinned/np.sqrt(regElementsPerBin), errX, linestyle='None', label = 'DNN')
    ax.errorbar(x, LooseMeanBinned,  LooseRmsBinned/np.sqrt(LooseElementsPerBin), errX, linestyle='None', label = 'Loose')
    ax.errorbar(x, kinMeanBinned,  kinRmsBinned/np.sqrt(kinElementsPerBin), errX, linestyle='None', label = 'Kin')
    ax.set_xlabel("$m_{tt}^{True}$ (GeV)")
    ax.hlines(y=0, xmin=300, xmax=800, linestyles='dotted', color='red')
    ax.set_ylabel("Mean(Pred - True) [GeV]")
    ax.legend(fontsize=18)
    fig.savefig(outFolder+"/allMeans.pdf", bbox_inches='tight')

    fig, ax = plt.subplots(1, 1)
    ax.errorbar(x, regRmsBinned, regErrRmsBinned/10, errX, linestyle='None', label = 'DNN err/10')
    ax.errorbar(x, LooseRmsBinned, LooseErrRmsBinned/10, errX, linestyle='None', label = 'Loose err/10')
    ax.errorbar(x, kinRmsBinned, kinErrRmsBinned/10, errX, linestyle='None', label = 'Kin err/10')
    ax.legend(fontsize=18, loc='best')
    ax.set_ylabel("RMS(Pred-True) [GeV]", fontsize=18)
    fig.savefig(outFolder+"/allRms.pdf", bbox_inches='tight')
    plt.cla()

    
    ax.errorbar(x, np.sqrt(regSquaredBinned), None, errX, linestyle='None', label = 'DNN ')
    ax.errorbar(x, np.sqrt(LooseSquaredBinned), None, errX, linestyle='None', label = 'Loose ')
    ax.errorbar(x, np.sqrt(kinSquaredBinned), None, errX, linestyle='None', label = 'Kin ')
    ax.legend(fontsize=18, loc='best')
    ax.set_xlabel("m$_{tt}^{True}$")
    ax.tick_params(labelsize=18)
    ax.set_ylabel("$\sqrt{<(m_{tt}^{Reg}-m_{tt}^{True})^2>}$ [GeV]", fontsize=16)
    fig.savefig(outFolder+"/allSquared.pdf", bbox_inches='tight')
    plt.cla()
    
# *******************************
# *                             *
# *      Response Matrix        *
# *                             *
# ******************************* 
    dnnMatrix = np.empty((nRecoBin, nRecoBin))
    print("DNN matrix")
    dnnMatrix = ResponseMatrix(y=yPredicted, yGen = yTest, matrixName = 'dnnMatrix', outFolder = outFolder, totGen=totGen)
    print("\nCondition Number of the DNN matrix:", np.linalg.cond(dnnMatrix))
    with open(outFolder+"/../model/Info.txt", "a+") as f:
        print("\nCondition Number of the DNN matrix:  "+str(np.linalg.cond(dnnMatrix))+"\n", file=f)
    #print(np.linalg.inv(dnnMatrix).dot(dnnMatrix))
    looseMatrix = np.empty((nRecoBin, nRecoBin))
    looseMatrix = ResponseMatrix(y=lkrM, yGen=yTest, matrixName = 'looseMatrix', outFolder = outFolder, totGen=totGen)

    kinMatrix = np.empty((nRecoBin, nRecoBin))
    kinMatrix = ResponseMatrix(y=krM, yGen=yTest, matrixName = 'kinMatrix', outFolder = outFolder, totGen=totGen)


# *******************************
# *                             *
# *      Diff cross section     *
# *                             *
# ******************************* 
    
# !!! agggiungere un boolean che ti dice se e andato a buon fine e in caso contrario non stampare nessuna matrice

    dnnCov      = covarianceMatrix(yPredicted, dnnMatrix)
    looseCov    = covarianceMatrix(lkrM, looseMatrix)
    kinCov      = covarianceMatrix(krM, kinMatrix)
    #counts_gen, bins_gen = np.histogram(yTest, bins=recoBin, density=False)
    

    #rho=np.empty([nRecoBin, nRecoBin])
    #for i in range(nRecoBin):
    #    for j in range(nRecoBin):
    #        rho[i, j] = cov[i, j] / np.sqrt(cov[i, i] * cov[j, j])
    '''ax.matshow(rho, cmap=plt.cm.jet, alpha=0.7)
    ax.set_title("Correlation Matrix", fontsize=18)
    ax.tick_params(labelsize=18)
    for y in range(nRecoBin):
        for x in range(nRecoBin):
            plt.text(x , y , '%.2f' % rho[y, x],
                    horizontalalignment='center',
                    verticalalignment='center',
                    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'$\rho$', fontsize=18)
    cbar.ax.tick_params(labelsize=18)
    fig.savefig(outFolder+"/rho.pdf", bbox_inches='tight')
    plt.cla()'''
    
    #fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(14, 10))
    fig.subplots_adjust(hspace=0.2)
    dnnCounts, b_ = np.histogram(yPredicted,   bins=recoBin, density=False)
    looseCounts,b_= np.histogram(lkrM,     bins=recoBin, density=False)
    kinCounts, b_= np.histogram(krM,            bins=recoBin, density=False)
    #genCounts, bins_gen, bars_gen = ax1.hist(yTest,         bins=recoBin, density=False, histtype=u'step', alpha=0.5, label='Gen mtt Reco', edgecolor='C3')
    trueCounts, binsTrue, barsTrue = ax1.hist(totGen,         bins=recoBin, density=False, histtype=u'step', alpha=0.5, label='True', edgecolor='black')
    ax1.set_xlim(100, recoBin[-1])
    ax2.set_xlim(100, recoBin[-1])
    x_ = recoBin[:len(recoBin)-1]
    dx_= (recoBin[1:]-recoBin[:len(recoBin)-1])/2
    ax1.errorbar(x_+dx_     , np.linalg.inv(dnnMatrix).dot(dnnCounts)       , yerr=np.sqrt(np.diag(dnnCov)), label='Unfolded DNN',      linestyle=' ', marker='o', markersize = 6, color='C0')
    ax1.errorbar(x_+4/3*dx_  , np.linalg.inv(looseMatrix).dot(looseCounts)   , yerr=np.sqrt(np.diag(looseCov)), label='Unfolded Loose',  linestyle=' ', marker='o', markersize = 6, color='C1')
    ax1.errorbar(x_+2/3*dx_  , np.linalg.inv(kinMatrix).dot(kinCounts)       , yerr=np.sqrt(np.diag(kinCov)), label='Unfolded kin',      linestyle=' ', marker='o', markersize = 6, color='C2')
    ax1.set_xscale('log')
    ax2.set_xscale('log')
    ax1.set_ylabel("Events", fontsize=18)
    ax2.set_ylabel("Relative Uncertainty", fontsize=18)
    #ax1.set_ylabel("Counts per %(first)d or %(second)d GeV/c" %{"first":bins[1]-bins[0], "second":(bins[len(bins)-1]-bins[len(bins)-2])} , fontsize=18)
    ax1.tick_params(labelsize=18)
    ax1.legend(fontsize=18)

    ax2.errorbar(x_+dx_     , np.linalg.inv(dnnMatrix).dot(dnnCounts)    - trueCounts   , yerr=np.sqrt(np.diag(dnnCov))/trueCounts  , label='Unfolded DNN', linestyle=' ', marker='o', markersize = 6, color='C0')
    ax2.errorbar(x_+4/3*dx_ , np.linalg.inv(looseMatrix).dot(looseCounts)- trueCounts   , yerr=np.sqrt(np.diag(looseCov))/(2*trueCounts), label='Unfolded Loose Err/10', linestyle=' ', marker='o', markersize = 6, color='C1')
    ax2.errorbar(x_+2/3*dx_ , np.linalg.inv(kinMatrix).dot(kinCounts)    - trueCounts   , yerr=np.sqrt(np.diag(kinCov))/trueCounts  , label='Unfolded kin', linestyle=' ', marker='o', markersize = 6, color='C2' )
    ax2.set_xlabel("m$_{tt}$ (GeV/c$^2$)", fontsize=18)
    ax2.tick_params(labelsize=18)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.4)

    fig.savefig(outFolder+"/diff_tt.pdf", bbox_inches='tight')
    plt.cla()
    np.set_printoptions(precision=3)
    print("sum of true counts", np.array(trueCounts).sum())
    print("sum of dnn counts", np.array(np.linalg.inv(dnnMatrix).dot(dnnCounts)).sum())
    #print("sum of kinn counts", np.array(dnnCounts).sum())
    
    print("Difference in: A*x - y" ,  dnnMatrix.dot(trueCounts)-dnnCounts)
    print("Difference in: A-1*y - x" ,np.linalg.inv(dnnMatrix).dot(dnnCounts)  - trueCounts)
    print("Errors in DNN unfolding:",   np.sqrt(np.diag(dnnCov)))
    print("Errors in Loose unfolding:", np.sqrt(np.diag(looseCov)))
    print("Errors in kin unfolding:",   np.sqrt(np.diag(kinCov)))
    with open(outFolder+"/../model/Info.txt", "a+") as f:
        print("Difference in: A*x - y"    , dnnMatrix.dot(trueCounts)-dnnCounts, file=f)
        print("Difference in: A-1*y - x"  , np.linalg.inv(dnnMatrix).dot(dnnCounts)  - trueCounts, file=f)
        print("Errors in DNN unfolding:"  , np.sqrt(np.diag(dnnCov)), file = f)
        print("Errors in Loose unfolding:", np.sqrt(np.diag(looseCov)), file=f)
        print("Errors in kin unfolding:"  , np.sqrt(np.diag(kinCov)), file=f)
#except:
  #      print("singular matrix")








# ROOT PLOT    




    '''hk1 = ROOT.TH1F( "kinRecoResolution1", "kinReco - True [%d < m < %d];\Delta m_{tt}" %(recoBin[0] ,recoBin[1]), 200, -400, 400)
    hk2 = ROOT.TH1F( "kinRecoResolution2", "kinReco - True [%d < m < %d];\Delta m_{tt}" %(recoBin[1] ,recoBin[2]), 200, -400, 400)
    hk3 = ROOT.TH1F( "kinRecoResolution3", "kinReco - True [%d < m < %d];\Delta m_{tt}" %(recoBin[2] ,recoBin[3]), 200, -400, 400)
    hk4 = ROOT.TH1F( "kinRecoResolution4", "kinReco - True [%d < m < %d];\Delta m_{tt}" %(recoBin[3] ,recoBin[4]), 200, -400, 400)
    hl1 = ROOT.TH1F( "LooseRecoResolution1", "LooseReco - True [%d < m < %d];\Delta m_{tt}" %(recoBin[0] ,recoBin[1]), 200, -400, 400)
    hl2 = ROOT.TH1F( "LooseRecoResolution2", "LooseReco - True [%d < m < %d];\Delta m_{tt}" %(recoBin[1] ,recoBin[2]), 200, -400, 400)
    hl3 = ROOT.TH1F( "LooseRecoResolution3", "LooseReco - True [%d < m < %d];\Delta m_{tt}" %(recoBin[2] ,recoBin[3]), 200, -400, 400)
    hl4 = ROOT.TH1F( "LooseRecoResolution4", "LooseReco - True [%d < m < %d];\Delta m_{tt}" %(recoBin[3] ,recoBin[4]), 200, -400, 400)
    hn1 = ROOT.TH1F( "DNNRecoResolution1", "DNNReco - True [%d < m < %d];\Delta m_{tt}" %(recoBin[0] ,recoBin[1]), 200, -400, 400)
    hn2 = ROOT.TH1F( "DNNRecoResolution2", "DNNReco - True [%d < m < %d];\Delta m_{tt}" %(recoBin[1] ,recoBin[2]), 200, -400, 400)
    hn3 = ROOT.TH1F( "DNNRecoResolution3", "DNNReco - True [%d < m < %d];\Delta m_{tt}" %(recoBin[2] ,recoBin[3]), 200, -400, 400)
    hn4 = ROOT.TH1F( "DNNRecoResolution4", "DNNReco - True [%d < m < %d];\Delta m_{tt}" %(recoBin[3] ,recoBin[4]), 200, -400, 400)
    kinH = [hk1, hk2, hk3, hk4]
    LooseH = [hl1, hl2, hl3, hl4]
    DnnH = [hn1, hn2, hn3, hn4]

    for Mtrue, Mpred, weight, lkrMass, krMass in zip(yTest, yPredicted, weightTest, lkrM, krM):
        if lkrMass>0.:
            FillHisto(Mtrue, lkrMass, LooseH)
        if krMass >0.:
            FillHisto(Mtrue, krMass, kinH)
        FillHisto(Mtrue, Mpred, DnnH)

    


    c2=ROOT.TCanvas("c2","c2",2400,800)
    c2.cd()
    defaultStyle()
    p1 = ROOT.TPad("pad1", "", 0    , 0     , 1./4  , 1.0)
    p2 = ROOT.TPad("pad2", "", 1./4 , 0     , 2./4  , 1.0)
    p3 = ROOT.TPad("pad3", "", 2./4 , 0.    , 3./4  , 1.0)
    p4 = ROOT.TPad("pad4", "", 3./4 , 0.    , 1.    , 1.0)
    p1.Draw()
    p2.Draw()
    p3.Draw()
    p4.Draw()
    
    pads = [p1,p2,p3,p4]
    for i in range(len(pads)):
        kinH[i].Scale(1./kinH[i].Integral())
        LooseH[i].Scale(1./LooseH[i].Integral())
        DnnH[i].Scale(1./DnnH[i].Integral())

    upperY_ = 1.4*max(kinH[0].GetMaximum(),kinH[1].GetMaximum(),kinH[2].GetMaximum(),kinH[3].GetMaximum(),
                      LooseH[0].GetMaximum(),LooseH[1].GetMaximum(),LooseH[2].GetMaximum(),LooseH[3].GetMaximum(),
                      DnnH[0].GetMaximum(),DnnH[1].GetMaximum(),DnnH[2].GetMaximum(),DnnH[3].GetMaximum())
    leg = ROOT.TLegend(0.55,0.6,0.9,0.85)
    leg.AddEntry(DnnH[0], "DNN","f")
    leg.AddEntry(LooseH[0], "LoosekinReco","f")
    leg.AddEntry(kinH[0], "FullkinReco","f")
    leg.SetTextSize(0.04)


    
    for i in range(len(pads)):
        #pads[i].Draw()
        pads[i].cd()
        kinH[i].SetLineColor(ROOT.kRed)
        kinH[i].GetYaxis().SetRangeUser(0, upperY_)
        kinH[i].SetStats(0)
        kinH[i].SetLineWidth(2)
        kinH[i].Draw("hist")
        LooseH[i].Draw("same")
        LooseH[i].SetStats(0)
        LooseH[i].SetLineWidth(2)
        DnnH[i].SetLineColor(ROOT.kGreen)
        DnnH[i].Draw("histsame")
        DnnH[i].SetStats(0)
        DnnH[i].SetLineWidth(2)
        if (i!=0):
            leg.Draw("histsame")
        #leg.SetFillStyle(0)


    c2.SaveAs(outFolder+"/resVsTrue.pdf")
    c2.Close()'''

# *******************************
# *                             *
# *     2D Resolution plots     *
# *                             *
# *******************************
    plt.close('all')

    index = 0      
    for el in [yPredicted, lkrM, krM]:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        fig.subplots_adjust(right=0.8)
          
        counts, xedges, yedges, im = ax.hist2d(yTest, el - yTest, bins=(80, 80), range=[[340, 1500], [-500, 500]] ,norm=mpl.colors.LogNorm(), cmap=plt.cm.jet)
        cbar_ax = fig.add_axes([0.81, 0.11, 0.05, 0.77])
        cbar = fig.colorbar(im, cax=cbar_ax)
        ax.set_xlabel("m$_{tt}^{True}$", fontsize=18)
        ax.set_ylabel(['m$_{tt}^{DNN}$', 'm$_{tt}^{Loose}$', 'm$_{tt}^{Kin}$'][index]+" - True", fontsize=18)
        ax.tick_params(labelsize=18)
        cbar.set_label('Counts', fontsize=19)
        cbar.ax.tick_params(labelsize=18)
        print(outFolder+"/"+['Reg', 'Loose', 'Kin'][index]+"MinusTrueVsTrue.pdf")
        fig.savefig(outFolder+"/"+['Reg', 'Loose', 'Kin'][index]+"MinusTrueVsTrue.pdf", bbox_inches='tight')
        plt.cla()   
        index = index +1

# *******************************
# *                             *
# *       Loss Functions        *
# *                             *
# *******************************

def doPlotLoss(fit, outFolder):
    #  "mean_absolute_error"
    '''plt.figure(0)
    plt.plot(fit.history['mean_absolute_error'])
    plt.plot(fit.history['val_mean_absolute_error'])
    plt.title('model mean_absolute_error')
    plt.ylabel('mean_absolute_error')
    plt.xlabel('epoch')
    # plt.yscale('log')
    plt.ylim(ymax = min(fit.history['mean_absolute_error'])*1.4, ymin = min(fit.history['mean_absolute_error'])*0.9)
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(outFolder+"mean_absolute_error.pdf")
    #  "mean_absolute_percentage_error"
    plt.figure(1)
    plt.plot(fit.history['mean_absolute_percentage_error'])
    plt.plot(fit.history['val_mean_absolute_percentage_error'])
    plt.title('model mean_absolute_percentage_error')
    plt.ylabel('mean_absolute_percentage_error')
    plt.xlabel('epoch')
    # plt.yscale('log')
    plt.ylim(ymax = min(fit.history['mean_absolute_percentage_error'])*1.4, ymin = min(fit.history['mean_absolute_percentage_error'])*0.9)
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(outFolder+"mean_absolute_percentage_error.pdf")'''
    # "Loss"
    plt.figure(2)
    plt.plot(fit.history['loss'])
    plt.plot(fit.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    # plt.yscale('log')
    plt.ylim(ymax = max(min(fit.history['loss']), min(fit.history['val_loss']))*1.4, ymin = min(min(fit.history['loss']),min(fit.history['val_loss']))*0.9)
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(outFolder+"/loss.pdf")




    #style.style2d()
    #s = style.style2d()
    '''
    histo = ROOT.TH2F( "histo",         "DNNMinusTrue_vs_True;    	m_{tt}^{true};		m_{tt}^{DNN} - m_{tt}^{true}"	  	,nlogbin, logbins, nlogbin_diff, logbins_diff )	
    histo_kr = ROOT.TH2F( "histo_kr", 		"krMinusTrue_vs_True;		m_{tt}^{true};		m_{tt}^{kinReco} - m_{tt}^{true}"	,nlogbin, logbins, nlogbin_diff,logbins_diff )		
    histo_krReg = ROOT.TH2F( "histo_krReg", 	"krRegMinusTrue_vs_True;	m_{tt}^{true};		m_{tt}^{kinReco} - m_{tt}^{true}"	,nlogbin, logbins, nlogbin_diff, logbins_diff )
    histo_lkr = ROOT.TH2F( "histo_lkr", 	"lrMinusTrue_vs_True;		m_{tt}^{true};		m_{tt}^{LooseReco} - m_{tt}^{true}"	,nlogbin, logbins, nlogbin_diff, logbins_diff )
    histTrue = ROOT.TH1F( "histTrue", 		"m_{tt}^{true} distribution;	m_{tt}^{true}", nlogbin, logbins)



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
    c.SaveAs(outFolder+"reg_GenRecoDiff2d.pdf")
    c.Clear()
    histo_kr.SetStats(0)
    histo_kr.Draw("colz")
    c.SaveAs(outFolder+"kr_GenRecoDiff2d.pdf")
    c.Clear()
    histo_krReg.SetStats(0)
    histo_krReg.Draw("colz")
    c.SaveAs(outFolder+"krReg_GenRecoDiff2d.pdf")
    c.Clear()
    histo_lkr.SetStats(0)
    histo_lkr.Draw("colz")
    c.SaveAs(outFolder+"lkr_GenRecoDiff2d.pdf")
    c.Clear()
#    histo_newHybrid.SetStats(0)
#    histo_newHybrid.Draw("colz")
#    c.SaveAs(outFolder+"newHybrid_GenRecoDiff2d.pdf")
#    c.Clear()
#    histo_newHybrid2.SetStats(0)
#    histo_newHybrid2.Draw("colz")
#    c.SaveAs(outFolder+"newHybrid2_GenRecoDiff2d.pdf")
#    c.Clear()

    histoRecoGen.Draw("colz")
    c.SetLogy()
    corrLatex = ROOT.TLatex()
    corrLatex.SetTextSize(0.65 * corrLatex.GetTextSize())
    corrLatex.DrawLatexNDC(0.65, 0.85, str(np.round(histoRecoGen.GetCorrelationFactor(),3)))
    c.SaveAs(outFolder+"reg_GenReco2d.png")
    c.Clear()
    
    histoRecoGen_kr.Draw("colz")
    corrLatex_kr = ROOT.TLatex()
    corrLatex_kr.SetTextSize(0.65 * corrLatex_kr.GetTextSize())
    corrLatex_kr.DrawLatexNDC(0.65, 0.85, str(np.round(histoRecoGen_kr.GetCorrelationFactor(),3)))
    c.SaveAs(outFolder+"kr_GenReco2d.png")
    c.Clear()
    c.SetLogy()
    histoRecoGen_krReg.Draw("colz")
    corrLatex_krReg = ROOT.TLatex()
    corrLatex_krReg.SetTextSize(0.65 * corrLatex_kr.GetTextSize())
    corrLatex_krReg.DrawLatexNDC(0.65, 0.85, str(np.round(histoRecoGen_kr.GetCorrelationFactor(),3)))
    c.SaveAs(outFolder+"krReg_GenReco2d.png")
    c.Clear()
    histoRecoGen_lkr.Draw("colz")
    corrLatex_lkr = ROOT.TLatex()
    corrLatex_lkr.SetTextSize(0.65 * corrLatex_lkr.GetTextSize())
    corrLatex_lkr.DrawLatexNDC(0.65, 0.85, str(np.round(histoRecoGen_lkr.GetCorrelationFactor(),3)))
    c.SaveAs(outFolder+"lkr_GenReco2d.png")
    c.Clear()
#    histoRecoGen_newHybrid.Draw("colz")
#    corrLatex_newHybrid = ROOT.TLatex()
#    corrLatex_newHybrid.SetTextSize(0.65 * corrLatex_newHybrid.GetTextSize())
#    corrLatex_newHybrid.DrawLatexNDC(0.65, 0.85, str(np.round(histoRecoGen_newHybrid.GetCorrelationFactor(),3)))
#    c.SaveAs(outFolder+"newHybrid_GenReco2d.pdf")
#    c.Clear()
#    histoRecoGen_newHybrid2.Draw("colz")
#    corrLatex_newHybrid2 = ROOT.TLatex()
#    corrLatex_newHybrid2.SetTextSize(0.65 * corrLatex_newHybrid2.GetTextSize())
#    corrLatex_newHybrid2.DrawLatexNDC(0.65, 0.85, str(np.round(histoRecoGen_newHybrid2.GetCorrelationFactor(),3)))
#    c.SaveAs(outFolder+"newHybrid2_GenReco2d.pdf")
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
    c.SaveAs(outFolder+"reg_response.pdf")

    c.Clear()'''

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
    c.SaveAs(outFolder+"newHybrid_response.pdf")

    c.Clear()'''
    '''
    histoRecoGen2.Scale(1./histoRecoGen2.Integral())
    # histoRecoGen2.Scale(100.)
    ROOT.gStyle.SetPaintTextFormat("1.2f");
    histoRecoGen2.SetStats(0)
    histoRecoGen2.SetMarkerSize(1)
#   histoRecoGen2.Draw("colz text")
    histoRecoGen2.Draw("colz")
    c.SaveAs(outFolder+"resp.pdf")
    c.Clear()
    histoRecoGen2_kr.Scale(1./histoRecoGen2_kr.Integral())
    ROOT.gStyle.SetPaintTextFormat("1.2f");
    histoRecoGen2_kr.SetStats(0)
    histoRecoGen2_kr.SetMarkerSize(1)
#    histoRecoGen2_kr.Draw("colz text")
    histoRecoGen2_kr.Draw("colz")
    c.SaveAs(outFolder+"kr_resp.pdf")
    c.Clear()
    histoRecoGen2_krReg.Scale(1./histoRecoGen2_krReg.Integral())
    ROOT.gStyle.SetPaintTextFormat("1.2f");
    histoRecoGen2_krReg.SetStats(0)
    histoRecoGen2_krReg.SetMarkerSize(1)
#    histoRecoGen2_krReg.Draw("colz text")
    histoRecoGen2_krReg.Draw("colz")
    c.SaveAs(outFolder+"krReg_resp.pdf")
    c.Clear()
    histoRecoGen2_lkr.Scale(1./histoRecoGen2_lkr.Integral())
    ROOT.gStyle.SetPaintTextFormat("1.2f");
    histoRecoGen2_lkr.SetStats(0)
    histoRecoGen2_lkr.SetMarkerSize(1)
#    histoRecoGen2_lkr.Draw("colz text")
    histoRecoGen2_lkr.Draw("colz")
    c.SaveAs(outFolder+"lkr_resp.pdf")
    c.Clear()
# HYBRID
#    histoRecoGen2_newHybrid.Scale(1./histoRecoGen2_newHybrid.Integral())
    # histoRecoGen2_lkr.Scale(100.)
#    ROOT.gStyle.SetPaintTextFormat("1.2f");
    # ROOT.gStyle.SetPaintTextFormat("1.0f");
#    histoRecoGen2_newHybrid.SetStats(0)
#    histoRecoGen2_newHybrid.SetMarkerSize(1)
#    histoRecoGen2_newHybrid.Draw("colz text")
#    c.SaveAs(outFolder+"newHybrid_resp.pdf")
#    c.Clear()

# Q? RMS and mean based on histograms if you cut data you must know it
    rms_reg, mean_reg, respRMS_reg = doRMSandMean(histo, "reg", outFolder)
    rms_kr, mean_kr, respRMS_kr = doRMSandMean(histo_kr, "kr", outFolder)
    rms_krReg, mean_krReg, respRMS_krReg = doRMSandMean(histo_krReg, "krReg", outFolder)
    rms_lkr, mean_lkr, respRMS_lkr = doRMSandMean(histo_lkr, "lkr", outFolder)
#    rms_newHybrid, mean_newHybrid, respRMS_newHybrid = doRMSandMean(histo_newHybrid, "newHybrid", outFolder)
#    rms_newHybrid2, mean_newHybrid2, respRMS_newHybrid2 = doRMSandMean(histo_newHybrid2, "newHybrid2", outFolder)


#GC new plot for all the means together
    style.style1d()
    s = style.style1d()
    c.SetLogx(False)
    c.SetLogy(False)
    c.Clear()
    def SetStatsColor(h,color):
        h.SetStats(0)
        h.SetLineColorAlpha(color, 0.6)
        h.SetFillColorAlpha(color, 0.6)
        return h
    mean_reg 	= SetStatsColor(mean_reg, 0)
    mean_kr 	= SetStatsColor(mean_kr, 1)
    mean_lkr 	= SetStatsColor(mean_lkr, 2)
    mean_krReg 	= SetStatsColor(mean_krReg, 3)
    
    mean_reg.Draw("histe")
    mean_kr.Draw("same")
    mean_lkr.Draw("same")
    mean_krReg.Draw("same")
    l = ROOT.TLegend(0.7,0.5,0.9,0.7)
    l.AddEntry(mean_reg, "DNN")
    l.AddEntry(mean_kr, "KR")
    l.AddEntry(mean_lkr, "LKR")
    l.AddEntry(mean_krReg, "KR+DNN(noKR)")
    c.SaveAs(outFolder+"allMeans.pdf")
    

    
    
    purity_kr, stability_kr, eff_kr = doPSE(hResp_kr, histTrue, "kr", outFolder)
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

    c.SaveAs(outFolder+"reg_pse.pdf")

    outRootFile = ROOT.TFile(outFolder+"/AllPlots.root","RECREATE")
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
    outRootFile.Close()'''
