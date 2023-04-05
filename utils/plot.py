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
import shap
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



def doEvaluationPlots(yTest, yPredicted, weightTest, lkrM, krM, outFolder, totGen, write):
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
    if (write):
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
    fig.subplots_adjust(hspace=0.05)
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

    ax2.plot(x_+dx_     , np.sqrt(np.diag(dnnCov))/trueCounts      ,   label='Unfolded DNN', linestyle='-', marker='o', markersize = 6, color='C0')
    ax2.plot(x_+dx_ , np.sqrt(np.diag(looseCov))/(trueCounts), label='Unfolded Loose Err', linestyle='-', marker='o', markersize = 6, color='C1')
    ax2.plot(x_+dx_ , np.sqrt(np.diag(kinCov))/trueCounts     , label='Unfolded kin', linestyle='-', marker='o', markersize = 6, color='C2' )
    ax2.set_xlabel("m$_{tt}$ (GeV/c$^2$)", fontsize=18)
    ax2.tick_params(labelsize=18)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.4)
    #for i in range(len(x_)):
    #    ax2.text(x_[i]+dx_[i], np.sqrt(np.diag(dnnCov)[i])/trueCounts[i], "%d%%"%(np.sqrt(np.diag(dnnCov)[i])*100/trueCounts[i]), ha='center', va='bottom')
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
    if (write):
        with open(outFolder+"/../model/Info.txt", "a+") as f:
            print("Difference in: A*x - y"    , dnnMatrix.dot(trueCounts)-dnnCounts, file=f)
            print("Difference in: A-1*y - x"  , np.linalg.inv(dnnMatrix).dot(dnnCounts)  - trueCounts, file=f)
            print("Errors in DNN unfolding:"  , np.sqrt(np.diag(dnnCov)), file = f)
            print("Errors in Loose unfolding:", np.sqrt(np.diag(looseCov)), file=f)
            print("Errors in kin unfolding:"  , np.sqrt(np.diag(kinCov)), file=f)
#except:
  #      print("singular matrix")



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
    plt.cla()


# *******************************
# *                             *
# *           SHAP              *
# *                             *
# *******************************
def doPlotShap(featureNames, model, inX_test, outFolder):
    max_display = inX_test.shape[1]
    for explainer, name  in [(shap.GradientExplainer(model,inX_test[:1000]),"GradientExplainer"),]:
        shap.initjs()
        print("... {0}: explainer.shap_values(X)".format(name))
        shap_values = explainer.shap_values(inX_test[:1000])
        print("... shap.summary_plot")
        plt.close('all')
        plt.figure()
        shap.summary_plot(	shap_values, inX_test[:1000], plot_type="bar",
			            feature_names=featureNames,
            			max_display=max_display,
                        plot_size=[15.0,0.4*max_display+1.5],
                        show=False)
        plt.savefig(outFolder+"/model/"+"shap_summary_{0}.pdf".format(name))
