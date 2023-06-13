import ROOT
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
import seaborn as sns
import numpy as np
from scipy.stats import moment
from mpl_toolkits.axes_grid1 import make_axes_locatable
import shap
import matplotlib.patches as mpatches
from scipy.stats.stats import pearsonr
import pandas as pd
import matplotlib.ticker as ticker
ROOT.gROOT.SetBatch(True)



logbins =  np.concatenate((np.linspace(320, 500, 16, endpoint=False), [500, 520, 540, 560, 580, 600, 650, 750])) #]
logbins = np.array([380, 470, 620, 820,  1100, 1500])
recoBin = np.array([1,   380, 470, 620, 820,  1100, 1500, 250000]) #
diffBin = np.arange(300, 2050, 50) #1200
#recoBinPlusBin = np.array([1, 380, 450, 500, 625, 812.5,  1200, 5000])
nRecoBin = len(recoBin)-1

def getRecoBin():
    return recoBin.copy()

def getDiffBin(extrabins=False):
    if (extrabins):
        diffBin[0] = 1
        diffBin[len(diffBin)-1] = 5000
    return diffBin

#def FillHisto(mtrue, mpred, histos):
#    for i in range(0,4):
#        if (mtrue>recoBin[i] and mtrue<recoBin[i+1]):
#            histos[i].Fill(mpred-mtrue)


def linearCorrelation(yPredicted, lkrM, krM, totGen, outFolder,  weights):
    if not os.path.exists(outFolder):
        os.makedirs(outFolder)
    index = 0    

    myMasks = [(yPredicted>-998), (lkrM>-998), (krM>-998)]  
    fig, ax = plt.subplots(1, 3, figsize=(24, 8), sharey=True)
    fig.subplots_adjust(wspace=0)
    #vmin = min(np.min(hist1), np.min(hist2), np.min(hist3))
    #vmax = max(np.max(hist1), np.max(hist2), np.max(hist3))
    hist1, xedges, yedges = np.histogram2d(totGen[myMasks[0]], yPredicted[myMasks[0]], bins=(80, 80), range=[[340, 1999], [340, 1999]],    weights=weights[myMasks[0]])
    hist2, xedges, yedges = np.histogram2d(totGen[myMasks[1]], lkrM[myMasks[1]],       bins=(80, 80), range=[[340, 1999], [340, 1999]],    weights=weights[myMasks[1]])
    hist3, xedges, yedges = np.histogram2d(totGen[myMasks[2]], krM[myMasks[2]],        bins=(80, 80), range=[[340, 1999], [340, 1999]],    weights=weights[myMasks[2]])

    vmin = 1#min(np.min(hist1), np.min(hist2), np.min(hist3))
    vmax = max(np.max(hist1), np.max(hist2), np.max(hist3))
        #cbar_ax = fig.add_axes([0.81, 0.11, 0.05, 0.77])
        #cbar = fig.colorbar(im, cax=cbar_ax)
    ax[0].hist2d(totGen[myMasks[0]], yPredicted[myMasks[0]], bins=(80, 80), range=[[340, 1999], [340, 1999]] ,norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=plt.cm.jet, cmin=1, weights=weights[myMasks[0]])
    ax[1].hist2d(totGen[myMasks[1]], lkrM[myMasks[1]],       bins=(80, 80), range=[[340, 1999], [340, 1999]] ,norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=plt.cm.jet, cmin=1, weights=weights[myMasks[1]])
    ax[2].hist2d(totGen[myMasks[2]], krM[myMasks[2]],        bins=(80, 80), range=[[340, 1999], [340, 1999]] ,norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=plt.cm.jet, cmin=1, weights=weights[myMasks[2]])
    corr = []
    
    #corr.append(pearsonr(totGen[myMasks[0]], yPredicted)[0])
    #corr.append(pearsonr(totGen[myMasks[1]], lkrM[lkrM>-998])[0])
    #corr.append(pearsonr(totGen[myMasks[2]],  krM[krM>-998])[0])

    
    for i in range(len(myMasks)):
        x = totGen[myMasks[i]]
        y = ([yPredicted, lkrM, krM][i])[myMasks[i]]
        mx = np.average(x, weights = weights[myMasks[i]])
        my = np.average(y, weights = weights[myMasks[i]])
        num = np.sum(weights[myMasks[i]] * (x - mx) * (y - my))
        den = np.sqrt(np.sum(weights[myMasks[i]] * (x - mx)**2) * np.sum(weights[myMasks[i]] * (y - my)**2))
        corr.append(num/den)

    ax[0].set_ylabel(r"m$_{{\mathrm{{t\bar{{t}}}}}}^\mathrm{{Reg}} $ [GeV]", fontsize=18)
    for i in range(3):        
        ax[i].set_xlabel(r"m$_{{\mathrm{{t\bar{{t}}}}}}^\mathrm{{True}} $ [GeV] ", fontsize=18)
        ax[i].set_title(['Regression Neural Network', 'Loose Kinematic Reconstruction', 'Full Kinematic Reconstruction'][i], fontsize=22)
        ax[i].tick_params(labelsize=18)
        ax[i].add_patch(plt.Rectangle((0.725, 0.02), 0.25, 0.1, facecolor='white',  linewidth=2, edgecolor='black', alpha=0.7, transform=ax[i].transAxes))
        ax[i].text(0.75, 0.06, r'$\rho$ = %.3f'%(corr[i]), transform=ax[i].transAxes, fontsize=18, fontweight='bold')
        
        index = index +1
    
    fig.savefig(outFolder+"/"+"CorrelationRegVsTrue.pdf", bbox_inches='tight')
    plt.cla()   
    plt.close('all')
# *******************************
# *                             *
# *     Invariant Mass Plot     *
# *                             *
# ******************************* 



def invariantMass(yPredicted, lkrM, krM, totGen, mask, outFolder, weights):
        myMasks = [(yPredicted>-998), (lkrM>-998), (krM>-998)]  
        print("Invariant Mass plot...")
        yPredicted = yPredicted[myMasks[0]]
        print("******************************\n\n*******************************\n\n", yPredicted.max(), yPredicted.min())
        lkrM = lkrM[myMasks[1]]
        krM  = krM[myMasks[2]]
        f, ax = plt.subplots(1, 1,  figsize = (7,7))
        if (totGen is not None):
            ax.hist(yPredicted, bins = 80, range=(200,1500), label="DNN   R = %.3f"%(weights[myMasks[0]].sum()/weights[mask].sum()),  histtype=u'step', alpha=0.8, linewidth = 2., density=True, weights=weights[myMasks[0]]) 
            ax.hist(lkrM,       bins = 80, range=(200,1500), label="Loose R = %.3f"%(weights[myMasks[1]].sum()/weights[mask].sum()),        histtype=u'step', alpha=0.8, linewidth = 2., density=True, weights=weights[myMasks[1]])
            ax.hist(krM,        bins = 80, range=(200,1500), label="Full  R = %.3f"%(weights[myMasks[2]].sum()//weights[mask].sum()),         histtype=u'step', alpha=0.8, linewidth = 2., density=True, weights=weights[myMasks[2]])
            ax.hist(totGen[mask],     bins = 80, range=(200,1500), label="True  I = %dk"%(len(totGen)*0.001),                                                       histtype=u'step', alpha=0.8, linewidth = 2., density=True, weights=weights[mask])
        else:
            print("Error")
            return 0
            #ax.hist(yPredicted,   bins = 80, range=(200,1500), label="DNN   I = %d"%(len(yPredicted)),    histtype=u'step', alpha=0.8, density=True) 
            #ax.hist(lkrM,         bins = 80, range=(200,1500), label="Loose I = %d"%(len(lkrM)),  histtype=u'step', alpha=0.5, density=True)
            #ax.hist(krM,          bins = 80, range=(200,1500), label="Full  I = %d"%(len(krM)),histtype=u'step', alpha=0.5, density=True)
            
        ax.set_xlim(ax.get_xlim()[0], 1500)
        ax.legend(loc = "best", fontsize=18)
        ax.tick_params(labelsize=20)
        ax.set_ylabel('Normalized Counts', fontsize=18)
        ax.set_xlabel(r"m$_{{\mathrm{{t\bar{{t}}}}}} $ [GeV]" , fontsize=18 )
        f.savefig(outFolder+"/m_tt.pdf", bbox_inches='tight')
        plt.close()
        print("Invariant Mass plot saved in"+ outFolder+"/m_tt.pdf")

    

def diffCrossSec(yPredicted, lkrM, krM, totGen, dnnMatrix, looseMatrix, kinMatrix, outFolder,  weights, write, recoBin=getRecoBin() ):
        '''
        yPrediceted = np.array of prediction of DNN. length equal to the number of events passing all the cuts. if events pass criteria but either loose or kin do not exist not included (the NN does not learn fromt these events)
        (l)krM      = np.array of solutions of Loose (kin) reconstruction. If events passed criteria but solutions do not exist they are -999
        totGen      = total list of generated masses even when events do not pass basic selection criteria
        dnnMatrix, looseMatrix, kinMatrix are reponse matrices
        outFolder   = folder where to save plots
        write       = decide if writing in txt file some output information
        recoBin     = bins of unfolding
        '''

        myMasks = [yPredicted>-998, lkrM>-998, krM>-998]

        dnnCov      = covarianceMatrix(yPredicted[myMasks[0]], dnnMatrix, weights = weights[myMasks[0]])
        looseCov    = covarianceMatrix(lkrM[myMasks[1]], looseMatrix, weights = weights[myMasks[1]])
        kinCov      = covarianceMatrix(krM[myMasks[2]], kinMatrix, weights = weights[myMasks[2]])
        np.save(outFolder+'/../model/dnnCov.npy', dnnCov)
        np.save(outFolder+'/../model/looseCov.npy', looseCov)
        np.save(outFolder+'/../model/kinCov.npy', kinCov)

        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(14, 10))
        fig.subplots_adjust(hspace=0.00)
        fig.align_ylabels([ax1,ax2])
        

#        assert ((yPredicted[myMasks[0]] > 1)).all(), "yPredicted out of range"




        dnnCounts, b_ = np.histogram(yPredicted[myMasks[0]] , bins=getRecoBin(), weights=weights[myMasks[0]], density=False)
        looseCounts,b_= np.histogram(lkrM[myMasks[1]]       , bins=getRecoBin(), weights=weights[myMasks[1]], density=False)
        kinCounts, b_ = np.histogram(krM[myMasks[2]]        , bins=getRecoBin(), weights=weights[myMasks[2]], density=False)
        #genCounts, bins_gen, bars_gen = ax1.hist(yTrue,      bins=getRecoBin(), density=False, histtype=u'step', alpha=0.5, label='Gen mtt Reco', edgecolor='C3')
        if (totGen is not None):
            #trueCounts, binsTrue, barsTrue = ax1.hist(totGen,         bins=recoBin, density=False, histtype=u'step', alpha=0.5, label='True', edgecolor='black')
            trueCounts, b_ = np.histogram(totGen,   bins=getRecoBin(), density=False, weights=weights)
            ax1.hist(recoBin[:-1], bins=recoBin, weights = trueCounts/np.diff(recoBin), histtype=u'step', alpha=0.5, label='True', edgecolor='black')
        # I did the unfolding with first bin equal to 1 and last bin equal to 5000
        # To visualize it I put x[0] = 300 and x[-1] = 2500
        x = getRecoBin()
        x[0]  = 300
        x[-1] = 2500
        assert x[0]!=recoBin[0]
        ax1.set_xlim(x[0], x[-1])
        ax2.set_xlim(x[0], x[-1])

        dx= (x[1:]-x[:-1])/2
        dnnUnfolded   = np.linalg.inv(dnnMatrix).dot(dnnCounts)
        looseUnfolded = np.linalg.inv(looseMatrix).dot(looseCounts) 
        kinUnfolded   = np.linalg.inv(kinMatrix).dot(kinCounts) 
        ax1.errorbar(x[:-1]+dx     , dnnUnfolded/np.diff(recoBin)    , yerr=np.sqrt(np.diag(dnnCov))/np.diff(recoBin)    , label='Unfolded DNN',   linestyle=' ', marker='o', markersize = 6, color='C0')
        ax1.errorbar(x[:-1]+4/3*dx , looseUnfolded/np.diff(recoBin)  , yerr=np.sqrt(np.diag(looseCov))/np.diff(recoBin)  , label='Unfolded Loose', linestyle=' ', marker='o', markersize = 6, color='C1')
        ax1.errorbar(x[:-1]+2/3*dx , kinUnfolded/np.diff(recoBin)    , yerr=np.sqrt(np.diag(kinCov))/np.diff(recoBin)    , label='Unfolded kin',   linestyle=' ', marker='o', markersize = 6, color='C2')
        #ax1.set_xscale('log')
        #ax2.set_xscale('log')
        ax1.set_ylabel(r"$\mathrm{dN_{Events}/dm_{{t\overline{{t}}}}}$ [GeV$^{-1}$] ", fontsize=22)
        ax2.set_ylabel("Rel. Uncertainty", fontsize=22)
        ax2.set_xlabel(r"m$_\mathrm{{t\bar{{t}}}}$ [GeV]", fontsize=22)
        ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax1.yaxis.offsetText.set_fontsize(18)
        #ax1.set_ylabel("Counts per %(first)d or %(second)d GeV/c" %{"first":bins[1]-bins[0], "second":(bins[len(bins)-1]-bins[len(bins)-2])} , fontsize=18)
        ax1.tick_params(labelsize=22)
        ax1.legend(fontsize=22)

        ax2.plot(x[:-1]+dx , np.sqrt(np.diag(dnnCov))/dnnUnfolded      ,   label='Unfolded DNN', linestyle='-', marker='o', markersize = 6, color='C0')
        ax2.plot(x[:-1]+dx , np.sqrt(np.diag(looseCov))/(looseUnfolded), label='Unfolded Loose ', linestyle='-', marker='o', markersize = 6, color='C1')
        ax2.plot(x[:-1]+dx , np.sqrt(np.diag(kinCov))/kinUnfolded     , label='Unfolded kin', linestyle='-', marker='o', markersize = 6, color='C2' )
        ax2.tick_params(axis = 'y', labelsize=18)
        ax2.tick_params(axis = 'x', labelsize=22)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.4)
        
        axins = ax2.inset_axes([0.06, 0.5, 0.3, 0.3])
        axins.plot(x[:-1]+dx , np.sqrt(np.diag(dnnCov))/dnnUnfolded      ,   label='Unfolded DNN', linestyle='-', marker='o', markersize = 6, color='C0')
        axins.plot(x[:-1]+dx , np.sqrt(np.diag(looseCov))/(looseUnfolded), label='Unfolded Loose ', linestyle='-', marker='o', markersize = 6, color='C1')
        axins.plot(x[:-1]+dx , np.sqrt(np.diag(kinCov))/kinUnfolded     , label='Unfolded kin', linestyle='-', marker='o', markersize = 6, color='C2' )
        axins.yaxis.tick_left()
        axins.set_xticklabels([])
        x1, x2, y1, y2 = 330, 750, 0, 0.004 
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        
        ax2.indicate_inset_zoom(axins, edgecolor="black")
        fig.savefig(outFolder+"/diff_tt.pdf", bbox_inches='tight')
        plt.cla()
        np.set_printoptions(precision=3)
        if (totGen is not None):
            print("sum of true counts", np.array(trueCounts).sum())
        print("sum of dnn counts", dnnUnfolded.sum())
        print("sum of kin counts", kinUnfolded.sum())
        
        if (totGen is not None):
            print("Difference in: A*x - y" ,  dnnMatrix.dot(trueCounts)-dnnCounts)
            print("Difference in: A-1*y - x" ,np.linalg.inv(dnnMatrix).dot(dnnCounts)  - trueCounts)
            print("Determinant Cov Matrix    :", np.linalg.det(dnnCov))
            print("True Counts               :", trueCounts, weights.sum())
            print("dnnMatrix                 :", dnnMatrix)
            print("Fiducial weighted events  :", dnnCounts, weights[myMasks[0]].sum())
            print("number of generated events:", len(totGen))
            print("Errors in DNN unfolding   :",   np.sqrt(np.diag(dnnCov)))
            print("Errors in Loose unfolding :", np.sqrt(np.diag(looseCov)))
            print("Errors in kin unfolding   :",   np.sqrt(np.diag(kinCov)))
            if (write):
                with open(outFolder+"/../model/Info.txt", "a+") as f:
                    print("Difference in: A*x - y"    , dnnMatrix.dot(trueCounts)-dnnCounts, file=f)
                    print("Difference in: A-1*y - x"  , np.linalg.inv(dnnMatrix).dot(dnnCounts)  - trueCounts, file=f)
                    print("Determinant Cov Matrix    :", np.linalg.det(dnnCov)      , file=f)
                    print("Errors in DNN unfolding   :", np.sqrt(np.diag(dnnCov))   , file = f)
                    print("Errors in Loose unfolding :", np.sqrt(np.diag(looseCov)) , file=f)
                    print("Errors in kin unfolding   :", np.sqrt(np.diag(kinCov))   , file=f)
                    print("Counts in dnnUnfolded     :", dnnUnfolded/np.diff(recoBin), file=f )
                    print("Sum of DNNunfolded events:" , dnnUnfolded.sum(), file=f )
                    print()
        plt.close('all')

def ResponseMatrix(matrixName, y, outFolder, totGen, weights, recoBin = getRecoBin()):
    '''create a response matrix with an additional bin used to count overflow event'''
    nRecoBin = len(recoBin)-1
    matrix = np.empty((nRecoBin, nRecoBin))
    for i in range(nRecoBin):
        maskGen = (totGen >= recoBin[i]) & (totGen < recoBin[i+1])
       
        #maskNorm = (totGen >= recoBin[i]) & (totGen < recoBin[i+1])
        normalization = np.sum(weights[maskGen])
        for j in range(nRecoBin):
            #print(f"Bin generated #{i}\tBin reconstructed #{j}\r", end="")
            maskRecoGen = (y >= recoBin[j]) & (y < recoBin[j+1]) & maskGen
            entries = np.sum(weights[maskRecoGen])
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
    ax.xaxis.set_ticks(np.arange(0, nRecoBin, 1.0))
    ax.yaxis.set_ticks(np.arange(0, nRecoBin, 1.0))
    for y in range(nRecoBin):
        for x in range(nRecoBin):
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


def covarianceMatrix(y, matrix,  weights):
    counts_reco, bins_reco = np.histogram(y, bins=getRecoBin(), density=False, weights = weights)
    dy = np.diag(counts_reco)
    try:
        A_inv = np.linalg.inv(matrix)
        cov = (A_inv.dot(dy)).dot(A_inv.T)
        return cov
    except:
        print("Singular Matrix\n\n\n\n\nCheck the matrix")



def doEvaluationPlots(yTrue, yPredicted, lkrM, krM, outFolder, totGen, mask_test, weights, dnn2Mask, write):
    if not os.path.exists(outFolder):
        os.makedirs(outFolder)
    
    yTrue = yTrue[:,0]
    
    #assert len(yPredicted)==len(totGen[mask_test]), "yPredicted length does not match totGen[mask_test]"

    invariantMass(yPredicted=yPredicted, lkrM=lkrM, krM=krM, totGen=totGen, mask = mask_test ,outFolder=outFolder, weights=weights)


# *******************************
# *                             *
# * Purity Stability Efficinecy *
# *                             *
# *******************************
    fig, ax = plt.subplots(1, 1)
    '''print("Purity Stability Efficiency") 
    GenBinith = np.array([])
    RecoBinith = np.array([])
    GenRecoBinith = np.array([])
    x = getRecoBin()
    for i in range(len(x)-1):
        maskTotGen = np.array((totGen >= x[i]) & (totGen<x[i+1]))           # all the generated
        maskGen = np.array((yTrue >= x[i]) & (yTrue<x[i+1]))                # all the generated that satisfy the cuts
        maskReco = np.array((yPredicted >= x[i]) & (yPredicted<x[i+1]))     # all the reconstructed in bin
        maskGenReco = maskGen & maskReco
        GenBinith = np.append(GenBinith, len(maskTotGen[maskTotGen==True]))                              # Generated in bin i-th
        RecoBinith = np.append(RecoBinith, len(maskReco[maskReco==True]))                           # Reconstructed in bin ith
        GenRecoBinith = np.append(GenRecoBinith,  len(maskGenReco[maskGenReco==True]))                        # Generated and Reconstructed in bin ith

    
    
    dx_ = (x[1:] - x[:len(x)-1])/2
    x_ = x[:len(x)-1] + dx_

    ax.errorbar(x_, GenRecoBinith/GenBinith, None, dx_, linestyle='None', label='Purity')
    ax.errorbar(x_, GenRecoBinith/RecoBinith, None, dx_, linestyle='None', label='Stability')
    ax.set_xlabel("m$_{tt}^{True}$")
    ax.legend(fontsize=18, loc='best',bbox_to_anchor=(1, 0.5))
    fig.savefig(outFolder+"/pse.pdf", bbox_inches='tight')
    plt.cla()'''



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

    logbins_wu = np.concatenate(([1], logbins, [25000])) 
    #consider all the events below 340 in one bin.
# for loose and kin masks to be applied to totgen
    for i in range(len(logbins_wu)-1):
        dnnMask = (mask_test) & (totGen >= logbins_wu[i]) & (totGen < logbins_wu[i+1])
        kinMask =  (krM>-998) & (totGen >= logbins_wu[i]) & (totGen < logbins_wu[i+1])
        LooseMask =  (lkrM>-998) & (totGen >= logbins_wu[i]) & (totGen < logbins_wu[i+1])
        

        

        regMeanBinned       = np.append( regMeanBinned      ,   np.average(yPredicted[dnnMask]-totGen[dnnMask], weights = weights[dnnMask]))
        LooseMeanBinned     = np.append( LooseMeanBinned    ,   np.average(lkrM[LooseMask]-totGen[LooseMask], weights = weights[LooseMask]))
        kinMeanBinned       = np.append( kinMeanBinned      ,   np.average(krM[kinMask]-totGen[kinMask], weights = weights[kinMask]))

        regSquaredBinned       = np.append( regSquaredBinned      ,   np.sqrt(np.average((yPredicted[dnnMask]-totGen[dnnMask])**2, weights = weights[dnnMask])))
        LooseSquaredBinned     = np.append( LooseSquaredBinned    ,   np.sqrt(np.average((lkrM[LooseMask]-totGen[LooseMask])**2, weights = weights[LooseMask])))
        kinSquaredBinned       = np.append( kinSquaredBinned      ,   np.sqrt(np.average((krM[kinMask]-totGen[kinMask])**2, weights = weights[kinMask])))

        regRmsBinned        = np.append( regRmsBinned       ,   np.sqrt(np.average((yPredicted[dnnMask]-totGen[dnnMask] - np.average(yPredicted[dnnMask]-totGen[dnnMask], weights = weights[dnnMask]))**2, weights = weights[dnnMask])))
        LooseRmsBinned      = np.append( LooseRmsBinned     ,   np.sqrt(np.average((lkrM[LooseMask]-totGen[LooseMask] - np.average(lkrM[LooseMask]-totGen[LooseMask], weights = weights[LooseMask]))**2, weights = weights[LooseMask])))
        kinRmsBinned        = np.append( kinRmsBinned       ,   np.sqrt(np.average((krM[kinMask]-totGen[kinMask] - np.average(krM[kinMask]-totGen[kinMask], weights = weights[kinMask]))**2, weights = weights[kinMask])))
        
        regElementsPerBin      = np.append( regElementsPerBin   ,  weights[dnnMask].sum())
        LooseElementsPerBin    = np.append( LooseElementsPerBin ,  weights[kinMask].sum())
        kinElementsPerBin      = np.append( kinElementsPerBin   ,  weights[LooseMask].sum())

        '''regErrRmsBinned     = np.append( regErrRmsBinned    ,   np.sqrt((moment(yPredicted[dnnMask]-totGen[dnnMask], 4)-((regElementsPerBin[i]-3)/(regElementsPerBin[i]-1)*regRmsBinned**4)[0])/regElementsPerBin[i]))
        LooseErrRmsBinned   = np.append( LooseErrRmsBinned  ,   np.sqrt((moment(lkrM[LooseMask]-totGen[LooseMask], 4)-((LooseElementsPerBin[i]-3)/(LooseElementsPerBin[i]-1)*LooseRmsBinned**4)[0])/LooseElementsPerBin[i]))
        kinErrRmsBinned     = np.append( kinErrRmsBinned    ,   np.sqrt((moment(krM[kinMask]-totGen[kinMask], 4)-((kinElementsPerBin[i]-3)/(kinElementsPerBin[i]-1)*kinRmsBinned**4)[0])/kinElementsPerBin[i]))'''


    logbins_wu[0] = 280 # for visualization purpose the underflow is between 320, 340
    logbins_wu[-1] = 1700

    errX = (logbins_wu[1:]-logbins_wu[:len(logbins_wu)-1])/2
    x = logbins_wu[:len(logbins_wu)-1] + errX
    ax.errorbar(x, regMeanBinned,  regRmsBinned/np.sqrt(regElementsPerBin), errX, linestyle='None', label = 'DNN')
    ax.errorbar(x, LooseMeanBinned, LooseRmsBinned/np.sqrt(LooseElementsPerBin), errX, linestyle='None', label = 'Loose')
    ax.errorbar(x, kinMeanBinned,  kinRmsBinned/np.sqrt(kinElementsPerBin), errX, linestyle='None', label = 'Kin')
    ax.set_xlabel("$m_{tt}^{True}$ (GeV)")
    ax.set_xlim(logbins_wu[0], logbins_wu[-1])
    ax.hlines(y=0, xmin=300, xmax=logbins_wu[-1], linestyles='dotted', color='red')
    ax.set_ylabel("Mean(Pred - True) [GeV]")
    ax.legend(fontsize=18)
    fig.savefig(outFolder+"/allMeans.pdf", bbox_inches='tight')
    plt.cla()

    #fig, ax = plt.subplots(1, 1)
    ax.errorbar(x, regRmsBinned,   xerr=errX, linestyle='None', label = 'DNN err/100')
    ax.errorbar(x, LooseRmsBinned, xerr=errX, linestyle='None', label = 'Loose err/100')
    ax.errorbar(x, kinRmsBinned,   xerr=errX, linestyle='None', label = 'Kin err/1000')
    ax.legend(fontsize=18, loc='best')
    ax.set_ylabel("RMS(Pred-True) [GeV]", fontsize=18)
    ax.set_yscale('log')
    ax.set_xlim(logbins_wu[0], logbins_wu[-1])
    fig.savefig(outFolder+"/allRms.pdf", bbox_inches='tight')
    plt.cla()

    
    ax.errorbar(x, regSquaredBinned,    None, errX, linestyle='None', label = 'DNN ')
    ax.errorbar(x, LooseSquaredBinned,  None, errX, linestyle='None', label = 'Loose ')
    ax.errorbar(x, kinSquaredBinned,    None, errX, linestyle='None', label = 'Kin ')
    ax.legend(fontsize=18, loc='best')
    ax.set_xlabel("m$_{tt}^\mathrm{{True}}$ [GeV]", fontsize=16)
    ax.set_xlim(logbins_wu[0], logbins_wu[-1])
    ax.set_yscale('log')
    ax.tick_params(labelsize=18)
    ax.set_ylabel("$\sqrt{<(m_{\mathrm{{t\overline{t}}}}^\mathrm{{Reg}}-m_{\mathrm{{t\overline{t}}}}^\mathrm{{True}})^2>}$ [GeV]", fontsize=16)
    fig.savefig(outFolder+"/allSquared.pdf", bbox_inches='tight')
    plt.cla()
    
# *******************************
# *                             *
# *      Response Matrix        *
# *                             *
# ******************************* 
    dnnMatrix = np.empty((nRecoBin, nRecoBin))
    print("DNN matrix")
    dnnMatrix = ResponseMatrix(y=yPredicted, matrixName = 'dnnMatrix', outFolder = outFolder, totGen=totGen, weights=weights)
    print("\nCondition Number of the DNN matrix:", np.linalg.cond(dnnMatrix))
    if (write):
        with open(outFolder+"/../model/Info.txt", "a+") as f:
            print("\nCondition Number of the DNN matrix:  "+str(np.linalg.cond(dnnMatrix))+"\n", file=f)
    #print(np.linalg.inv(dnnMatrix).dot(dnnMatrix))
    looseMatrix = np.empty((nRecoBin, nRecoBin))
    looseMatrix = ResponseMatrix(y=lkrM, matrixName = 'looseMatrix', outFolder = outFolder, totGen=totGen, weights=weights)

    kinMatrix = np.empty((nRecoBin, nRecoBin))
    kinMatrix = ResponseMatrix(y=krM, matrixName = 'kinMatrix', outFolder = outFolder, totGen=totGen, weights=weights)
    np.save(outFolder+'/../model/dnnM.npy', dnnMatrix)
    np.save(outFolder+'/../model/looseM.npy', looseMatrix)
    np.save(outFolder+'/../model/kinM.npy', kinMatrix)


# *******************************
# *                             *
# *      Diff cross section     *
# *                             *
# ******************************* 
    
# !!! aggiungere un boolean che ti dice se e andato a buon fine e in caso contrario non stampare nessuna matrice
    
    diffCrossSec(yPredicted=yPredicted, lkrM=lkrM, krM=krM, recoBin=getRecoBin(), totGen=totGen, dnnMatrix=dnnMatrix, looseMatrix=looseMatrix, kinMatrix=kinMatrix, outFolder=outFolder, weights=weights, write=write )

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
    



# *******************************
# *                             *
# *     2D Resolution plots     *
# *                             *
# *******************************

    index = 0    
    myMasks = [(yPredicted>-998), (lkrM>-998), (krM>-998)]  
    

    hist1, xedges, yedges = np.histogram2d(totGen[myMasks[0]], yPredicted[myMasks[0]],  bins=(80, 80), range=[[340, 1999], [-500, 500]], weights=weights[myMasks[0]])
    hist2, xedges, yedges = np.histogram2d(totGen[myMasks[1]], lkrM[myMasks[1]],        bins=(80, 80), range=[[340, 1999], [-500, 500]], weights=weights[myMasks[1]])
    hist3, xedges, yedges = np.histogram2d(totGen[myMasks[2]], krM[myMasks[2]],         bins=(80, 80), range=[[340, 1999], [-500, 500]], weights=weights[myMasks[2]])

    vmin = 1#min(np.min(hist1), np.min(hist2), np.min(hist3))
    vmax = max(np.max(hist1), np.max(hist2), np.max(hist3))


    
    for el in [yPredicted[myMasks[0]], lkrM[lkrM>-998], krM[krM>-998]]:
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        fig.subplots_adjust(right=0.8)
          
        counts, xedges, yedges, im = ax.hist2d(totGen[myMasks[index]], el - totGen[myMasks[index]], bins=(80, 80), range=[[340, 1500], [-500, 500]] ,norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=plt.cm.jet, weights=weights[myMasks[index]])
        cbar_ax = fig.add_axes([0.81, 0.11, 0.05, 0.77])
        cbar = fig.colorbar(im, cax=cbar_ax)
        ax.set_xlabel("m$_{tt}^{True}$ [GeV]", fontsize=18)
        ax.set_ylabel(['m$_{tt}^{DNN}$', 'm$_{tt}^{Loose}$', 'm$_{tt}^{Kin}$'][index]+" - True [GeV]", fontsize=18)
        ax.tick_params(labelsize=18)
        cbar.set_label('Counts', fontsize=19)
        cbar.ax.tick_params(labelsize=18)
        print(outFolder+"/"+['Reg', 'Loose', 'Kin'][index]+"MinusTrueVsTrue.pdf")
        fig.savefig(outFolder+"/"+['Reg', 'Loose', 'Kin'][index]+"MinusTrueVsTrue.pdf", bbox_inches='tight')
        plt.cla()   
        index = index +1
    plt.close('all')

    ###### plot for events of the second NN
    if (yPredicted[dnn2Mask]>-998).all():
            
        hist1, xedges, yedges = np.histogram2d(totGen[dnn2Mask], yPredicted[dnn2Mask],  bins=(80, 80), range=[[340, 1999], [-500, 500]], weights=weights[dnn2Mask])
        
        vmin = 1
        vmax = np.max(hist1)
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        fig.subplots_adjust(right=0.8)  
        counts, xedges, yedges, im = ax.hist2d(totGen[dnn2Mask], yPredicted[dnn2Mask] - totGen[dnn2Mask], bins=(80, 80), range=[[340, 1500], [-500, 500]] ,norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=plt.cm.jet, weights=weights[dnn2Mask])
        cbar_ax = fig.add_axes([0.81, 0.11, 0.05, 0.77])
        cbar = fig.colorbar(im, cax=cbar_ax)
        ax.set_xlabel("m$_{tt}^{True}$ [GeV]", fontsize=18)
        ax.set_ylabel("m$_{tt}^{DNN}$ - True [GeV]", fontsize=18)
        ax.tick_params(labelsize=18)
        cbar.set_label('Counts', fontsize=19)
        cbar.ax.tick_params(labelsize=18)
        print(outFolder+"/dnn2MaskRegMinusTrueVsTrue.pdf")
        fig.savefig(outFolder+"/dnn2MaskRegMinusTrueVsTrue.pdf", bbox_inches='tight')
        plt.cla()   
        plt.close('all')

# *******************************
# *                             *
# *         Profile             *
# *                             *
# *******************************

    from scipy.stats import gaussian_kde
    
    
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    x = getRecoBin()

# to finish
        
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i < 7:
            mask = (totGen>x[i]) & (totGen<x[i+1])
            
            if i <= 2:
                bins = 200
                myrange = (-500, 500)
            elif 2<i<5:
                bins = 100
                myrange = (-500, 500)
            else:
                bins = 50
                if i == 5:
                    myrange = (-500, 500)
                elif i==6:
                    myrange = (-1000, 1000)
            df = pd.DataFrame({ 'yPredicted': yPredicted - totGen,
                                'lkrM': lkrM - totGen,
                                'krM': krM - totGen,
                                'weights': weights,
                                #'weightsLoose': weights[(myMasks[1]) & mask],
                                #'weightsKin': weights[(myMasks[2]) & mask]
                                })
            df.loc[df['yPredicted'] > myrange[1], 'yPredicted'] = myrange[1]-1
            df.loc[df['lkrM'] > myrange[1], 'lkrM'] = myrange[1]-1
            df.loc[df['krM'] > myrange[1], 'krM'] = myrange[1]-1
            df.loc[df['yPredicted'] < myrange[0], 'yPredicted'] = myrange[0] + 1
            df.loc[df['lkrM'] < myrange[0], 'lkrM'] = myrange[0] + 1
            df.loc[df['krM'] < myrange[0], 'krM'] = myrange[0] + 1
            kwargs = dict(histtype='stepfilled', alpha=0.3, density=True,  ec="k")
            #ax.set_ylim(0,None)
            
            overflowBinWidth = 0.03*myrange[1]
            bins = np.linspace(myrange[0] + overflowBinWidth, myrange[1]-overflowBinWidth, bins)
            bins = np.insert(bins, 0, myrange[0])
            bins = np.append(bins, myrange[1])
            ax.hist(df['yPredicted'][(mask & myMasks[0])], bins = bins, label="DNN "  ,  weights=weights[(mask & myMasks[0] )], **kwargs) 
            ax.hist(df['lkrM'][mask & myMasks[1]] ,        bins = bins, label="Loose" ,  weights=weights[(mask & myMasks[1] )], **kwargs)
            ax.hist(df['krM'][mask & myMasks[2]] ,         bins = bins, label="Full " ,  weights=weights[(mask & myMasks[2] )], **kwargs)

            kwargs = dict(histtype='step', alpha=1., density=True)
            ax.hist(df['yPredicted'][(mask & myMasks[0])], bins = bins, label="DNN "  ,  weights=weights[(mask & myMasks[0] )], **kwargs, ec='C0') 
            ax.hist(df['lkrM'][mask & myMasks[1]] ,        bins = bins, label="Loose" ,  weights=weights[(mask & myMasks[1] )], **kwargs, ec='C1')
            ax.hist(df['krM'][mask & myMasks[2]] ,         bins = bins, label="Full " ,  weights=weights[(mask & myMasks[2] )], **kwargs, ec='C2')
            
            #g=sns.kdeplot(data = data , weights=weights[(myMasks[0]) & mask], common_norm=False, fill=True, alpha=.5, linewidth=0, ax=ax, label='DNN', gridsize=5000)
            #sns.kdeplot(data=df[(myMasks[0]) & mask], x='yPredicted', weights='weights',   fill=True, alpha=.5, linewidth=0, ax=ax, label='DNN', gridsize=5000)
            #sns.kdeplot(data=df[(myMasks[1]) & mask],     x='lkrM',       weights='weights', fill=True, alpha=.5, linewidth=0, ax=ax, label='Full', gridsize=5000)
            #sns.kdeplot(data=df[(myMasks[2]) & mask],     x='krM',        weights='weights',   fill=True, alpha=.5, linewidth=0, ax=ax, label='Loose', gridsize=5000)

            #sns.kdeplot  (data = (krM[(myMasks[1]) & mask]        - totGen[(myMasks[1]) & mask]) , weights=weights[(myMasks[1]) & mask], common_norm=False, fill=True, alpha=.5, linewidth=0, ax=ax, label='Full', gridsize=5000)
            #sns.kdeplot  (data = (lkrM[(myMasks[2]) & (mask) ]       - totGen[(myMasks[2]) & (mask) ]) , weights=weights[(myMasks[2]) & (mask) ], common_norm=False, fill=True, alpha=.5, linewidth=0, ax=ax, label='Loose', gridsize=5000)
            
            ax.set_xlim(myrange)
            ax.set_ylim(0, ax.get_ylim()[1]*1.5)
            ax.set_xlabel(r"m$_\mathrm{t\bar{t}}$ [GeV]", fontsize= 18)
            ax.set_ylabel("Normalized Counts", fontsize= 18)
            if (i==6):
                ax.text(0, ax.get_ylim()[1]*0.9,  r"${}$ GeV $\leq \mathrm{{m}}_{{\mathrm{{t\bar{{t}}}}}}^\mathrm{{True}}$ ".format(x[i]), fontsize = 18, ha='center', va='center')
            else:
                ax.text(0, ax.get_ylim()[1]*0.9,  r"${} \leq \mathrm{{m}}_{{\mathrm{{t\bar{{t}}}}}}^\mathrm{{True}} < {} $ GeV".format(x[i], x[i+1]), fontsize = 18, ha='center', va='center')
            ax.tick_params(axis = 'x', labelsize= 18)
            ax.tick_params(axis = 'y', labelsize= 18)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            ax.yaxis.set_label_coords(-0.2, 0.5)
            ax.yaxis.offsetText.set_fontsize(18)

    #for j in range(7, len(axes)):
    axes[7].spines['top'].set_visible(False)
    axes[7].spines['bottom'].set_visible(False)
    axes[7].spines['left'].set_visible(False)
    axes[7].spines['right'].set_visible(False)
    patches = [mpatches.Patch(color='C0', alpha=0.5, label='Regression NN'), 
               mpatches.Patch(color='C1',  alpha=0.5, label='Loose Kinematic\nReconstruction'),
               mpatches.Patch(color='C2',  alpha=0.5, label='Full Kinematic\nReconstruction') ]

    axes[7].legend(handles = patches, loc='center', fontsize=20)
    axes[7].set_xticks([])
    axes[7].set_yticks([])

    plt.tight_layout()
    fig.savefig(outFolder+"/kdeResolution.pdf", bbox_inches='tight')



# *******************************
# *                             *
# *       Loss Functions        *
# *                             *
# *******************************

def doPlotLoss(fit, outName):

    # "Loss"
    plt.close('all')
    plt.figure(2)
    plt.plot(fit.history['loss'])
    plt.plot(fit.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    # plt.yscale('log')
    plt.ylim(ymax = max(min(fit.history['loss']), min(fit.history['val_loss']))*1.4, ymin = min(min(fit.history['loss']),min(fit.history['val_loss']))*0.9)
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(outName)
    plt.cla()


# *******************************
# *                             *
# *           SHAP              *
# *                             *
# *******************************
def doPlotShap(featureNames, model, inX_test, outName):

    plt.figure()
    max_display = inX_test[0].shape[1]
    explainer = shap.GradientExplainer(model, inX_test)

    # Compute Shapley values for inX_test[:1000,:]
    shap_values = explainer.shap_values(inX_test[0])


    # Generate summary plot
    shap.initjs()
    shap.summary_plot(shap_values, inX_test[0], plot_type="bar",
                    feature_names=featureNames,
                    max_display=max_display,
                    plot_size=[15.0,0.4*max_display+1.5],
                    show=False)
    plt.savefig(outName)
    '''for explainer, name  in [(shap.GradientExplainer(model,inX_test[:1000]),"GradientExplainer"),]:
        *shap.initjs()
        *print("... {0}: explainer.shap_values(X)".format(name))
        *shap_values = explainer.shap_values(inX_test[:1000])
        *print("... shap.summary_plot")
        *plt.close('all')
        *plt.figure()
        shap.summary_plot(	shap_values, inX_test[:1000], plot_type="bar",
			            feature_names=featureNames,
            			max_display=max_display,
                        plot_size=[15.0,0.4*max_display+1.5],
                        show=False)
        plt.savefig(outFolder+"/model/"+"shap_summary_{0}.pdf".format(name))'''
