import numpy as np
import ROOT
from array import array
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def openInThePhaseSpace(folder):
    print("Opening ",folder)
    kinM    = np.load(folder+"/krM.npy")
    looseM  = np.load(folder+"/lkrM.npy")
    inX     = np.load(folder+"/inX.npy")
    mask    = np.load(folder+"/mask.npy")
    weights = np.load(folder+"/weights.npy")

    inX     = inX[mask]
    looseM  = looseM[mask]
    kinM    = kinM[mask]
    weights = weights[mask]

    mll = inX[:,18]
    njets = inX[:,19]
    return njets, weights, mask, kinM, looseM, mll
def njet():
    folder = "/nfs/dust/cms/user/celottog/mttNN/npyData/15*None"
    njets, weights, mask, kinM, looseM, mll = openInThePhaseSpace(folder)
    assert len(njets)==len(kinM)
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    x = np.arange(2, 11, 1)
    kinCounts = []
    errKinCounts = []
    looseCounts = []
    errLooseCounts = []
    for i in range(2, 11):
        m = (njets==i)
        kinCounts.append(weights[(m) & (kinM>-998)].sum()/weights[(m)].sum())
        errKinCounts.append(np.sqrt(weights[(m) & (kinM>-998)].sum())/weights[(m)].sum())
        looseCounts.append(weights[(m) & (looseM>-998)].sum()/weights[(m)].sum())
        errLooseCounts.append(np.sqrt(weights[(m) & (looseM>-998)].sum())/weights[(m)].sum())
    ax.errorbar(x, kinCounts, yerr=errKinCounts, xerr=0.5, marker = 'o', color='C2', linestyle='', label='Full Kin. Reco (Eff = %.1f %%)'%(weights[kinM>-998].sum()/weights.sum()*100))
    ax.errorbar(x, looseCounts, yerr=errLooseCounts, xerr=0.5, marker= 'v', color='C1', linestyle='',  label='Loose Kin Reco (Eff = %.1f %%)'%(weights[looseM>-998].sum()/weights.sum()*100))
    ax.set_ylabel("Efficiency", fontsize=20)
    ax.set_xlabel("$\mathrm{N_{jets}}$", fontsize=20)
    
    
    
    xtick_positions = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    ax.set_xticks(xtick_positions)
    #ax.grid(True)
    ax.hlines(xmin=1.5, xmax=10.5, y=1, color='black', linestyle='--', linewidth=0.75)
    ax.set_ylim(0.85, 1.05)
    ax.set_xlim(1.5, 10.5)
    
    ax2 = ax.twinx()
    ax2.set_ylim(0.85, 1.05)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(6))
    ax.tick_params(axis='both', direction='in', length=6, width=1, colors='black')
    ax2.tick_params(axis='y', direction='in', length=6, width=1, colors='black')
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
    ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
    ax.xaxis.set_tick_params(direction='in', which='both')
    ax.yaxis.set_tick_params(direction='in', which='both')
    ax2.yaxis.set_tick_params(direction='in', which='both')
    ax2.set_yticklabels([])
    
    ax.legend(fontsize=18, loc='upper left', frameon=False)
    ax.tick_params(labelsize=18)
    ax.text(s="Private Work (CMS Simulation)", x=0.00, y=1.02, ha='left', fontsize=16,  transform=ax.transAxes ,  **{'fontname':'Arial'})
    ax.text(s="(13 TeV)", x=1.00, y=1.02, ha='right', fontsize=16,  transform=ax.transAxes,  **{'fontname':'Arial'})
    fig.savefig("/nfs/dust/cms/user/celottog/mttNN/outputs/effNjet.pdf", bbox_inches='tight')

def main():
    print("Start")
    folder = "/nfs/dust/cms/user/celottog/mttNN/npyData/15*None"
    njets, weights, mask, kinM, looseM, mll = openInThePhaseSpace(folder)
    assert len(njets)==len(kinM)
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    mlmax=500
    mll[mll>mlmax]=mlmax-1
    mllBins = np.linspace(20, mlmax, 20)
    kinCounts = []
    errKinCounts = []
    looseCounts = []
    errLooseCounts = []

    for i in range(len(mllBins)-1):
        m = ((mll>=mllBins[i]) & (mll<mllBins[i+1]))
        kinCounts.append(weights[(m) & (kinM>-998)].sum()/weights[(m)].sum())
        errKinCounts.append(np.sqrt(weights[(m) & (kinM>-998)].sum())/weights[(m)].sum())
        looseCounts.append(weights[(m) & (looseM>-998)].sum()/weights[(m)].sum())
        errLooseCounts.append(np.sqrt(weights[(m) & (looseM>-998)].sum())/weights[(m)].sum())

    print(len(kinCounts))
    print(len(mllBins[:len(mllBins)]))
    print(len(np.diff(mllBins)))
    ax.errorbar(mllBins[:len(mllBins)-1] + np.diff(mllBins)/2, kinCounts, yerr=errKinCounts, xerr=np.diff(mllBins)/2, marker = 'o', color='C2', linestyle='', label='Full Kin. Reco (Eff = %.1f %%)'%(weights[kinM>-998].sum()/weights.sum()*100))
    ax.errorbar(mllBins[:len(mllBins)-1] + np.diff(mllBins)/2, looseCounts, yerr=errLooseCounts, xerr=np.diff(mllBins)/2, marker= 'v', color='C1', linestyle='',  label='Loose Kin Reco (Eff = %.1f %%)'%(weights[looseM>-998].sum()/weights.sum()*100))
    ax.set_ylabel("Efficiency", fontsize=20)
    ax.set_xlabel(r"$\mathrm{m_{\ell\bar{\ell}}}$ [GeV]", fontsize=20)
    
    
    ax.hlines(xmin=0, xmax=mlmax, y=1, color='black', linestyle='--', linewidth=0.75)
    ax.set_ylim(0.75, 1.05)
    ax.set_xlim(20, mlmax-0.01)
    ax2 = ax.twinx()
    ax2.set_ylim(0.75, 1.05)
    
    ax.tick_params(axis='both', direction='in', length=6, width=1, colors='black')
    ax2.tick_params(axis='y', direction='in', length=6, width=1, colors='black')
    ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(n=5))

    ax.xaxis.set_tick_params(direction='in', which='both')
    ax.yaxis.set_tick_params(direction='in', which='both')
    ax2.yaxis.set_tick_params(direction='in', which='both')
    ax2.set_yticklabels([])
    
    ax.legend(fontsize=18, loc='upper left', frameon=False)
    ax.tick_params(labelsize=18)
    ax.text(s="Private Work (CMS Simulation)", x=0.00, y=1.02, ha='left', fontsize=16,  transform=ax.transAxes ,  **{'fontname':'Arial'})
    ax.text(s="(13 TeV)", x=1.00, y=1.02, ha='right', fontsize=16,  transform=ax.transAxes,  **{'fontname':'Arial'})
    fig.savefig("/nfs/dust/cms/user/celottog/mttNN/outputs/effMll.pdf", bbox_inches='tight')
    




#def root():
#    folder = "/nfs/dust/cms/user/celottog/mttNN/npyData/15*None"
#    njets, weights, mask, kinM, looseM = openInThePhaseSpace(folder)
#    assert len(njets)==len(kinM)
#    
#    print(njets.max())
#    kinVsJet  = ROOT.TH1F( "kinVsJet", "Efficiency;N_{jet}", 8, 1.999, 10.001)
#    looseVsJet  = ROOT.TH1F( "looseVsJet", "Efficiency;N_{jet}", 8, 1.999, 10.001)
#    for i in range(2, 11):
#        m = (njets==i)
#        
#        kinVsJet.SetBinContent(i-2, weights[(m) & (kinM>-998)].sum()/weights[(m)].sum())
#        looseVsJet.SetBinContent(i-2, weights[(m) & (looseM>-998)].sum()/weights[(m)].sum())
#        
#        #if (looseM[i]>-998):
#        #    looseVsJet.Fill(njets[i], weights[i])
#
#    kinVsJet.SetLineWidth(2)
#    looseVsJet.SetLineWidth(2)
#    c0=ROOT.TCanvas("c0","c0",800,800)
#    kinVsJet.SetLineColor(ROOT.kBlue)
#    looseVsJet.SetLineColor(ROOT.kRed)
#    kinVsJet.Draw("hist") 
#    looseVsJet.Draw("histsame")
#    l = ROOT.TLegend(0.7,0.1,0.9,0.3)
#    l.AddEntry(kinVsJet,    "Full Kinematic Reconstruction (Eff. %.1f )"%(weights[kinM>-998].sum()/weights.sum()*100))
#    l.AddEntry(looseVsJet,  "Loose Kinematcic Reconstruction (Eff. %.1f )"%(weights[looseM>-998].sum()/weights.sum()*100))
#    l.Draw()
#    c0.SaveAs("/nfs/dust/cms/user/celottog/mttNN/outputs/efficiencyRoot.pdf")
#    return
#


if __name__ == "__main__":
    #root()
    njet()
    main()
