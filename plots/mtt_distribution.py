import numpy as np
import ROOT
from array import array
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
import matplotlib.patches as mpatches

def openInThePhaseSpace(folder):
    print("Opening ",folder)
    #kinM    = np.load(folder+"/krM.npy")
    totGen  = np.load(folder+"/totGen.npy")
    weights = np.load(folder+"/weights.npy")
    mask    = np.load(folder+"/mask.npy")
    
    
    
    
    weights = weights[mask]
    totGen  = totGen[mask]

    
    return totGen, weights


def main():
    print("Start")
    folder = "/nfs/dust/cms/user/celottog/mttNN/npyData/15*None"
    totGen, weights = openInThePhaseSpace(folder)

    minGen, maxGen, nbin = 320, 2100, 100


    totGen[totGen>maxGen]=maxGen+0.01
    totGen[totGen<minGen]=minGen-0.01

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    bins= np.linspace(minGen, maxGen, nbin)
    bins = np.array([ 320, 340, 342.5, 345, 347.5, 350, 355, 360, 365, 370, 375, 380, 390, 400, 410,  420, 440, 460, 480, 500, 520, 540, 560, 580, 600, 630, 660, 690, 729, 750, 780, 810, 850, 870, 900, 950, 1000, 1050, 1100, 1150, 1200, 1500, 1800, 2100], 'f')
    print("Number of bins used:\t", len(bins)-1)
    #bins= np.append(bins, maxGen + 40)
    #bins = np.concatenate([[280], bins])
    
    counts = ax.hist(totGen, weights=weights,  bins= bins, alpha=1, histtype='step', linewidth=2, density=True, color='C0')[0]
    ax.hist(totGen, weights=weights,  bins= bins, alpha=0.5, density=True, color='C0')


    #Line of the weights
    ax2 = ax.twinx()
    ax2.hist(bins[:-1], bins=bins, weights=np.max(counts)/counts*1/(bins[:-1])**2, histtype=u'step', color='black')
    ax2.set_yscale('log')
    ax2.tick_params(labelsize=18)
    ax2.set_ylabel('Weight correction', fontsize=20)


    blue_patch = mpatches.Patch(color='C0', alpha=0.5, label=r'MC $\mathrm{m^{gen}_{t\bar{t}}}$')
    red_patch = mpatches.Patch(facecolor='white', edgecolor='black', alpha=1, label=r'$\frac{1}{f}\cdot \left(\mathrm{m^{gen}_{t\bar{t}}}\right)^{-2}$')

    # Combine legend handles and labels
    handles = [blue_patch, red_patch]
    labels = [patch.get_label() for patch in handles]
    ax.legend(handles=handles, labels=labels, loc='lower right', fontsize=18)
    
    ax.set_xlabel(r"$\mathrm{m^{gen}_{t\bar{t}}}$ [GeV]", fontsize=20)
    ax.set_ylabel("Normalized counts", fontsize=20)
    ax.text(s="Private Work (CMS Simulation)", x=0.00, y=1.02, ha='left', fontsize=16,  transform=ax.transAxes , **{'fontname':'Arial'})
    ax.text(s="(13 TeV)", x=1.00, y=1.02, ha='right', fontsize=16,  transform=ax.transAxes)
    ax.tick_params(labelsize=18)
    ax.set_xlim(bins[0], bins[-1])
    ymax = ax.get_ylim()[1]
    ax.set_ylim(0, ymax)
    #ax.vlines(x=maxGen, ymin=0, ymax = ymax, color='black', linewidth=1)
    #ax.vlines(x=minGen, ymin=0, ymax = ymax, color='black', linewidth=1)
    #ax.legend(loc='best')
    
    fig.savefig("/nfs/dust/cms/user/celottog/mttNN/plots/mtt.pdf", bbox_inches='tight')


if __name__ == "__main__":
    main()
