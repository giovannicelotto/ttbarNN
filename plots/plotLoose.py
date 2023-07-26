import numpy as np
import ROOT
from array import array
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

def openInThePhaseSpace(folder):
    print("Opening ",folder)
    looseM    = np.load(folder+"/lkrM.npy")
    totGen  = np.load(folder+"/totGen.npy")
    weights = np.load(folder+"/weights.npy")

    mask = (looseM>-998)
    
    
    looseM    = looseM[mask]
    weights = weights[mask]
    totGen  = totGen[mask]

    
    return looseM, totGen, weights


def main():
    print("Start")
    folder = "/nfs/dust/cms/user/celottog/mttNN/npyData/15*None"
    looseM, totGen, weights = openInThePhaseSpace(folder)

    minLoose, maxLoose = 340, 14000
    minGen, maxGen = 340, 14000

    looseM[looseM>maxLoose]=maxLoose-0.01
    totGen[totGen>maxGen]=maxGen-0.01
    totGen[totGen<minGen]=minGen+0.01
    looseM[looseM<minLoose]=minLoose+0.01
    

    binLoose = np.logspace(np.log10(340), np.log10(13000), 60)
    binLoose = np.append(binLoose, 18000)
    binGen = binLoose
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    counts, xedges, yedges, im = ax.hist2d(totGen, looseM, bins=(binGen, binLoose),norm=mpl.colors.LogNorm(), cmap=plt.cm.jet, cmin=1, weights=weights)
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.set_xlabel("$\mathrm{m_{t\overline{t}}^{gen}}$ [GeV]", fontsize=16)
    ax.set_ylabel("$\mathrm{m_{t\overline{t}}^{loose}}$ [GeV]", fontsize=16)
    
    
    ax.hlines(xmin=0, xmax=14000, y=13000, color='black', linestyle='--', linewidth=0.75)
    ax.set_ylim(minLoose, maxLoose)
    ax.set_xlim(minGen, maxGen)
    cbar_ax = fig.add_axes([0.9, 0.11, 0.05, 0.77])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Counts', fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    ax.tick_params(labelsize=14)
    
    fig.savefig("/nfs/dust/cms/user/celottog/mttNN/plots/looseM.pdf", bbox_inches='tight')
    



if __name__ == "__main__":
    main()
