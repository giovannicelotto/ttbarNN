import numpy as np
import ROOT
from array import array
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl

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

    minGen, maxGen = 250, 1500


    totGen[totGen>maxGen]=maxGen-0.01
    totGen[totGen<minGen]=minGen+0.01

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    bins= np.linspace(minGen, maxGen, 100)
    ax.hist(totGen, weights=weights,  bins= bins, alpha=1, histtype='step', linewidth=2, density=True, color='C0')
    ax.hist(totGen, weights=weights,  bins= bins, alpha=0.5, density=True, color='C0')
    
    ax.set_xlabel("$\mathrm{m^{gen}_{t\overline{t}}}$ [GeV]", fontsize=16)
    ax.set_ylabel("Counts per bin [GeV$^{-1}$]", fontsize=16)
    ax.tick_params(labelsize=14)
    ax.set_xlim(minGen, maxGen)
    #ax.legend(loc='best')
    
    fig.savefig("/nfs/dust/cms/user/celottog/mttNN/plots/mtt.pdf", bbox_inches='tight')


if __name__ == "__main__":
    main()
