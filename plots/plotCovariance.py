import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors

def plotCovariance(modelDir):
    dnnCov      =  np.load(modelDir+"/dnnCov.npy")
    looseCov    =  np.load(modelDir+"/looseCov.npy")
    kinCov      =  np.load(modelDir+"/kinCov.npy")
    
    covs = [dnnCov, looseCov, kinCov]
    covsName = ['dnnCov', 'looseCov', 'kinCov']
    nRecoBin = 7
    for idx in range(len(covs)):
        matrix = covs[idx]
        print("DET", np.linalg.det(matrix))
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        im = ax.matshow(matrix[:nRecoBin,:nRecoBin], cmap=plt.cm.jet, alpha=0.7, )
        ax.set_xlabel("Generated (bin number)", fontsize=18)
        #ax.set_xlabel("Generated (bin number)", fontsize=18)
        ax.set_ylabel("Reconstructed (bin number)", fontsize=18)
        ax.tick_params(labelsize=18)
        ax.xaxis.set_ticks(np.arange(0, nRecoBin, 1.0))
        ax.yaxis.set_ticks(np.arange(0, nRecoBin, 1.0))
        ax.xaxis.set_ticks_position('bottom')
        ax.invert_yaxis()
        for y in range(nRecoBin):
            for x in range(nRecoBin):
                plt.text(x , y , '%.1e' % matrix[y, x],
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=16
                        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.25)
        cbar = plt.colorbar(im, ax=ax, cax=cax)
        cbar.set_label(r'', fontsize=18)
        cbar.ax.tick_params(labelsize=18)
        ax.text(s="CMS Simulation", x=0.00, y=1.02, ha='left', fontsize=18,  transform=ax.transAxes)
        ax.text(s="(13 TeV)", x=1.00, y=1.02, ha='right', fontsize=18,  transform=ax.transAxes)
        fig.savefig("/nfs/dust/cms/user/celottog/mttNN/plots/"+covsName[idx]+".pdf", bbox_inches='tight')
        plt.cla()
    

    return


if __name__ == "__main__":
    modelDir = "/nfs/dust/cms/user/celottog/mttNN/outputs/15*None_[256_128_8_8]DoubleNN_rebinNN2_simple/model"
    plotCovariance(modelDir)