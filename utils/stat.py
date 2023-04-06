import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde
from scipy.stats import entropy
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn import preprocessing
import ROOT
from array import array
import os
from scipy.stats.stats import pearsonr
from utils.helpers import NormalizeBinWidth1d

def KLdiv(p, q):
    xs_ = np.linspace(340, 800, 50)
    kde_p = gaussian_kde(p)(xs_) 
    kde_q = gaussian_kde(q)(xs_) 
    kl_divergence = entropy(kde_p, kde_q)
    return kl_divergence

def JSdist(p, q):
    xs_ = np.linspace(340, 800, 50)
    kde_p = gaussian_kde(p)(xs_) 
    kde_q = gaussian_kde(q)(xs_) 
    
    m = (kde_p + kde_q) / 2

    return [np.sqrt(entropy(kde_p, m)/2 + entropy(kde_q, m)/2), jensenshannon(kde_p, kde_q)]

def multiScale(featureNames, inX ):
    
    maxable   = [i for i in range(len(featureNames)) if any(substring in featureNames[i] for substring in ['_phi', 'njets', 'nbjets']) ]
    powerable = [i for i in range(len(featureNames)) if any(substring in featureNames[i] for substring in ['_pt', '_m', 'ht']) and not any(substring in featureNames[i] for substring in ['lep1_m', 'lep2_m','kr_ttbar_m'])]
    boxable   = [i for i in range(len(featureNames)) if any(substring in featureNames[i] for substring in ['kr_ttbar_m'])]
    keepable  = [i for i in range(len(featureNames)) if any(substring in featureNames[i] for substring in ['tag', 'score', 'channelID'])]
    scalable  = [i for i in range(len(featureNames)) if ((i not in maxable) & (i not in keepable) & (i not in powerable) & (i not in boxable))]
    
    inXs = inX

    maxer   = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(inX[:,maxable])
    powerer = PowerTransformer(method='yeo-johnson', standardize=True).fit(inX[:,powerable])
    boxer = PowerTransformer(method='box-cox', standardize=True).fit(inX[:,boxable])
    scaler  = preprocessing.StandardScaler().fit(inX[:,scalable])
    #for i in powerable:
    #    print(np.array(featureNames)[i], "\n")
    #    
    #    inXs[:, [i]] = powerer.transform(inXs[:, [i]])
    inXs[:, maxable] = maxer.transform(inXs[:,maxable])
    inXs[:, powerable] = powerer.transform(inXs[:, powerable])
    inXs[:, boxable] = boxer.transform(inXs[:, boxable])
    inXs[:, scalable] = scaler.transform(inXs[:,scalable])
    return inXs


def getWeightsTrain(outY_train, weights_train, outFolder, exp_tau, output=True):
    if (output):
        if not os.path.exists(outFolder+ "/model"):
            os.makedirs(outFolder+ "/model")

    weightBins = array('d', [340, 342.5, 345, 347.5, 350, 355, 360, 370, 380, 390, 400, 410, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600 ]) # 
    wegihtNBin = len(weightBins)-1
    weightHisto = ROOT.TH1F("weightHisto","weightHisto",wegihtNBin, weightBins)

    print("Filling histo with m_tt")
    for m, weight in zip(outY_train, weights_train):
        weightHisto.Fill(m,abs(weight))
    weightHisto = NormalizeBinWidth1d(weightHisto)

    
    '''xs_ = np.linspace(340, 800, 1000)
    kde = gaussian_kde(outY_train[:,0])(xs_)        
    kde = kde/np.max(kde)                           # set the max od the kde to 1
    xsmax = xs_[np.argmax(kde)]
    myexp = np.piecewise(xs_, [xs_<xsmax, xs_>=xsmax], [1, np.exp(-(xs_-xsmax)[xs_>=xsmax]/200)])
    #kde_prime = np.abs(kde[2:]-kde[:len(kde)-2]/(2*(xs_[1]-xs_[0])))
    #kde_sec = np.abs(kde[2:]-2*kde[1:len(kde)-1]+kde[:len(kde)-2]/(xs_[1]-xs_[0])**2)

    fig, ax = plt.subplots(1, 1)
    #ax.plot(xs_, myexp)
    ax.plot(xs_, (1/kde)*myexp)
    #ax.plot(xs_, (1/kde)*myexp)
    #ax.plot(xs_[1:len(xs_)-1], kde_prime)
    #ax.plot(xs_[1:len(xs_)-1], kde_sec)
    #ax.plot(xs_, 0.004*0.004*1/(0.004+kde), linewidth=0.5, marker='o', markersize=1)
    fig.savefig(outFolder+"/model/kde.pdf")
    plt.cla()'''
    canvas = ROOT.TCanvas()
    weightHisto.Draw("hist")
    weightHisto.SetTitle("Normalized and weighted m_{tt} distibutions. Training")
    weightHisto.SetYTitle('Normalized Counts')
    if (output):
        canvas.SaveAs(outFolder + "/model/mttOriginalWeight.pdf")

    weightHisto.Scale(1./weightHisto.Integral())
    maximumBin = weightHisto.GetMaximumBin()
    maximumBinContent = weightHisto.GetBinContent(maximumBin)

    firstMidpoint = (weightHisto.GetXaxis().GetBinLowEdge(1)+weightHisto.GetXaxis().GetBinUpEdge(1))/2
    #print("firstMidpoint: ", firstMidpoint)
    for binX in range(1,weightHisto.GetNbinsX()+1):
        midpoint = (weightHisto.GetXaxis().GetBinLowEdge(binX) + weightHisto.GetXaxis().GetBinUpEdge(binX))/2
        #print("Current midpoint:\t", midpoint)
        c = weightHisto.GetBinContent(binX)
        if (c>0):
            weightHisto.SetBinContent(binX, maximumBinContent/(c)*np.exp(-(midpoint-firstMidpoint)/exp_tau))
        #if ((binX >= maximumBin+10) & (c>0)):
        #    weightHisto.SetBinContent(binX, 1)
        #elif (c>0):
        #    weightHisto.SetBinContent(binX, maximumBinContent/c)
        else:
            weightHisto.SetBinContent(binX, 1.)     
    weightHisto.SetBinContent(0, weightHisto.GetBinContent(1))
    weightHisto.SetBinContent(weightHisto.GetNbinsX()+1, weightHisto.GetBinContent(weightHisto.GetNbinsX()))
    weightHisto.SetTitle("Weights distributions")
    canvas.SetLogy(1)
    if (output):
        canvas.SaveAs(outFolder+"/model/weights.pdf")
    weights_train_original = weights_train
    weights_train_=[]
    for w,m in zip(weights_train, outY_train):
        weightBin = weightHisto.FindBin(m)
        addW = weightHisto.GetBinContent(weightBin)	    # weights for importance
        weights_train_.append(abs(w)*addW)           	# weights of the training are the product of the original weights and the weights used to give more importance to regions with few events
    weights_train = np.array(weights_train_)		    # final weights_train is np array

    weights_train = 1./np.mean(weights_train)*weights_train
    return weights_train, weights_train_original


def printStat(outY_test, y_predicted, outFolder, model):
    corr = pearsonr(outY_test.reshape(outY_test.shape[0]), y_predicted.reshape(y_predicted.shape[0]))
    print("correlation:",corr[0])
    kl = KLdiv(y_predicted[:,0], outY_test[:,0])
    print ("KL", kl)
    js = JSdist(y_predicted[:,0], outY_test[:,0])
    print ("JS", str(js[0])[:6], str(js[1])[:6])
    with open(outFolder+"/model/Info.txt", "w") as f:
        f.write("JS\t"+ str(js[0])[:6]+"\n"+"KL\t"+ str(kl)[:6]+"\ncorrelation:\t"+str(corr[0])+"\n")
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write("\n")
        