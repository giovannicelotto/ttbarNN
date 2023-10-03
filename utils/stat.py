import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde
from scipy.stats import entropy
from sklearn.preprocessing import PowerTransformer
#from sklearn.preprocessing import QuantileTransformer
from sklearn import preprocessing
import ROOT
from array import array
import os
#from scipy.stats.stats import pearsonr
from utils.helpers import NormalizeBinWidth1d
import pickle
from sklearn.metrics import mean_squared_error
import sys
sys.path.insert(1, '/nfs/dust/cms/user/celottog/mttNN')
from npyData.checkFeatures import checkFeatures
from tabulate import tabulate

def KLdiv(p, q):
    xs = np.linspace(300, 1500, 100)
    kde_p = gaussian_kde(p)(xs) 
    kde_q = gaussian_kde(q)(xs) 
    kl_divergence = entropy(kde_p, kde_q)
    return kl_divergence

def JSdist(p, q, weights):
    x1, x2, nbin = 250, 1500, 50
    bins = np.linspace(x1, x2, nbin)
    p[p>x2]=x2-1
    p[p<x1]=x1+1
    q[q>x2]=x2-1
    q[q<x1]=x1+1

    histP = np.histogram(p, weights=weights, bins=bins)[0]
    histQ = np.histogram(q, weights=weights, bins=bins)[0]
    
    m = (histP + histQ) / 2

    return [np.sqrt(entropy(histP, m)/2 + entropy(histQ, m)/2), jensenshannon(histP, histQ)]

def multiScale(featureNames, inX, outName):
    
    maxable   = [i for i in range(len(featureNames)) if any(substring in featureNames[i] for substring in ['_phi', 'njets', 'nbjets', 'kr_ttbar_m']) ]
    powerable = [i for i in range(len(featureNames)) if any(substring in featureNames[i] for substring in ['_pt', '_m', 'ht']) and not any(substring in featureNames[i] for substring in ['lep1_m', 'lep2_m','kr_ttbar_m'])]
    boxable   = [i for i in range(len(featureNames)) if any(substring in featureNames[i] for substring in ['nulla qui'])]
    keepable  = [i for i in range(len(featureNames)) if any(substring in featureNames[i] for substring in ['tag', 'score'])]
    scalable  = [i for i in range(len(featureNames)) if ((i not in maxable) & (i not in keepable) & (i not in powerable) & (i not in boxable))]
    
    inXs = inX

    maxer   = preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit(inX[:,maxable])
    powerer = PowerTransformer(method='yeo-johnson', standardize=True).fit(inX[:,powerable])
    #boxer = PowerTransformer(method='box-cox', standardize=True).fit(inX[:,boxable])
    scaler  = preprocessing.StandardScaler().fit(inX[:,scalable])
    
    inXs[:, maxable] = maxer.transform(inXs[:,maxable])
    inXs[:, powerable] = powerer.transform(inXs[:, powerable])
    #inXs[:, boxable] = boxer.transform(inXs[:, boxable])
    inXs[:, scalable] = scaler.transform(inXs[:,scalable])


    print("Saving the scalers...")
    scalers = {
    'type'  : 'multi',
    'maxer': maxer,
    'powerer': powerer,
    'scaler': scaler,
    'maxable': maxable,
    'powerable': powerable,
    'boxable': boxable,
    'keepable': keepable,
    'scalable': scalable
    }

    with open(outName, 'wb') as file:
        pickle.dump(scalers, file)

    return inXs


def standardScale(featureNames, inX,  outName):

    #keepable  = [i for i in range(len(featureNames)) if any(substring in featureNames[i] for substring in ['tag', 'score'])]
    keepable = []
    scalable  = [i for i in range(len(featureNames)) if ( i not in keepable )]
    
    inXs = inX
#remove sample_weight
    scaler  = preprocessing.StandardScaler().fit(inX[:,scalable])
    inXs[:, scalable] = scaler.transform(inXs[:,scalable])


    print("Saving the scalers...")
    scalers = {
    'type'  : 'standard',
    'scaler': scaler,
    'keepable': keepable,
    'scalable': scalable
    }

    with open(outName, 'wb') as file:
        pickle.dump(scalers, file)
    return inXs



def getWeightsTrain(outY_train, weights_train, outFolder, alpha, output=True, outFix = None):
    print("Producing weights for training...")
    if (output):
        if not os.path.exists(outFolder+ "/model"):
            os.makedirs(outFolder+ "/model")

    assert (outY_train>0).all(), "Check stat getWeights"
    
    #weightBins = array('d', [ 340, 342.5, 345, 347.5, 350, 355, 360, 365, 370, 375, 380, 390,  410, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600 ]) # def
    
    #weightBins = array('d', [ 300, 320, 340, 342.5, 345, 347.5, 350, 355, 360, 365, 370, 375, 380, 390,  600, 650, 700, 750, 800, 850, 900, 950, 1200]) #weightHigh
    #weightBins = array('d', [ 300, 320, 340, 342.5, 345, 347.5, 350, 355, 360, 365, 370, 375, 380, 390, 400, 410,  420, 440, 460, 480, 500, 520, 540, 560, 580, 600, 630, 660, 690, 729, 750, 780, 810, 850, 870, 900, 950, 1000, 1050, 1100, 1150, 1300]) # nmape
    weightBins = array('d', [ 320, 340, 342.5, 345, 347.5, 350, 355, 360, 365, 370, 375, 380, 390, 400, 410,  420, 440, 460, 480, 500, 520, 540, 560, 580, 600, 630, 660, 690, 729, 750, 780, 810, 850, 870, 900, 950, 1000, 1050, 1100, 1150, 1200, 1500, 1800, 2100]) # newLoss
    if outFix == '2NN':
        weightBins = array('d',[ 330, 360, 380, 400, 425, 450, 500, 540, 600, 650, 750, 900, 1100, 1500]) 
    #weightBins = array('d', [ 300, 320, 340, 342.5, 345, 347.5, 350, 355, 360, 365, 370, 375, 380, 390,  410]) # def
    
    weightNBin = len(weightBins)-1
    weightHisto = ROOT.TH1F("weightHisto","weightHisto",weightNBin, weightBins)

    print("Filling histo with m_tt")
    for m, weight in zip(outY_train, weights_train):
        weightHisto.Fill(m, abs(weight))
    weightHisto = NormalizeBinWidth1d(weightHisto)


    canvas = ROOT.TCanvas()
    weightHisto.Draw("hist")
    weightHisto.SetTitle("Normalized and weighted $m_{t\\bar{t}}$ distributions. Training")

    weightHisto.SetYTitle('Counts')
    if (output):
        if outFix is None:
            canvas.SaveAs(outFolder + "/model/mttOriginalWeight.pdf")
        else:
            outFix = "_" + outFix
            canvas.SaveAs(outFolder + "/model/mttOriginalWeight" + outFix + ".pdf")
        

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

            weightHisto.SetBinContent(binX, (maximumBinContent)/(c*midpoint*midpoint)) 


        else:
            weightHisto.SetBinContent(binX, 1.)     
    weightHisto.SetBinContent(0, weightHisto.GetBinContent(1))
    weightHisto.SetBinContent(weightHisto.GetNbinsX()+1, weightHisto.GetBinContent(weightHisto.GetNbinsX()))
    weightHisto.SetTitle("addW distributions")
    canvas.SetLogy(1)
    if (output):
        if outFix is None:
            canvas.SaveAs(outFolder+"/model/weights.pdf")
        else:
            canvas.SaveAs(outFolder+"/model/weights"+outFix+".pdf")

    weights_train_=[]
    for w,m in zip(weights_train, outY_train):
        weightBin = weightHisto.FindBin(m)
        addW = weightHisto.GetBinContent(weightBin)	    # weights for importance
        weights_train_.append(abs(w)*addW)           	# weights of the training are the product of the original weights and the weights used to give more importance to regions with few events
    weights_train = np.array(weights_train_)		    # final weights_train is np array  
    
    #

    #Inverse squared weight
    assert (outY_train>0).all(), "outY_train negative value in getWeights train"
    assert len(weights_train)==len(outY_train)
    #weights_train = weights_train/(outY_train[:,0]*outY_train[:,0])
    #print(weights_train)
    #print(outY_train)
    
    print("Mean of Weights ", np.mean(weights_train)  )
    weights_train = weights_train/np.mean(weights_train) # to have mean = 1
    if (output):
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        ax[0].hist2d(outY_train[:,0], weights_train, bins=((50, 50)), range=((250, 1500), (0, 8)), cmap='Blues', norm=mpl.colors.LogNorm())
        ax[1].hist(weights_train, label='Training', bins=100)
        ax[1].set_yscale('log')
        ax[1].legend(fontsize=18)
        if outFix is None:
            fig.savefig(outFolder+"/model/weightsBeforeAndAfter.png")
        else:
            fig.savefig(outFolder+"/model/weightsBeforeAndAfter"+outFix+".png")

    return weights_train

def scaleNonAnalytical(featureNames, inX_train,  inX_test, npyDataFolder, outFolder):
# Do the sacling on training, save it and apply it to the testing
    dnn2Mask_train = inX_train[:,0]<-4998
    dnn2Mask_test  = inX_test[:,0]<-4998
    SinX_train       = inX_train[dnn2Mask_train,15:]
    #Sweights_train   = weights_train[dnn2Mask_train]
    SinX_test        = inX_test[dnn2Mask_test, 15:]
    #Sweights_train, Sweights_train_original = getWeightsTrain(outY_train[dnn2Mask_train], Sweights_train, outFolder=outFolder, alpha = hp['alpha'], output=False)
    
    print(" Scaling the events without K&L")
    
    SinX_train = standardScale(featureNames, SinX_train[:,:], outFolder+"/Sscalers.pkl")
    checkFeatures(SinX_train[:,:], npyDataFolder, name="SscaledPlot", featureNames = featureNames)
    
    with open(outFolder+"/Sscalers.pkl", 'rb') as file:
            scalers = pickle.load(file)
            scaler = scalers['scaler']
            scalable = scalers['scalable']
            SinX_test[:, scalable]  = scaler.transform( SinX_test [:, scalable])
    checkFeatures(SinX_train[:,:], npyDataFolder, name="SscaledPlotTest", featureNames = featureNames)
    return SinX_train, SinX_test


def scalerMC(modelDir, MCInX):
# modelDir = the folder where two scalers are present
# MCInX    = data features to be scaled
# return scaled data only
    
    with open(modelDir + "/scalers.pkl", 'rb') as file:
        scalers = pickle.load(file)
    # Getting the scalers od the first NN
        if (scalers['type']=='multi'):
            scalerType = scalers['type']
            maxer = scalers['maxer']
            powerer = scalers['powerer']
            scaler = scalers['scaler']
            maxable = scalers['maxable']
            powerable = scalers['powerable']
            scalable = scalers['scalable']

        if (scalers['type'] == 'standard'):
            scalerType = scalers['type']
            scaler = scalers['scaler']
            scalable = scalers['scalable']
    # Getting scalers of the second NN
    
    with open(modelDir + "/Sscalers.pkl", 'rb') as file:
        scalers = pickle.load(file)

        if (scalers['type']=='multi'):
            scalerType2 = scalers['type']
            maxer2      = scalers['maxer']
            powerer2    = scalers['powerer']
            scaler2     = scalers['scaler']
            maxable2    = scalers['maxable']
            powerable2  = scalers['powerable']
            scalable2   = scalers['scalable']

        if (scalers['type']=='standard'):
            scalerType2 = scalers['type']
            scaler2 = scalers['scaler']
            scalable2 = scalers['scalable']

    dnnMaskMC = (MCInX[:,0]>-998)
    MCInXscaled = MCInX[dnnMaskMC, :]
    
    if (scalerType == 'standard'):
        MCInXscaled[:, scalable]  = scaler.transform(   MCInXscaled [:, scalable])
    elif (scalerType == 'multi'):
        MCInXscaled[:, maxable]   = maxer.transform(    MCInXscaled [:, maxable])
        MCInXscaled[:, powerable] = powerer.transform(  MCInXscaled [:, powerable])
        MCInXscaled[:, scalable]  = scaler.transform(   MCInXscaled [:, scalable])
    MCInX[dnnMaskMC, :] = MCInXscaled            

    dnn2MaskMC = (MCInX[:,0]<-4998)
    MCInXscaled = MCInX[dnn2MaskMC, 15:]

    if (scalerType2 == 'standard'):
        MCInXscaled[:, scalable2]  = scaler2.transform( MCInXscaled [:, scalable2])
    elif (scalerType2 == 'multi'):
        MCInXscaled[:, maxable2]   = maxer2.transform( MCInXscaled  [:, maxable2])
        MCInXscaled[:, powerable2] = powerer2.transform( MCInXscaled[:, powerable2])
        MCInXscaled[:, scalable2]  = scaler2.transform( MCInXscaled [:, scalable2])
    MCInX[dnn2MaskMC, 15:] = MCInXscaled

    return MCInX



def printStat(totGen, ys_methods, weights, outFolder, model):
    # Take all the predictions and the corresponding generated values
    # Compute the Pearson weighted correlation coefficient, JS divergence and weighted MSE
    # Can be used for DNN, lkrM, and krM
    data = []
    
    methods = ['DNN', 'Loose', 'Kin']
    ind = 0
    for y_pred in ys_methods:
        mm = y_pred>-998
        x = totGen[mm]
        y = (y_pred)[mm]
        mx = np.average(x, weights = weights[mm])
        my = np.average(y, weights = weights[mm])
        num = np.sum(weights[mm] * (x - mx) * (y - my))
        den = np.sqrt(np.sum(weights[mm] * (x - mx)**2) * np.sum(weights[mm] * (y - my)**2))
        corr = num/den
        js = JSdist(y, x, weights=weights[mm])
        mseTest = mean_squared_error(x, y, sample_weight=weights[mm])

        normalizedMeanSquared = np.sqrt(np.average(((y-x)**2)/(x**2), weights = weights[mm]))

        data.append([methods[ind], corr, js, mseTest, normalizedMeanSquared])
        ind = ind +1

    print(tabulate(data, headers=['Method', 'Corr', 'JS', 'MSE', 'nwmse'], tablefmt='grid'))
    

    with open(outFolder+"/model/Info.txt", "a+") as f:
        f.write("\n")
        f.write(tabulate(data, headers=['Method', 'Corr', 'JS', 'MSE', 'nwmse'], tablefmt='grid'))
        f.write("\n")
        model.summary(print_fn=lambda x: f.write(x + '\n'))
        f.write("\n")
    
#def computePredicted(inX_train, inX_valid, inX_test, npyDataFolder, model):
#    print("Computing predictions...")
#    y_predicted = model.predict(inX_test)
#    y_predicted_train = model.predict(inX_train)
#    y_predicted_valid = model.predict(inX_valid)
#    print("Saving predictions...")
#    np.save(npyDataFolder+"/testing/flat_regY"   + "_test.npy", y_predicted)
#    np.save(npyDataFolder+"/testing/flat_regY"   + "_train.npy", y_predicted_train)
#    np.save(npyDataFolder+"/testing/flat_regY"   + "_valid.npy", y_predicted_valid)
#    return y_predicted_train, y_predicted_valid, y_predicted