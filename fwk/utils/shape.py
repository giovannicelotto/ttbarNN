import ROOT

import os

import uproot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
import six

def loadData(path,treeName,xTargets,yTargets,normY=True,normX=False):
	it= uproot.open(path)[treeName]

	df_in=it.pandas.df(xTargets)
	df_out=it.pandas.df(yTargets)

	in_=df_in.values
	out_=df_out.values



	if normY:
		std,mean,outNorm=normalizeData(out_)
	else:
		std=0.
		mean=0.
		outNorm=out_
	return in_,outNorm,std,mean


def normalizeData(data):
	std=np.std(data,axis=0)
	mean=np.mean(data,axis=0)

	out= (data-mean)/std
	return std, mean, out


inputFile="input/emu_ttbarsignalplustau_fromDilepton.root"
tName="plainTree_rec_step8"

targetVars=["Ett","Pxtt","Pytt","Pztt"]
inputVars = ["El1","Pxl1","Pyl1","Pzl1",
			"El2","Pxl2","Pyl2","Pzl2",
			"Ej1","Pxj1","Pyj1","Pzj1",
			"Ej2","Pxj2","Pyj2","Pzj2",
			"Emet","Pxmet","Pymet","Pzmet"]

xData,yData,std,mean = loadData(inputFile,tName,inputVars,targetVars,normY=False)


# from sklearn import preprocessing
# std_scale = preprocessing.StandardScaler().fit(xData)
# #~ std_scale = preprocessing.RobustScaler().fit(xData)
# #~ std_scale = preprocessing.QuantileTransformer(output_distribution='normal').fit(xData)
# outNorm2 = std_scale.transform(xData)
# std_scale2 = preprocessing.StandardScaler().fit(yData)
#~ std_scale2 = preprocessing.RobustScaler().fit(yData)
#~ std_scale2 = preprocessing.QuantileTransformer(output_distribution='normal').fit(yData)
# outNorm2y = std_scale2.transform(yData)
# outNorm2y = yData[:,3]
outNorm2y = yData[:,3]/np.mean(yData[:,0])

# print (outNorm2y[:,0].shape)
# print (outNorm2y[:,0])

#~ yy = np.histogram(outNorm2y[:,0],bins=140)
#~ print (yy[0].shape)
#~ print (yy[1].shape)

#~ plt.figure()

#~ newY=outNorm2y[:,0]/sum(outNorm2y[:,0])

#~ plt.hist(outNorm2y[:,0])
#~ a=plt.hist(outNorm2y[:,0],bins=100,range=(-3.5,3.5),density=True)
#~ a=plt.hist(newY,bins=100,range=(-3.5,3.5))
#~ plt.hist(outNorm2y[:,0],bins=280,range=(100,1500))
#~ plt.plot(yy[0],[1])


#~ y=(a[0])
#~ bins=(a[1])

#~ x = 0.5*(bins[1:]+bins[:-1])

#~ print(x.shape,y.shape)
#~ plt.plot(x,y)
#~ from scipy.optimize import curve_fit
#~ def func(x, a, b,c,d):
	#~ return 1./((x**2. - a**2.)**2. + a*a*b*b) + c*np.exp(d*x)
#~ , bounds=(0, [3., 1., 0.5]
#~ popt, pcov = curve_fit(func, x, y)

#~ def gauss(x, A, mu, sigma):
#~ def gauss(x,  mu, sigma):
    #~ return np.exp(-(x-mu)**2/(2.*sigma**2))
#~ popt, pcov = curve_fit(gauss, x, y)
#~ plt.plot(x,func(x,popt[0],popt[1],popt[2],popt[3]))
#~ plt.plot(x,gauss(x,popt[0],popt[1]))

#~ plt.savefig("shape.pdf")




h = ROOT.TH1F("hist","hist",100,-2.,5.)
for a in outNorm2y:
	h.Fill(a)
# h.Scale(1./h.Integral())
c = ROOT.TCanvas("c","c",800,800)
c.Draw()
h.Draw("hist")

#~ func = ROOT.TF1('func', '1. /((x**2 - [0]**2 ) + [1]**2 *[0]**2) + [3] * TMath::Exp([4]+x)', -2, 5)
#~ func = ROOT.TF1('func', '[0] * exp((x-[1])**2/[2]) + [3] * TMath::Exp(x)', -2, 5)
#~ func = ROOT.TF1('func', '[0] * TMath::Exp(-[1]*(x+[3])) + [2]', 0, 5)
#~ fit = h.Fit('func', 'S')



# def func(x,par):
# 	ret=1./(2.*np.pi)*(par[1])/((x[0]-par[0])**2. + par[1]**2. /4.)
# 	un=par[3]
# 	return par[2]*ret+un + par[4]*ROOT.TMath.Exp(par[5]*(x[0]-par[6]))
#
#
# #~ breitwigner = ROOT.TF1('breitwigner', func, -2.,4., 7)
# breitwigner = ROOT.TF1("fit","crystalball",-2.,4.)
# breitwigner.SetParameters(0.01,-0.5,1.,1.,1.)
# #~ breitwigner.SetParameters(175.,5.,100.,1.)
# #~ breitwigner.SetParLimits(1, 10., 50.)
#
# #~ h.Fit('breitwigner',"","",-2.,4.)
# h.Fit('fit',"","",-2.,4.)
# ROOT.gStyle.SetOptFit(1)
# print (breitwigner.GetChisquare()/breitwigner.GetNDF())
#
#
# breitwigner.Draw('same')

c.SaveAs("shit.pdf")
