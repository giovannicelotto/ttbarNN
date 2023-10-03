import ROOT
import numpy as np
from progress.bar import IncrementalBar
import pandas as pd
import pickle


def phaseSpaceChange(fileNames):
    print("Creating space factor")
    spaceFactors = {
          'NOMINAL' : 0,
          'NOMINAL_NORENORMALIZATION': 0,

          'PU_UP' : 0,
          'PU_DOWN' : 0,
          'BFRAG_UP' : 0,
          'BFRAG_DOWN' : 0,
          'BFRAG_PETERSON' : 0,
          'BSEMILEP_UP' : 0,
          'BSEMILEP_DOWN' : 0,
          'MESCALE_UP' : 0,
          'MESCALE_DOWN' : 0,
          'MEFACSCALE_UP' : 0,
          'MEFACSCALE_DOWN' : 0,
          'MERENSCALE_UP' : 0,
          'MERENSCALE_DOWN' : 0,
          'PDF_ALPHAS_UP' : 0,
          'PDF_ALPHAS_DOWN' : 0,
          'ML_HDAMP_UP'             : 0,
          'ML_HDAMP_DOWN'           : 0,
          'PSSCALE_WEIGHT_FSR_UP'             : 0,
          'PSSCALE_WEIGHT_FSR_DOWN'           : 0,
          'PSSCALE_WEIGHT_ISR_UP'             : 0,
          'PSSCALE_WEIGHT_ISR_DOWN'           : 0,
          'TOP_PT' : 0,



    }
    for idx in range (1, 52):
        spaceFactors['PDF_UP_'+str(idx)] = 0
        spaceFactors['PDF_DOWN_'+str(idx)] = 0
    spaceFactors['PDF_CENTRAL_0'] = 0

    #for idx in range (1, 24):
    #    spaceFactors['PSSCALE_WEIGHT_UP_'+str(idx)] = 0
    #    spaceFactors['PSSCALE_WEIGHT_DOWN_'+str(idx)] = 0
    #spaceFactors['PSSCALE_WEIGHT_CENTRAL_1'] = 0

    for fileName in fileNames:
            f = ROOT.TFile.Open(fileName)
            directory = f.Get("normalization_histograms")

            histo = f.Get("trueLevelWeightSum_hist")
            spaceFactors['NOMINAL'] = spaceFactors['NOMINAL'] + histo.GetBinContent(1)

            histo = f.Get("trueLevelNoRenormalisationWeightSum_hist")
            spaceFactors['NOMINAL_NORENORMALIZATION'] = spaceFactors['NOMINAL_NORENORMALIZATION'] + histo.GetBinContent(1)
            
            histo = directory.Get("trueLevelWeightSum_hist_PU_UP")
            spaceFactors['PU_UP'] = spaceFactors['PU_UP'] + histo.GetBinContent(1)

            histo = directory.Get("trueLevelWeightSum_hist_PU_DOWN")
            spaceFactors['PU_DOWN'] = spaceFactors['PU_DOWN'] + histo.GetBinContent(1)

            histo = directory.Get("trueLevelWeightSum_hist_TOP_PT")
            spaceFactors['TOP_PT'] = spaceFactors['TOP_PT'] + histo.GetBinContent(1)

            histo = directory.Get("trueLevelWeightSum_hist_MESCALE_UP")
            spaceFactors['MESCALE_UP'] = spaceFactors['MESCALE_UP'] + histo.GetBinContent(1)

            histo = directory.Get("trueLevelWeightSum_hist_MESCALE_DOWN")
            spaceFactors['MESCALE_DOWN'] = spaceFactors['MESCALE_DOWN'] + histo.GetBinContent(1)

            histo = directory.Get("trueLevelWeightSum_hist_MEFACSCALE_UP")
            spaceFactors['MEFACSCALE_UP'] = spaceFactors['MEFACSCALE_UP'] + histo.GetBinContent(1)

            histo = directory.Get("trueLevelWeightSum_hist_MEFACSCALE_DOWN")
            spaceFactors['MEFACSCALE_DOWN'] = spaceFactors['MEFACSCALE_DOWN'] + histo.GetBinContent(1)

            histo = directory.Get("trueLevelWeightSum_hist_MERENSCALE_UP")
            spaceFactors['MERENSCALE_UP'] = spaceFactors['MERENSCALE_UP'] + histo.GetBinContent(1)

            histo = directory.Get("trueLevelWeightSum_hist_MERENSCALE_DOWN")
            spaceFactors['MERENSCALE_DOWN'] = spaceFactors['MERENSCALE_DOWN'] + histo.GetBinContent(1)

            histo = directory.Get("trueLevelWeightSum_hist_BFRAG_UP")
            spaceFactors['BFRAG_UP'] = spaceFactors['BFRAG_UP'] + histo.GetBinContent(1)

            histo = directory.Get("trueLevelWeightSum_hist_BFRAG_DOWN")
            spaceFactors['BFRAG_DOWN'] = spaceFactors['BFRAG_DOWN'] + histo.GetBinContent(1)

            histo = directory.Get("trueLevelWeightSum_hist_BFRAG_PETERSON")
            spaceFactors['BFRAG_PETERSON'] = spaceFactors['BFRAG_PETERSON'] + histo.GetBinContent(1)

            histo = directory.Get("trueLevelWeightSum_hist_BSEMILEP_UP")
            spaceFactors['BSEMILEP_UP'] = spaceFactors['BSEMILEP_UP'] + histo.GetBinContent(1)

            histo = directory.Get("trueLevelWeightSum_hist_BSEMILEP_DOWN")
            spaceFactors['BSEMILEP_DOWN'] = spaceFactors['BSEMILEP_DOWN'] + histo.GetBinContent(1)

            histo = directory.Get("trueLevelWeightSum_hist_ML_HDAMP_UP")
            spaceFactors['ML_HDAMP_UP'] = spaceFactors['ML_HDAMP_UP'] + histo.GetBinContent(1)

            histo = directory.Get("trueLevelWeightSum_hist_ML_HDAMP_DOWN")
            spaceFactors['ML_HDAMP_DOWN'] = spaceFactors['ML_HDAMP_DOWN'] + histo.GetBinContent(1)

            histo = directory.Get("trueLevelWeightSum_hist_PDF_ALPHAS_UP")
            spaceFactors['PDF_ALPHAS_UP'] = spaceFactors['PDF_ALPHAS_UP'] + histo.GetBinContent(1)

            histo = directory.Get("trueLevelWeightSum_hist_PDF_ALPHAS_DOWN")
            spaceFactors['PDF_ALPHAS_DOWN'] = spaceFactors['PDF_ALPHAS_DOWN'] + histo.GetBinContent(1)

            histo = directory.Get("trueLevelWeightSum_hist_PSSCALE_WEIGHT_UP_14")
            spaceFactors['PSSCALE_WEIGHT_ISR_UP'] = spaceFactors['PSSCALE_WEIGHT_ISR_UP'] + histo.GetBinContent(1)

            histo = directory.Get("trueLevelWeightSum_hist_PSSCALE_WEIGHT_DOWN_14")
            spaceFactors['PSSCALE_WEIGHT_ISR_DOWN'] = spaceFactors['PSSCALE_WEIGHT_ISR_DOWN'] + histo.GetBinContent(1)

            histo = directory.Get("trueLevelWeightSum_hist_PSSCALE_WEIGHT_UP_3")
            spaceFactors['PSSCALE_WEIGHT_FSR_UP'] = spaceFactors['PSSCALE_WEIGHT_FSR_UP'] + histo.GetBinContent(1)

            histo = directory.Get("trueLevelWeightSum_hist_PSSCALE_WEIGHT_DOWN_3")
            spaceFactors['PSSCALE_WEIGHT_FSR_DOWN'] = spaceFactors['PSSCALE_WEIGHT_FSR_DOWN'] + histo.GetBinContent(1)



            for id in range(1, 52):
                histo = directory.Get("trueLevelWeightSum_hist_PDF_UP_"+str(id))
                spaceFactors['PDF_UP_'+str(id)] = spaceFactors['PDF_UP_'+str(id)] + histo.GetBinContent(1)
                histo = directory.Get("trueLevelWeightSum_hist_PDF_DOWN_"+str(id))
                spaceFactors['PDF_DOWN_'+str(id)] = spaceFactors['PDF_DOWN_'+str(id)] + histo.GetBinContent(1)
            histo = directory.Get("trueLevelWeightSum_hist_PDF_CENTRAL_0")
            spaceFactors['PDF_CENTRAL_0'] = spaceFactors['PDF_CENTRAL_0'] + histo.GetBinContent(1)

            #for id in range(1, 24):
            #    histo = directory.Get("trueLevelWeightSum_hist_PSSCALE_WEIGHT_UP_"+str(id))
            #    spaceFactors['PSSCALE_WEIGHT_UP_'+str(id)] = spaceFactors['PSSCALE_WEIGHT_UP_'+str(id)] + histo.GetBinContent(1)
            #    if id ==1:
            #        continue
            #    histo = directory.Get("trueLevelWeightSum_hist_PSSCALE_WEIGHT_DOWN_"+str(id))
            #    spaceFactors['PSSCALE_WEIGHT_DOWN_'+str(id)] = spaceFactors['PSSCALE_WEIGHT_DOWN_'+str(id)] + histo.GetBinContent(1)


    
    print(spaceFactors)

    with open("/nfs/dust/cms/user/celottog/mttNN/systematics/dictionaryPhaseFactors.pkl", "wb") as file:
        pickle.dump(spaceFactors, file)



    
    

     