
import ROOT
import numpy as np
from progress.bar import IncrementalBar
import pandas as pd


def getVarNames():
    varNames = ['weight',               'leptonSF',             'triggerSF',                    'btagSF',               'pileupSF', 'prefiringWeight',
                'var_ele_id_down',      'var_ele_id_up',        'var_btag_ele_id_down',         'var_btag_ele_id_up',
                'var_ele_reco_down',    'var_ele_reco_up',      'var_btag_ele_reco_down',       'var_btag_ele_reco_up',      
                'var_muon_id_down',     'var_muon_id_up',       'var_btag_muon_id_down',        'var_btag_muon_id_up',
                'var_muon_id_syst_down','var_muon_id_syst_up',  'var_btag_muon_id_syst_down',   'var_btag_muon_id_syst_up', 
                'var_muon_id_stat_down','var_muon_id_stat_up',  'var_btag_muon_id_stat_down',   'var_btag_muon_id_stat_up',
                'var_muon_iso_down',    'var_muon_iso_up',      'var_btag_muon_iso_down',       'var_btag_muon_iso_up',
                'var_muon_iso_syst_down','var_muon_iso_syst_up','var_btag_muon_iso_syst_down',  'var_btag_muon_iso_syst_up',
                'var_muon_iso_stat_down', 'var_muon_iso_stat_up', 'var_btag_muon_iso_stat_down', 'var_btag_muon_iso_stat_up',
                'var_pu_down',           'var_pu_up',           'var_btag_pu_down',             'var_btag_pu_up',
                'var_trig_down',         'var_trig_up',         'var_btag_trig_down',         'var_btag_trig_up',         
                'var_btag_down',         'var_btag_up',
                'var_btag_ljet_down',    'var_btag_ljet_up',    'var_btag_statistic_down',      'var_btag_statistic_up',
                'var_btag_pileup_down',  'var_btag_pileup_up',  'var_btag_jes_down',            'var_btag_jes_up',      'var_btag_type3_down',          'var_btag_type3_up',
                'var_l1prefiring_down', 'var_l1prefiring_up',   'var_btag_l1prefiring_down',    'var_btag_l1prefiring_up',
                'var_merenscale_down',  'var_merenscale_up',    'var_mefacscale_down',          'var_mefacscale_up',
                'var_mescale_down',     'var_mescale_up',       'var_btag_merenscale_down',     'var_btag_merenscale_up',
                'var_btag_mefacscale_down', 'var_btag_mefacscale_up','var_btag_mescale_down',   'var_btag_mescale_up',
                'var_bsemilep_down',    'var_bsemilep_up',      'var_btag_bsemilep_down',       'var_btag_bsemilep_up',
                'var_pdf_alphas_down',  'var_pdf_alphas_up',    'var_btag_pdf_alphas_down',     'var_btag_pdf_alphas_up',
                'var_ml_hdamp_up',      'var_ml_hdamp_down',    'var_btag_ml_hdamp_up',         'var_btag_ml_hdamp_down',
                'var_bfrag_down',       'var_bfrag_up',         'var_btag_bfrag_down',          'var_btag_bfrag_up',        'var_bfrag_central',    'var_btag_bfrag_central',
                # PDF in the end
                'var_psscale_weight_isr_up',            'var_psscale_weight_isr_down',          'var_btag_psscale_weight_isr_up',           'var_btag_psscale_weight_isr_down',
                'var_psscale_weight_fsr_up',            'var_psscale_weight_fsr_down',          'var_btag_psscale_weight_fsr_up',           'var_btag_psscale_weight_fsr_down',
                #'var_psscale_weight_fsr_G2GG_muR_up', 'var_psscale_weight_fsr_G2GG_muR_down', 'var_btag_psscale_weight_fsr_G2GG_muR_up', 'var_btag_psscale_weight_fsr_G2GG_muR_down',
                #'var_psscale_weight_fsr_G2QQ_muR_up', 'var_psscale_weight_fsr_G2QQ_muR_down', 'var_btag_psscale_weight_fsr_G2QQ_muR_up', 'var_btag_psscale_weight_fsr_G2QQ_muR_down',
                #'var_psscale_weight_fsr_Q2QG_muR_up', 'var_psscale_weight_fsr_Q2QG_muR_down', 'var_btag_psscale_weight_fsr_Q2QG_muR_up', 'var_btag_psscale_weight_fsr_Q2QG_muR_down',
                #'var_psscale_weight_fsr_X2XG_muR_up', 'var_psscale_weight_fsr_X2XG_muR_down', 'var_btag_psscale_weight_fsr_X2XG_muR_up', 'var_btag_psscale_weight_fsr_X2XG_muR_down',
                #'var_psscale_weight_fsr_G2GG_cNS_up', 'var_psscale_weight_fsr_G2GG_cNS_down', 'var_btag_psscale_weight_fsr_G2GG_cNS_up', 'var_btag_psscale_weight_fsr_G2GG_cNS_down',
                #'var_psscale_weight_fsr_G2QQ_cNS_up', 'var_psscale_weight_fsr_G2QQ_cNS_down', 'var_btag_psscale_weight_fsr_G2QQ_cNS_up', 'var_btag_psscale_weight_fsr_G2QQ_cNS_down',
                #'var_psscale_weight_fsr_Q2QG_cNS_up', 'var_psscale_weight_fsr_Q2QG_cNS_down', 'var_btag_psscale_weight_fsr_Q2QG_cNS_up', 'var_btag_psscale_weight_fsr_Q2QG_cNS_down',
                #'var_psscale_weight_fsr_X2XG_cNS_up', 'var_psscale_weight_fsr_X2XG_cNS_down', 'var_btag_psscale_weight_fsr_X2XG_cNS_up', 'var_btag_psscale_weight_fsr_X2XG_cNS_down',
                #'var_psscale_weight_isr_G2GG_muR_up', 'var_psscale_weight_isr_G2GG_muR_down', 'var_btag_psscale_weight_isr_G2GG_muR_up', 'var_btag_psscale_weight_isr_G2GG_muR_down',
                #'var_psscale_weight_isr_G2QQ_muR_up', 'var_psscale_weight_isr_G2QQ_muR_down', 'var_btag_psscale_weight_isr_G2QQ_muR_up', 'var_btag_psscale_weight_isr_G2QQ_muR_down',
                #'var_psscale_weight_isr_Q2QG_muR_up', 'var_psscale_weight_isr_Q2QG_muR_down', 'var_btag_psscale_weight_isr_Q2QG_muR_up', 'var_btag_psscale_weight_isr_Q2QG_muR_down',
                #'var_psscale_weight_isr_X2XG_muR_up', 'var_psscale_weight_isr_X2XG_muR_down', 'var_btag_psscale_weight_isr_X2XG_muR_up', 'var_btag_psscale_weight_isr_X2XG_muR_down',
                #'var_psscale_weight_isr_G2GG_cNS_up', 'var_psscale_weight_isr_G2GG_cNS_down', 'var_btag_psscale_weight_isr_G2GG_cNS_up', 'var_btag_psscale_weight_isr_G2GG_cNS_down',
                #'var_psscale_weight_isr_G2QQ_cNS_up', 'var_psscale_weight_isr_G2QQ_cNS_down', 'var_btag_psscale_weight_isr_G2QQ_cNS_up', 'var_btag_psscale_weight_isr_G2QQ_cNS_down',
                #'var_psscale_weight_isr_Q2QG_cNS_up', 'var_psscale_weight_isr_Q2QG_cNS_down', 'var_btag_psscale_weight_isr_Q2QG_cNS_up', 'var_btag_psscale_weight_isr_Q2QG_cNS_down',
                #'var_psscale_weight_isr_X2XG_cNS_up', 'var_psscale_weight_isr_X2XG_cNS_down', 'var_btag_psscale_weight_isr_X2XG_cNS_up', 'var_btag_psscale_weight_isr_X2XG_cNS_down',
                
                'var_bfrag_peterson',   'var_btag_bfrag_peterson',  'var_top_pt']
    varNames.append('var_pdf_central_0')
    for idx in range(1, 52):
        varNames.append('var_pdf_down_'+str(idx))
        varNames.append('var_pdf_up_'+str(idx))
    #for idx in range(1, 52):
    return varNames.copy()




def getVariations(fileNames, treeName):
    
# List that will be converted to numpy array to be used in the NN
    
    weightsSFVar, wsvList=[], []        # weights to be used in the NN
    

    for fileName in fileNames:
        f = ROOT.TFile.Open(fileName)
        tree = f.Get(treeName)
    
        print("Read TTree: {} (Entries: {})".format(treeName, tree.GetEntries()))
# Setting branches in the tree
        weight = np.array([0], dtype='d')
        leptonSF = np.array([0], dtype='f')
        triggerSF = np.array([0], dtype='f')
        btagSF = np.array([0], dtype='f')
        pileupSF = np.array([0], dtype='f')
        prefiringWeight = np.array([0], dtype='f')
        
        
# In principle btag is not affected by lepton systematics
# ELE ID and RECO        
        var_ele_id_down = np.array([0], dtype='f')
        var_ele_id_up = np.array([0], dtype='f')
        var_btag_ele_id_down = np.array([0], dtype='f')
        var_btag_ele_id_up = np.array([0], dtype='f')
        var_ele_reco_down = np.array([0], dtype='f')
        var_ele_reco_up = np.array([0], dtype='f')
        var_btag_ele_reco_down = np.array([0], dtype='f')
        var_btag_ele_reco_up = np.array([0], dtype='f')
        

# MUON ID and ISO
        var_muon_id_down = np.array([0], dtype='f')
        var_muon_id_up = np.array([0], dtype='f')
        var_muon_id_syst_down = np.array([0], dtype='f')
        var_muon_id_syst_up = np.array([0], dtype='f')
        var_muon_id_stat_down = np.array([0], dtype='f')
        var_muon_id_stat_up = np.array([0], dtype='f')
        var_muon_iso_down = np.array([0], dtype='f')
        var_muon_iso_up = np.array([0], dtype='f')
        var_muon_iso_syst_down = np.array([0], dtype='f')
        var_muon_iso_syst_up = np.array([0], dtype='f')
        var_muon_iso_stat_down = np.array([0], dtype='f')
        var_muon_iso_stat_up = np.array([0], dtype='f')
        var_btag_muon_id_down = np.array([0], dtype='f')
        var_btag_muon_id_up = np.array([0], dtype='f')
        var_btag_muon_id_syst_down = np.array([0], dtype='f')
        var_btag_muon_id_syst_up = np.array([0], dtype='f')
        var_btag_muon_id_stat_down = np.array([0], dtype='f')
        var_btag_muon_id_stat_up = np.array([0], dtype='f')
        var_btag_muon_iso_down = np.array([0], dtype='f')
        var_btag_muon_iso_up = np.array([0], dtype='f')
        var_btag_muon_iso_syst_down = np.array([0], dtype='f')
        var_btag_muon_iso_syst_up = np.array([0], dtype='f')
        var_btag_muon_iso_stat_down = np.array([0], dtype='f')
        var_btag_muon_iso_stat_up = np.array([0], dtype='f')


        var_pu_down = np.array([0], dtype='f')
        var_pu_up = np.array([0], dtype='f')
        var_btag_pu_down = np.array([0], dtype='f')
        var_btag_pu_up = np.array([0], dtype='f')

# TRG does not affect b-tag        
        var_trig_down = np.array([0], dtype='f')
        var_trig_up = np.array([0], dtype='f')
        var_btag_trig_down = np.array([0], dtype='f')
        var_btag_trig_up = np.array([0], dtype='f')
        
        
        var_btag_down = np.array([0], dtype='f')
        var_btag_up = np.array([0], dtype='f')
        var_btag_ljet_down = np.array([0], dtype='f')
        var_btag_ljet_up = np.array([0], dtype='f')
        var_btag_statistic_down = np.array([0], dtype='f')
        var_btag_statistic_up = np.array([0], dtype='f')
        var_btag_pileup_down = np.array([0], dtype='f')
        var_btag_pileup_up = np.array([0], dtype='f')
        var_btag_jes_down = np.array([0], dtype='f')
        var_btag_jes_up = np.array([0], dtype='f')
        var_btag_type3_down = np.array([0], dtype='f')
        var_btag_type3_up = np.array([0], dtype='f')
        
        var_l1prefiring_down = np.array([0], dtype='f')
        var_l1prefiring_up = np.array([0], dtype='f')
        var_btag_l1prefiring_down = np.array([0], dtype='f')
        var_btag_l1prefiring_up = np.array([0], dtype='f')
        
        
        
        var_merenscale_down = np.array([0], dtype='f')
        var_merenscale_up = np.array([0], dtype='f')
        var_mefacscale_down = np.array([0], dtype='f')
        var_mefacscale_up = np.array([0], dtype='f')
        var_mescale_down = np.array([0], dtype='f')
        var_mescale_up = np.array([0], dtype='f')
        var_btag_merenscale_down = np.array([0], dtype='f')
        var_btag_merenscale_up = np.array([0], dtype='f')
        var_btag_mefacscale_down = np.array([0], dtype='f')
        var_btag_mefacscale_up = np.array([0], dtype='f')
        var_btag_mescale_down = np.array([0], dtype='f')
        var_btag_mescale_up = np.array([0], dtype='f')

        var_bsemilep_down = np.array([0], dtype='f')
        var_bsemilep_up = np.array([0], dtype='f')
        var_btag_bsemilep_down = np.array([0], dtype='f')
        var_btag_bsemilep_up = np.array([0], dtype='f')
        var_pdf_alphas_down = np.array([0], dtype='f')
        var_pdf_alphas_up = np.array([0], dtype='f')
        var_btag_pdf_alphas_down = np.array([0], dtype='f')
        var_btag_pdf_alphas_up = np.array([0], dtype='f')

        var_ml_hdamp_down = np.array([0], dtype='f')
        var_ml_hdamp_up = np.array([0], dtype='f')
        var_btag_ml_hdamp_down = np.array([0], dtype='f')
        var_btag_ml_hdamp_up = np.array([0], dtype='f')

        var_bfrag_down = np.array([0], dtype='f')
        var_bfrag_up = np.array([0], dtype='f')
        var_btag_bfrag_down = np.array([0], dtype='f')
        var_btag_bfrag_up = np.array([0], dtype='f')
        var_bfrag_central = np.array([0], dtype='f')
        var_btag_bfrag_central = np.array([0], dtype='f')

        var_pdf = np.array([0]*103, dtype='f')

        
        var_psscale_weight_isr_up = np.array([0], dtype='f')
        var_psscale_weight_isr_down = np.array([0], dtype='f')
        var_btag_psscale_weight_isr_up = np.array([0], dtype='f')
        var_btag_psscale_weight_isr_down = np.array([0], dtype='f')
        var_psscale_weight_fsr_up = np.array([0], dtype='f')
        var_psscale_weight_fsr_down = np.array([0], dtype='f')
        var_btag_psscale_weight_fsr_up = np.array([0], dtype='f')
        var_btag_psscale_weight_fsr_down = np.array([0], dtype='f')
        #var_psscale_weight_fsr_G2GG_muR_up = np.array([0], dtype='f')
        #var_psscale_weight_fsr_G2GG_muR_down = np.array([0], dtype='f')
        #var_btag_psscale_weight_fsr_G2GG_muR_up = np.array([0], dtype='f')
        #var_btag_psscale_weight_fsr_G2GG_muR_down = np.array([0], dtype='f')
        #var_psscale_weight_fsr_G2QQ_muR_up = np.array([0], dtype='f')
        #var_psscale_weight_fsr_G2QQ_muR_down = np.array([0], dtype='f')
        #var_btag_psscale_weight_fsr_G2QQ_muR_up = np.array([0], dtype='f')
        #var_btag_psscale_weight_fsr_G2QQ_muR_down = np.array([0], dtype='f')
        #var_psscale_weight_fsr_Q2QG_muR_up = np.array([0], dtype='f')
        #var_psscale_weight_fsr_Q2QG_muR_down = np.array([0], dtype='f')
        #var_btag_psscale_weight_fsr_Q2QG_muR_up = np.array([0], dtype='f')
        #var_btag_psscale_weight_fsr_Q2QG_muR_down = np.array([0], dtype='f')
        #var_psscale_weight_fsr_X2XG_muR_up = np.array([0], dtype='f')
        #var_psscale_weight_fsr_X2XG_muR_down = np.array([0], dtype='f')
        #var_btag_psscale_weight_fsr_X2XG_muR_up = np.array([0], dtype='f')
        #var_btag_psscale_weight_fsr_X2XG_muR_down = np.array([0], dtype='f')
        #var_psscale_weight_fsr_G2GG_cNS_up = np.array([0], dtype='f')
        #var_psscale_weight_fsr_G2GG_cNS_down = np.array([0], dtype='f')
        #var_btag_psscale_weight_fsr_G2GG_cNS_up = np.array([0], dtype='f')
        #var_btag_psscale_weight_fsr_G2GG_cNS_down = np.array([0], dtype='f')
        #var_psscale_weight_fsr_G2QQ_cNS_up = np.array([0], dtype='f')
        #var_psscale_weight_fsr_G2QQ_cNS_down = np.array([0], dtype='f')
        #var_btag_psscale_weight_fsr_G2QQ_cNS_up = np.array([0], dtype='f')
        #var_btag_psscale_weight_fsr_G2QQ_cNS_down = np.array([0], dtype='f')
        #var_psscale_weight_fsr_Q2QG_cNS_up = np.array([0], dtype='f')
        #var_psscale_weight_fsr_Q2QG_cNS_down = np.array([0], dtype='f')
        #var_btag_psscale_weight_fsr_Q2QG_cNS_up = np.array([0], dtype='f')
        #var_btag_psscale_weight_fsr_Q2QG_cNS_down = np.array([0], dtype='f')
        #var_psscale_weight_fsr_X2XG_cNS_up = np.array([0], dtype='f')
        #var_psscale_weight_fsr_X2XG_cNS_down = np.array([0], dtype='f')
        #var_btag_psscale_weight_fsr_X2XG_cNS_up = np.array([0], dtype='f')
        #var_btag_psscale_weight_fsr_X2XG_cNS_down = np.array([0], dtype='f')

        #var_psscale_weight_isr_G2GG_muR_up=np.array([0], dtype='f')
        #var_psscale_weight_isr_G2GG_muR_down=np.array([0], dtype='f')
        #var_btag_psscale_weight_isr_G2GG_muR_up=np.array([0], dtype='f')
        #var_btag_psscale_weight_isr_G2GG_muR_down=np.array([0], dtype='f')
        #var_psscale_weight_isr_G2QQ_muR_up=np.array([0], dtype='f')
        #var_psscale_weight_isr_G2QQ_muR_down=np.array([0], dtype='f')
        #var_btag_psscale_weight_isr_G2QQ_muR_up=np.array([0], dtype='f')
        #var_btag_psscale_weight_isr_G2QQ_muR_down=np.array([0], dtype='f')
        #var_psscale_weight_isr_Q2QG_muR_up=np.array([0], dtype='f')
        #var_psscale_weight_isr_Q2QG_muR_down=np.array([0], dtype='f')
        #var_btag_psscale_weight_isr_Q2QG_muR_up=np.array([0], dtype='f')
        #var_btag_psscale_weight_isr_Q2QG_muR_down=np.array([0], dtype='f')
        #var_psscale_weight_isr_X2XG_muR_up=np.array([0], dtype='f')
        #var_psscale_weight_isr_X2XG_muR_down=np.array([0], dtype='f')
        #var_btag_psscale_weight_isr_X2XG_muR_up=np.array([0], dtype='f')
        #var_btag_psscale_weight_isr_X2XG_muR_down=np.array([0], dtype='f')
        #var_psscale_weight_isr_G2GG_cNS_up=np.array([0], dtype='f')
        #var_psscale_weight_isr_G2GG_cNS_down=np.array([0], dtype='f')
        #var_btag_psscale_weight_isr_G2GG_cNS_up=np.array([0], dtype='f')
        #var_btag_psscale_weight_isr_G2GG_cNS_down=np.array([0], dtype='f')
        #var_psscale_weight_isr_G2QQ_cNS_up=np.array([0], dtype='f')
        #var_psscale_weight_isr_G2QQ_cNS_down=np.array([0], dtype='f')
        #var_btag_psscale_weight_isr_G2QQ_cNS_up=np.array([0], dtype='f')
        #var_btag_psscale_weight_isr_G2QQ_cNS_down=np.array([0], dtype='f')
        #var_psscale_weight_isr_Q2QG_cNS_up=np.array([0], dtype='f')
        #var_psscale_weight_isr_Q2QG_cNS_down=np.array([0], dtype='f')
        #var_btag_psscale_weight_isr_Q2QG_cNS_up=np.array([0], dtype='f')
        #var_btag_psscale_weight_isr_Q2QG_cNS_down=np.array([0], dtype='f')
        #var_psscale_weight_isr_X2XG_cNS_up=np.array([0], dtype='f')
        #var_psscale_weight_isr_X2XG_cNS_down=np.array([0], dtype='f')
        #var_btag_psscale_weight_isr_X2XG_cNS_up=np.array([0], dtype='f')
        #var_btag_psscale_weight_isr_X2XG_cNS_down=np.array([0], dtype='f')

        var_bfrag_peterson = np.array([0], dtype='f')
        var_btag_bfrag_peterson = np.array([0], dtype='f')

        var_top_pt = np.array([0], dtype='f')

        tree.SetBranchAddress("weight", weight)
        tree.SetBranchAddress("leptonSF", leptonSF)
        tree.SetBranchAddress("triggerSF", triggerSF)
        tree.SetBranchAddress("btagSF", btagSF)
        tree.SetBranchAddress("pileupSF", pileupSF)
        tree.SetBranchAddress("l1PrefiringWeight", prefiringWeight)

        tree.SetBranchAddress("var_ELE_ID_DOWN", var_ele_id_down)
        tree.SetBranchAddress("var_ELE_ID_UP", var_ele_id_up)
        tree.SetBranchAddress("var_BTAG_ELE_ID_DOWN", var_btag_ele_id_down)
        tree.SetBranchAddress("var_BTAG_ELE_ID_UP", var_btag_ele_id_up)
        tree.SetBranchAddress("var_ELE_RECO_DOWN", var_ele_reco_down)
        tree.SetBranchAddress("var_ELE_RECO_UP", var_ele_reco_up)
        tree.SetBranchAddress("var_BTAG_ELE_RECO_DOWN", var_btag_ele_reco_down)
        tree.SetBranchAddress("var_BTAG_ELE_RECO_UP", var_btag_ele_reco_up)

        tree.SetBranchAddress("var_MUON_ID_DOWN", var_muon_id_down)
        tree.SetBranchAddress("var_MUON_ID_UP", var_muon_id_up)
        tree.SetBranchAddress("var_MUON_ID_SYST_DOWN", var_muon_id_syst_down)
        tree.SetBranchAddress("var_MUON_ID_SYST_UP", var_muon_id_syst_up)
        tree.SetBranchAddress("var_MUON_ID_STAT_DOWN", var_muon_id_stat_down)
        tree.SetBranchAddress("var_MUON_ID_STAT_UP", var_muon_id_stat_up)
        tree.SetBranchAddress("var_MUON_ISO_DOWN", var_muon_iso_down)
        tree.SetBranchAddress("var_MUON_ISO_UP", var_muon_iso_up)
        tree.SetBranchAddress("var_MUON_ISO_SYST_DOWN", var_muon_iso_syst_down)
        tree.SetBranchAddress("var_MUON_ISO_SYST_UP", var_muon_iso_syst_up)
        tree.SetBranchAddress("var_MUON_ISO_STAT_DOWN", var_muon_iso_stat_down)
        tree.SetBranchAddress("var_MUON_ISO_STAT_UP", var_muon_iso_stat_up)
        
        tree.SetBranchAddress("var_BTAG_MUON_ID_DOWN", var_btag_muon_id_down)
        tree.SetBranchAddress("var_BTAG_MUON_ID_UP", var_btag_muon_id_up)
        tree.SetBranchAddress("var_BTAG_MUON_ID_SYST_DOWN", var_btag_muon_id_syst_down)
        tree.SetBranchAddress("var_BTAG_MUON_ID_SYST_UP", var_btag_muon_id_syst_up)
        tree.SetBranchAddress("var_BTAG_MUON_ID_STAT_DOWN", var_btag_muon_id_stat_down)
        tree.SetBranchAddress("var_BTAG_MUON_ID_STAT_UP", var_btag_muon_id_stat_up)
        tree.SetBranchAddress("var_BTAG_MUON_ISO_DOWN", var_btag_muon_iso_down)
        tree.SetBranchAddress("var_BTAG_MUON_ISO_UP", var_btag_muon_iso_up)
        tree.SetBranchAddress("var_BTAG_MUON_ISO_SYST_DOWN", var_btag_muon_iso_syst_down)
        tree.SetBranchAddress("var_BTAG_MUON_ISO_SYST_UP", var_btag_muon_iso_syst_up)
        tree.SetBranchAddress("var_BTAG_MUON_ISO_STAT_DOWN", var_btag_muon_iso_stat_down)
        tree.SetBranchAddress("var_BTAG_MUON_ISO_STAT_UP", var_btag_muon_iso_stat_up)
        
        tree.SetBranchAddress("var_PU_DOWN", var_pu_down)
        tree.SetBranchAddress("var_PU_UP", var_pu_up)
        tree.SetBranchAddress("var_BTAG_PU_DOWN", var_btag_pu_down)
        tree.SetBranchAddress("var_BTAG_PU_UP", var_btag_pu_up)
        
        tree.SetBranchAddress("var_TRIG_DOWN", var_trig_down)
        tree.SetBranchAddress("var_TRIG_UP", var_trig_up)
        tree.SetBranchAddress("var_BTAG_TRIG_DOWN", var_btag_trig_down)
        tree.SetBranchAddress("var_BTAG_TRIG_UP", var_btag_trig_up)

        tree.SetBranchAddress("var_BTAG_DOWN", var_btag_down)
        tree.SetBranchAddress("var_BTAG_UP", var_btag_up)
        tree.SetBranchAddress("var_BTAG_LJET_DOWN", var_btag_ljet_down)
        tree.SetBranchAddress("var_BTAG_LJET_UP", var_btag_ljet_up)
        tree.SetBranchAddress("var_BTAG_STATISTIC_DOWN", var_btag_statistic_down)
        tree.SetBranchAddress("var_BTAG_STATISTIC_UP", var_btag_statistic_up)
        tree.SetBranchAddress("var_BTAG_PILEUP_DOWN", var_btag_pileup_down)
        tree.SetBranchAddress("var_BTAG_PILEUP_UP", var_btag_pileup_up)
        tree.SetBranchAddress("var_BTAG_JES_DOWN", var_btag_jes_down)
        tree.SetBranchAddress("var_BTAG_JES_UP", var_btag_jes_up)
        tree.SetBranchAddress("var_BTAG_TYPE3_DOWN", var_btag_type3_down)
        tree.SetBranchAddress("var_BTAG_TYPE3_UP", var_btag_type3_up)
        
        tree.SetBranchAddress("var_L1PREFIRING_DOWN", var_l1prefiring_down)
        tree.SetBranchAddress("var_L1PREFIRING_UP", var_l1prefiring_up)
        tree.SetBranchAddress("var_BTAG_L1PREFIRING_DOWN", var_btag_l1prefiring_down)
        tree.SetBranchAddress("var_BTAG_L1PREFIRING_UP", var_btag_l1prefiring_up)
        
        tree.SetBranchAddress("var_MERENSCALE_DOWN", var_merenscale_down)
        tree.SetBranchAddress("var_MERENSCALE_UP", var_merenscale_up)
        tree.SetBranchAddress("var_MEFACSCALE_DOWN", var_mefacscale_down)
        tree.SetBranchAddress("var_MEFACSCALE_UP", var_mefacscale_up)
        tree.SetBranchAddress("var_MESCALE_DOWN", var_mescale_down)
        tree.SetBranchAddress("var_MESCALE_UP", var_mescale_up)

        tree.SetBranchAddress("var_BTAG_MERENSCALE_DOWN", var_btag_merenscale_down)
        tree.SetBranchAddress("var_BTAG_MERENSCALE_UP", var_btag_merenscale_up)
        tree.SetBranchAddress("var_BTAG_MEFACSCALE_DOWN", var_btag_mefacscale_down)
        tree.SetBranchAddress("var_BTAG_MEFACSCALE_UP", var_btag_mefacscale_up)
        tree.SetBranchAddress("var_BTAG_MESCALE_DOWN", var_btag_mescale_down)
        tree.SetBranchAddress("var_BTAG_MESCALE_UP", var_btag_mescale_up)

        tree.SetBranchAddress("var_BSEMILEP_DOWN", var_bsemilep_down)
        tree.SetBranchAddress("var_BSEMILEP_UP", var_bsemilep_up)
        tree.SetBranchAddress("var_BTAG_BSEMILEP_DOWN", var_btag_bsemilep_down)
        tree.SetBranchAddress("var_BTAG_BSEMILEP_UP", var_btag_bsemilep_up)

        tree.SetBranchAddress("var_PDF_ALPHAS_DOWN", var_pdf_alphas_down)
        tree.SetBranchAddress("var_PDF_ALPHAS_UP", var_pdf_alphas_up)
        tree.SetBranchAddress("var_BTAG_PDF_ALPHAS_DOWN", var_btag_pdf_alphas_down)
        tree.SetBranchAddress("var_BTAG_PDF_ALPHAS_UP", var_btag_pdf_alphas_up)

        tree.SetBranchAddress("var_ML_HDAMP_DOWN", var_ml_hdamp_down)
        tree.SetBranchAddress("var_ML_HDAMP_UP", var_ml_hdamp_up)
        tree.SetBranchAddress("var_BTAG_ML_HDAMP_DOWN", var_btag_ml_hdamp_down)
        tree.SetBranchAddress("var_BTAG_ML_HDAMP_UP", var_btag_ml_hdamp_up)

        tree.SetBranchAddress("var_BFRAG_DOWN", var_bfrag_down)
        tree.SetBranchAddress("var_BFRAG_UP", var_bfrag_up)
        tree.SetBranchAddress("var_BTAG_BFRAG_DOWN", var_btag_bfrag_down)
        tree.SetBranchAddress("var_BTAG_BFRAG_UP", var_btag_bfrag_up)
        tree.SetBranchAddress("var_BFRAG_CENTRAL", var_bfrag_central)
        tree.SetBranchAddress("var_BTAG_BFRAG_CENTRAL", var_btag_bfrag_central)

        tree.SetBranchAddress("var_PSSCALE_WEIGHT_ISR_UP", var_psscale_weight_isr_up)
        tree.SetBranchAddress("var_PSSCALE_WEIGHT_ISR_DOWN", var_psscale_weight_isr_down)
        tree.SetBranchAddress("var_BTAG_PSSCALE_WEIGHT_ISR_UP", var_btag_psscale_weight_isr_up)
        tree.SetBranchAddress("var_BTAG_PSSCALE_WEIGHT_ISR_DOWN", var_btag_psscale_weight_isr_down)
        tree.SetBranchAddress("var_PSSCALE_WEIGHT_FSR_UP", var_psscale_weight_fsr_up)
        tree.SetBranchAddress("var_PSSCALE_WEIGHT_FSR_DOWN", var_psscale_weight_fsr_down)
        tree.SetBranchAddress("var_BTAG_PSSCALE_WEIGHT_FSR_UP", var_btag_psscale_weight_fsr_up)
        tree.SetBranchAddress("var_BTAG_PSSCALE_WEIGHT_FSR_DOWN", var_btag_psscale_weight_fsr_down)

        #tree.SetBranchAddress("var_PSSCALE_WEIGHT_fsr_G2GG_muR_up", var_psscale_weight_fsr_G2GG_muR_up)
        #tree.SetBranchAddress("var_PSSCALE_WEIGHT_fsr_G2GG_muR_dn", var_psscale_weight_fsr_G2GG_muR_down)
        #tree.SetBranchAddress("var_BTAG_PSSCALE_WEIGHT_fsr_G2GG_muR_up", var_btag_psscale_weight_fsr_G2GG_muR_up)
        #tree.SetBranchAddress("var_BTAG_PSSCALE_WEIGHT_fsr_G2GG_muR_dn", var_btag_psscale_weight_fsr_G2GG_muR_down)
        #tree.SetBranchAddress("var_PSSCALE_WEIGHT_fsr_G2QQ_muR_up", var_psscale_weight_fsr_G2QQ_muR_up)
        #tree.SetBranchAddress("var_PSSCALE_WEIGHT_fsr_G2QQ_muR_dn", var_psscale_weight_fsr_G2QQ_muR_down)
        #tree.SetBranchAddress("var_BTAG_PSSCALE_WEIGHT_fsr_G2QQ_muR_up", var_btag_psscale_weight_fsr_G2QQ_muR_up)
        #tree.SetBranchAddress("var_BTAG_PSSCALE_WEIGHT_fsr_G2QQ_muR_dn", var_btag_psscale_weight_fsr_G2QQ_muR_down)
        #tree.SetBranchAddress("var_PSSCALE_WEIGHT_fsr_Q2QG_muR_up", var_psscale_weight_fsr_Q2QG_muR_up)
        #tree.SetBranchAddress("var_PSSCALE_WEIGHT_fsr_Q2QG_muR_dn", var_psscale_weight_fsr_Q2QG_muR_down)
        #tree.SetBranchAddress("var_BTAG_PSSCALE_WEIGHT_fsr_Q2QG_muR_up", var_btag_psscale_weight_fsr_Q2QG_muR_up)
        #tree.SetBranchAddress("var_BTAG_PSSCALE_WEIGHT_fsr_Q2QG_muR_dn", var_btag_psscale_weight_fsr_Q2QG_muR_down)
        #tree.SetBranchAddress("var_PSSCALE_WEIGHT_fsr_X2XG_muR_up", var_psscale_weight_fsr_X2XG_muR_up)
        #tree.SetBranchAddress("var_PSSCALE_WEIGHT_fsr_X2XG_muR_dn", var_psscale_weight_fsr_X2XG_muR_down)
        #tree.SetBranchAddress("var_BTAG_PSSCALE_WEIGHT_fsr_X2XG_muR_up", var_btag_psscale_weight_fsr_X2XG_muR_up)
        #tree.SetBranchAddress("var_BTAG_PSSCALE_WEIGHT_fsr_X2XG_muR_dn", var_btag_psscale_weight_fsr_X2XG_muR_down)
        #tree.SetBranchAddress("var_PSSCALE_WEIGHT_fsr_G2GG_cNS_up", var_psscale_weight_fsr_G2GG_cNS_up)
        #tree.SetBranchAddress("var_PSSCALE_WEIGHT_fsr_G2GG_cNS_dn", var_psscale_weight_fsr_G2GG_cNS_down)
        #tree.SetBranchAddress("var_BTAG_PSSCALE_WEIGHT_fsr_G2GG_cNS_up", var_btag_psscale_weight_fsr_G2GG_cNS_up)
        #tree.SetBranchAddress("var_BTAG_PSSCALE_WEIGHT_fsr_G2GG_cNS_dn", var_btag_psscale_weight_fsr_G2GG_cNS_down)
        #tree.SetBranchAddress("var_PSSCALE_WEIGHT_fsr_G2QQ_cNS_up", var_psscale_weight_fsr_G2QQ_cNS_up)
        #tree.SetBranchAddress("var_PSSCALE_WEIGHT_fsr_G2QQ_cNS_dn", var_psscale_weight_fsr_G2QQ_cNS_down)
        #tree.SetBranchAddress("var_BTAG_PSSCALE_WEIGHT_fsr_G2QQ_cNS_up", var_btag_psscale_weight_fsr_G2QQ_cNS_up)
        #tree.SetBranchAddress("var_BTAG_PSSCALE_WEIGHT_fsr_G2QQ_cNS_dn", var_btag_psscale_weight_fsr_G2QQ_cNS_down)
        #tree.SetBranchAddress("var_PSSCALE_WEIGHT_fsr_Q2QG_cNS_up", var_psscale_weight_fsr_Q2QG_cNS_up)
        #tree.SetBranchAddress("var_PSSCALE_WEIGHT_fsr_Q2QG_cNS_dn", var_psscale_weight_fsr_Q2QG_cNS_down)
        #tree.SetBranchAddress("var_BTAG_PSSCALE_WEIGHT_fsr_Q2QG_cNS_up", var_btag_psscale_weight_fsr_Q2QG_cNS_up)
        #tree.SetBranchAddress("var_BTAG_PSSCALE_WEIGHT_fsr_Q2QG_cNS_dn", var_btag_psscale_weight_fsr_Q2QG_cNS_down)
        #tree.SetBranchAddress("var_PSSCALE_WEIGHT_fsr_X2XG_cNS_up", var_psscale_weight_fsr_X2XG_cNS_up)
        #tree.SetBranchAddress("var_PSSCALE_WEIGHT_fsr_X2XG_cNS_dn", var_psscale_weight_fsr_X2XG_cNS_down)
        #tree.SetBranchAddress("var_BTAG_PSSCALE_WEIGHT_fsr_X2XG_cNS_up", var_btag_psscale_weight_fsr_X2XG_cNS_up)
        #tree.SetBranchAddress("var_BTAG_PSSCALE_WEIGHT_fsr_X2XG_cNS_dn", var_btag_psscale_weight_fsr_X2XG_cNS_down)

        #tree.SetBranchAddress("var_PSSCALE_WEIGHT_isr_G2GG_muR_up", var_psscale_weight_isr_G2GG_muR_up)
        #tree.SetBranchAddress("var_PSSCALE_WEIGHT_isr_G2GG_muR_dn", var_psscale_weight_isr_G2GG_muR_down)
        #tree.SetBranchAddress("var_BTAG_PSSCALE_WEIGHT_isr_G2GG_muR_up", var_btag_psscale_weight_isr_G2GG_muR_up)
        #tree.SetBranchAddress("var_BTAG_PSSCALE_WEIGHT_isr_G2GG_muR_dn", var_btag_psscale_weight_isr_G2GG_muR_down)
        #tree.SetBranchAddress("var_PSSCALE_WEIGHT_isr_G2QQ_muR_up", var_psscale_weight_isr_G2QQ_muR_up)
        #tree.SetBranchAddress("var_PSSCALE_WEIGHT_isr_G2QQ_muR_dn", var_psscale_weight_isr_G2QQ_muR_down)
        #tree.SetBranchAddress("var_BTAG_PSSCALE_WEIGHT_isr_G2QQ_muR_up", var_btag_psscale_weight_isr_G2QQ_muR_up)
        #tree.SetBranchAddress("var_BTAG_PSSCALE_WEIGHT_isr_G2QQ_muR_dn", var_btag_psscale_weight_isr_G2QQ_muR_down)
        #tree.SetBranchAddress("var_PSSCALE_WEIGHT_isr_Q2QG_muR_up", var_psscale_weight_isr_Q2QG_muR_up)
        #tree.SetBranchAddress("var_PSSCALE_WEIGHT_isr_Q2QG_muR_dn", var_psscale_weight_isr_Q2QG_muR_down)
        #tree.SetBranchAddress("var_BTAG_PSSCALE_WEIGHT_isr_Q2QG_muR_up", var_btag_psscale_weight_isr_Q2QG_muR_up)
        #tree.SetBranchAddress("var_BTAG_PSSCALE_WEIGHT_isr_Q2QG_muR_dn", var_btag_psscale_weight_isr_Q2QG_muR_down)
        #tree.SetBranchAddress("var_PSSCALE_WEIGHT_isr_X2XG_muR_up", var_psscale_weight_isr_X2XG_muR_up)
        #tree.SetBranchAddress("var_PSSCALE_WEIGHT_isr_X2XG_muR_dn", var_psscale_weight_isr_X2XG_muR_down)
        #tree.SetBranchAddress("var_BTAG_PSSCALE_WEIGHT_isr_X2XG_muR_up", var_btag_psscale_weight_isr_X2XG_muR_up)
        #tree.SetBranchAddress("var_BTAG_PSSCALE_WEIGHT_isr_X2XG_muR_dn", var_btag_psscale_weight_isr_X2XG_muR_down)
        #tree.SetBranchAddress("var_PSSCALE_WEIGHT_isr_G2GG_cNS_up", var_psscale_weight_isr_G2GG_cNS_up)
        #tree.SetBranchAddress("var_PSSCALE_WEIGHT_isr_G2GG_cNS_dn", var_psscale_weight_isr_G2GG_cNS_down)
        #tree.SetBranchAddress("var_BTAG_PSSCALE_WEIGHT_isr_G2GG_cNS_up", var_btag_psscale_weight_isr_G2GG_cNS_up)
        #tree.SetBranchAddress("var_BTAG_PSSCALE_WEIGHT_isr_G2GG_cNS_dn", var_btag_psscale_weight_isr_G2GG_cNS_down)
        #tree.SetBranchAddress("var_PSSCALE_WEIGHT_isr_G2QQ_cNS_up", var_psscale_weight_isr_G2QQ_cNS_up)
        #tree.SetBranchAddress("var_PSSCALE_WEIGHT_isr_G2QQ_cNS_dn", var_psscale_weight_isr_G2QQ_cNS_down)
        #tree.SetBranchAddress("var_BTAG_PSSCALE_WEIGHT_isr_G2QQ_cNS_up", var_btag_psscale_weight_isr_G2QQ_cNS_up)
        #tree.SetBranchAddress("var_BTAG_PSSCALE_WEIGHT_isr_G2QQ_cNS_dn", var_btag_psscale_weight_isr_G2QQ_cNS_down)
        #tree.SetBranchAddress("var_PSSCALE_WEIGHT_isr_Q2QG_cNS_up", var_psscale_weight_isr_Q2QG_cNS_up)
        #tree.SetBranchAddress("var_PSSCALE_WEIGHT_isr_Q2QG_cNS_dn", var_psscale_weight_isr_Q2QG_cNS_down)
        #tree.SetBranchAddress("var_BTAG_PSSCALE_WEIGHT_isr_Q2QG_cNS_up", var_btag_psscale_weight_isr_Q2QG_cNS_up)
        #tree.SetBranchAddress("var_BTAG_PSSCALE_WEIGHT_isr_Q2QG_cNS_dn", var_btag_psscale_weight_isr_Q2QG_cNS_down)
        #tree.SetBranchAddress("var_PSSCALE_WEIGHT_isr_X2XG_cNS_up", var_psscale_weight_isr_X2XG_cNS_up)
        #tree.SetBranchAddress("var_PSSCALE_WEIGHT_isr_X2XG_cNS_dn", var_psscale_weight_isr_X2XG_cNS_down)
        #tree.SetBranchAddress("var_BTAG_PSSCALE_WEIGHT_isr_X2XG_cNS_up", var_btag_psscale_weight_isr_X2XG_cNS_up)
        #tree.SetBranchAddress("var_BTAG_PSSCALE_WEIGHT_isr_X2XG_cNS_dn", var_btag_psscale_weight_isr_X2XG_cNS_down)
        
        tree.SetBranchAddress("var_PDF", var_pdf)

        tree.SetBranchAddress("var_BFRAG_PETERSON", var_bfrag_peterson)
        tree.SetBranchAddress("var_BTAG_BFRAG_PETERSON", var_btag_bfrag_peterson)

        tree.SetBranchAddress("var_TOP_PT", var_top_pt)
        
# Weights and SF
        maxEntries = tree.GetEntries() 
        bigNumber = 20000
        maxForBar = int(maxEntries/bigNumber)

        bar = IncrementalBar('Processing', max=maxForBar, suffix='%(percent).1f%% - %(eta)ds')
        for i in range(tree.GetEntries()):
            tree.GetEntry(i)
            if(i % bigNumber==0):
                bar.next()
            weightsSFVar = []
            weightsSFVar.append(weight[0])        
            weightsSFVar.append(leptonSF[0])
            weightsSFVar.append(triggerSF[0])
            weightsSFVar.append(btagSF[0])
            weightsSFVar.append(pileupSF[0])
            weightsSFVar.append(prefiringWeight[0])
            
            weightsSFVar.append(var_ele_id_down[0])
            weightsSFVar.append(var_ele_id_up[0])
            weightsSFVar.append(var_btag_ele_id_down[0])
            weightsSFVar.append(var_btag_ele_id_up[0])
            weightsSFVar.append(var_ele_reco_down[0])
            weightsSFVar.append(var_ele_reco_up[0])
            weightsSFVar.append(var_btag_ele_reco_down[0])
            weightsSFVar.append(var_btag_ele_reco_up[0])

            weightsSFVar.append(var_muon_id_down[0])
            weightsSFVar.append(var_muon_id_up[0])
            weightsSFVar.append(var_btag_muon_id_down[0])
            weightsSFVar.append(var_btag_muon_id_up[0])
            weightsSFVar.append(var_muon_id_syst_down[0])
            weightsSFVar.append(var_muon_id_syst_up[0])
            weightsSFVar.append(var_btag_muon_id_syst_down[0])
            weightsSFVar.append(var_btag_muon_id_syst_up[0])
            weightsSFVar.append(var_muon_id_stat_down[0])
            weightsSFVar.append(var_muon_id_stat_up[0])
            weightsSFVar.append(var_btag_muon_id_stat_down[0])
            weightsSFVar.append(var_btag_muon_id_stat_up[0])
            weightsSFVar.append(var_muon_iso_down[0])
            weightsSFVar.append(var_muon_iso_up[0])
            weightsSFVar.append(var_btag_muon_iso_down[0])
            weightsSFVar.append(var_btag_muon_iso_up[0])
            weightsSFVar.append(var_muon_iso_syst_down[0])
            weightsSFVar.append(var_muon_iso_syst_up[0])
            weightsSFVar.append(var_btag_muon_iso_syst_down[0])
            weightsSFVar.append(var_btag_muon_iso_syst_up[0])
            weightsSFVar.append(var_muon_iso_stat_down[0])
            weightsSFVar.append(var_muon_iso_stat_up[0])
            weightsSFVar.append(var_btag_muon_iso_stat_down[0])
            weightsSFVar.append(var_btag_muon_iso_stat_up[0])

            weightsSFVar.append(var_pu_down[0])
            weightsSFVar.append(var_pu_up[0])
            weightsSFVar.append(var_btag_pu_down[0])
            weightsSFVar.append(var_btag_pu_up[0])

            weightsSFVar.append(var_trig_down[0])
            weightsSFVar.append(var_trig_up[0])
            weightsSFVar.append(var_btag_trig_down[0])
            weightsSFVar.append(var_btag_trig_up[0])

            weightsSFVar.append(var_btag_down[0])
            weightsSFVar.append(var_btag_up[0])
            weightsSFVar.append(var_btag_ljet_down[0])
            weightsSFVar.append(var_btag_ljet_up[0])
            weightsSFVar.append(var_btag_statistic_down[0])
            weightsSFVar.append(var_btag_statistic_up[0])
            weightsSFVar.append(var_btag_pileup_down[0])
            weightsSFVar.append(var_btag_pileup_up[0])
            weightsSFVar.append(var_btag_jes_down[0])
            weightsSFVar.append(var_btag_jes_up[0])
            weightsSFVar.append(var_btag_type3_down[0])
            weightsSFVar.append(var_btag_type3_up[0])

            weightsSFVar.append(var_l1prefiring_down[0])
            weightsSFVar.append(var_l1prefiring_up[0])
            weightsSFVar.append(var_btag_l1prefiring_down[0])
            weightsSFVar.append(var_btag_l1prefiring_up[0])
            
            weightsSFVar.append(var_merenscale_down[0])
            weightsSFVar.append(var_merenscale_up[0])
            weightsSFVar.append(var_mefacscale_down[0])
            weightsSFVar.append(var_mefacscale_up[0])
            weightsSFVar.append(var_mescale_down[0])
            weightsSFVar.append(var_mescale_up[0])

            weightsSFVar.append(var_btag_merenscale_down[0])
            weightsSFVar.append(var_btag_merenscale_up[0])
            weightsSFVar.append(var_btag_mefacscale_down[0])
            weightsSFVar.append(var_btag_mefacscale_up[0])
            weightsSFVar.append(var_btag_mescale_down[0])
            weightsSFVar.append(var_btag_mescale_up[0])

            weightsSFVar.append(var_bsemilep_down[0])
            weightsSFVar.append(var_bsemilep_up[0])
            weightsSFVar.append(var_btag_bsemilep_down[0])
            weightsSFVar.append(var_btag_bsemilep_up[0])

            weightsSFVar.append(var_pdf_alphas_down[0])
            weightsSFVar.append(var_pdf_alphas_up[0])
            weightsSFVar.append(var_btag_pdf_alphas_down[0])
            weightsSFVar.append(var_btag_pdf_alphas_up[0])

            weightsSFVar.append(var_ml_hdamp_down[0])
            weightsSFVar.append(var_ml_hdamp_up[0])
            weightsSFVar.append(var_btag_ml_hdamp_down[0])
            weightsSFVar.append(var_btag_ml_hdamp_up[0])
            
            weightsSFVar.append(var_bfrag_down[0])
            weightsSFVar.append(var_bfrag_up[0])
            weightsSFVar.append(var_btag_bfrag_down[0])
            weightsSFVar.append(var_btag_bfrag_up[0])
            weightsSFVar.append(var_bfrag_central[0])
            weightsSFVar.append(var_btag_bfrag_central[0])         

            weightsSFVar.append(var_psscale_weight_isr_up[0])
            weightsSFVar.append(var_psscale_weight_isr_down[0])
            weightsSFVar.append(var_btag_psscale_weight_isr_up[0])
            weightsSFVar.append(var_btag_psscale_weight_isr_down[0])
            weightsSFVar.append(var_psscale_weight_fsr_up[0])
            weightsSFVar.append(var_psscale_weight_fsr_down[0])
            weightsSFVar.append(var_btag_psscale_weight_fsr_up[0])
            weightsSFVar.append(var_btag_psscale_weight_fsr_down[0])

            #weightsSFVar.append(var_psscale_weight_fsr_G2GG_muR_up[0])
            #weightsSFVar.append(var_psscale_weight_fsr_G2GG_muR_down[0])
            #weightsSFVar.append(var_btag_psscale_weight_fsr_G2GG_muR_up[0])
            #weightsSFVar.append(var_btag_psscale_weight_fsr_G2GG_muR_down[0])
            #weightsSFVar.append(var_psscale_weight_fsr_G2QQ_muR_up[0])
            #weightsSFVar.append(var_psscale_weight_fsr_G2QQ_muR_down[0])
            #weightsSFVar.append(var_btag_psscale_weight_fsr_G2QQ_muR_up[0])
            #weightsSFVar.append(var_btag_psscale_weight_fsr_G2QQ_muR_down[0])
            #weightsSFVar.append(var_psscale_weight_fsr_Q2QG_muR_up[0])
            #weightsSFVar.append(var_psscale_weight_fsr_Q2QG_muR_down[0])
            #weightsSFVar.append(var_btag_psscale_weight_fsr_Q2QG_muR_up[0])
            #weightsSFVar.append(var_btag_psscale_weight_fsr_Q2QG_muR_down[0])
            #weightsSFVar.append(var_psscale_weight_fsr_X2XG_muR_up[0])
            #weightsSFVar.append(var_psscale_weight_fsr_X2XG_muR_down[0])
            #weightsSFVar.append(var_btag_psscale_weight_fsr_X2XG_muR_up[0])
            #weightsSFVar.append(var_btag_psscale_weight_fsr_X2XG_muR_down[0])
            #weightsSFVar.append(var_psscale_weight_fsr_G2GG_cNS_up[0])
            #weightsSFVar.append(var_psscale_weight_fsr_G2GG_cNS_down[0])
            #weightsSFVar.append(var_btag_psscale_weight_fsr_G2GG_cNS_up[0])
            #weightsSFVar.append(var_btag_psscale_weight_fsr_G2GG_cNS_down[0])
            #weightsSFVar.append(var_psscale_weight_fsr_G2QQ_cNS_up[0])
            #weightsSFVar.append(var_psscale_weight_fsr_G2QQ_cNS_down[0])
            #weightsSFVar.append(var_btag_psscale_weight_fsr_G2QQ_cNS_up[0])
            #weightsSFVar.append(var_btag_psscale_weight_fsr_G2QQ_cNS_down[0])
            #weightsSFVar.append(var_psscale_weight_fsr_Q2QG_cNS_up[0])
            #weightsSFVar.append(var_psscale_weight_fsr_Q2QG_cNS_down[0])
            #weightsSFVar.append(var_btag_psscale_weight_fsr_Q2QG_cNS_up[0])
            #weightsSFVar.append(var_btag_psscale_weight_fsr_Q2QG_cNS_down[0])
            #weightsSFVar.append(var_psscale_weight_fsr_X2XG_cNS_up[0])
            #weightsSFVar.append(var_psscale_weight_fsr_X2XG_cNS_down[0])
            #weightsSFVar.append(var_btag_psscale_weight_fsr_X2XG_cNS_up[0])
            #weightsSFVar.append(var_btag_psscale_weight_fsr_X2XG_cNS_down[0])

            #weightsSFVar.append(var_psscale_weight_isr_G2GG_muR_up[0])
            #weightsSFVar.append(var_psscale_weight_isr_G2GG_muR_down[0])
            #weightsSFVar.append(var_btag_psscale_weight_isr_G2GG_muR_up[0])
            #weightsSFVar.append(var_btag_psscale_weight_isr_G2GG_muR_down[0])
            #weightsSFVar.append(var_psscale_weight_isr_G2QQ_muR_up[0])
            #weightsSFVar.append(var_psscale_weight_isr_G2QQ_muR_down[0])
            #weightsSFVar.append(var_btag_psscale_weight_isr_G2QQ_muR_up[0])
            #weightsSFVar.append(var_btag_psscale_weight_isr_G2QQ_muR_down[0])
            #weightsSFVar.append(var_psscale_weight_isr_Q2QG_muR_up[0])
            #weightsSFVar.append(var_psscale_weight_isr_Q2QG_muR_down[0])
            #weightsSFVar.append(var_btag_psscale_weight_isr_Q2QG_muR_up[0])
            #weightsSFVar.append(var_btag_psscale_weight_isr_Q2QG_muR_down[0])
            #weightsSFVar.append(var_psscale_weight_isr_X2XG_muR_up[0])
            #weightsSFVar.append(var_psscale_weight_isr_X2XG_muR_down[0])
            #weightsSFVar.append(var_btag_psscale_weight_isr_X2XG_muR_up[0])
            #weightsSFVar.append(var_btag_psscale_weight_isr_X2XG_muR_down[0])
            #weightsSFVar.append(var_psscale_weight_isr_G2GG_cNS_up[0])
            #weightsSFVar.append(var_psscale_weight_isr_G2GG_cNS_down[0])
            #weightsSFVar.append(var_btag_psscale_weight_isr_G2GG_cNS_up[0])
            #weightsSFVar.append(var_btag_psscale_weight_isr_G2GG_cNS_down[0])
            #weightsSFVar.append(var_psscale_weight_isr_G2QQ_cNS_up[0])
            #weightsSFVar.append(var_psscale_weight_isr_G2QQ_cNS_down[0])
            #weightsSFVar.append(var_btag_psscale_weight_isr_G2QQ_cNS_up[0])
            #weightsSFVar.append(var_btag_psscale_weight_isr_G2QQ_cNS_down[0])
            #weightsSFVar.append(var_psscale_weight_isr_Q2QG_cNS_up[0])
            #weightsSFVar.append(var_psscale_weight_isr_Q2QG_cNS_down[0])
            #weightsSFVar.append(var_btag_psscale_weight_isr_Q2QG_cNS_up[0])
            #weightsSFVar.append(var_btag_psscale_weight_isr_Q2QG_cNS_down[0])
            #weightsSFVar.append(var_psscale_weight_isr_X2XG_cNS_up[0])
            #weightsSFVar.append(var_psscale_weight_isr_X2XG_cNS_down[0])
            #weightsSFVar.append(var_btag_psscale_weight_isr_X2XG_cNS_up[0])
            #weightsSFVar.append(var_btag_psscale_weight_isr_X2XG_cNS_down[0])

            weightsSFVar.append(var_bfrag_peterson[0])
            weightsSFVar.append(var_btag_bfrag_peterson[0])
            weightsSFVar.append(var_top_pt[0])
          

            wsvList.append(weightsSFVar)

            for zz in range(0, len(var_pdf)):
                weightsSFVar.append(var_pdf[zz]) #central, down, up, down, up, ...
            tree.SetBranchStatus("*",1)
        assert len(weightsSFVar) == len (getVarNames()), "length of names %d and variation %d not matching" %(len(getVarNames()), len(weightsSFVar))
        print("\nConverting to df... %d" %len(wsvList))
        df = pd.DataFrame(wsvList)
        column_names = getVarNames()
        df.columns = column_names
        if (fileNames.index(fileName)==0):
            df.to_pickle('/nfs/dust/cms/user/celottog/mttNN/systematics/df.pkl')
            wsvList  = []
        else:
            print("Getting df..")
            df_old = pd.read_pickle('/nfs/dust/cms/user/celottog/mttNN/systematics/df.pkl')
            df = pd.concat([df_old, df])
            df.to_pickle('/nfs/dust/cms/user/celottog/mttNN/systematics/df.pkl')
            wsvList = []

    return df