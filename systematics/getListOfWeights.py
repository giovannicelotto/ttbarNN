import pandas as pd
import numpy as np

def getSystNames():
   systNames = ['weights','var_ele_id_down',      'var_ele_id_up',        'var_ele_reco_down',    'var_ele_reco_up',
                'var_muon_id_down',     'var_muon_id_up',
                #  'var_muon_id_syst_down','var_muon_id_syst_up', 'var_muon_id_stat_down', 'var_muon_id_stat_up',
                  'var_muon_iso_down', 'var_muon_iso_up',
                #'var_muon_iso_syst_down', 'var_muon_iso_syst_up','var_muon_iso_stat_down', 'var_muon_iso_stat_up',
                'var_pu_down',          'var_pu_up',            'var_trig_down',                'var_trig_up',
                'var_btag_down',        'var_btag_up',          'var_btag_ljet_down',           'var_btag_ljet_up',
                'var_btag_statistic_down', 'var_btag_statistic_up', 'var_btag_pileup_down',     'var_btag_pileup_up',
                'var_btag_jes_down',    'var_btag_jes_up',      'var_btag_type3_down',          'var_btag_type3_up',
                'var_l1prefiring_down', 'var_l1prefiring_up',   'var_merenscale_down',          'var_merenscale_up',
                'var_mefacscale_down',  'var_mefacscale_up',    
                'var_mescale_down',     'var_mescale_up',
                'var_bsemilep_down',    'var_bsemilep_up',      'var_ml_hdamp_down', 'var_ml_hdamp_up',
                'var_pdf_alphas_down',  'var_pdf_alphas_up',
                'var_bfrag_down',       'var_bfrag_up',          'var_bfrag_central',#'pdf',
                'var_psscale_weight_isr_up',    'var_psscale_weight_isr_down',   'var_psscale_weight_fsr_up',    'var_psscale_weight_fsr_down',
                #'var_psscale_weight_fsr_G2GG_muR_up',    'var_psscale_weight_fsr_G2GG_muR_down',
                #'var_psscale_weight_fsr_G2QQ_muR_up',    'var_psscale_weight_fsr_G2QQ_muR_down',   'var_psscale_weight_fsr_Q2QG_muR_up',    'var_psscale_weight_fsr_Q2QG_muR_down',
                #'var_psscale_weight_fsr_X2XG_muR_up',    'var_psscale_weight_fsr_X2XG_muR_down',   'var_psscale_weight_fsr_G2GG_cNS_up',    'var_psscale_weight_fsr_G2GG_cNS_down',
                #'var_psscale_weight_fsr_G2QQ_cNS_up',    'var_psscale_weight_fsr_G2QQ_cNS_down',   'var_psscale_weight_fsr_Q2QG_cNS_up',    'var_psscale_weight_fsr_Q2QG_cNS_down',
                #'var_psscale_weight_fsr_X2XG_cNS_up',    'var_psscale_weight_fsr_X2XG_cNS_down',   'var_psscale_weight_isr_G2GG_muR_up',    'var_psscale_weight_isr_G2GG_muR_down',
                #'var_psscale_weight_isr_G2QQ_muR_up',    'var_psscale_weight_isr_G2QQ_muR_down',   'var_psscale_weight_isr_Q2QG_muR_up',    'var_psscale_weight_isr_Q2QG_muR_down',
                #'var_psscale_weight_isr_X2XG_muR_up',    'var_psscale_weight_isr_X2XG_muR_down',   'var_psscale_weight_isr_G2GG_cNS_up',    'var_psscale_weight_isr_G2GG_cNS_down',
                #'var_psscale_weight_isr_G2QQ_cNS_up',    'var_psscale_weight_isr_G2QQ_cNS_down',   'var_psscale_weight_isr_Q2QG_cNS_up',    'var_psscale_weight_isr_Q2QG_cNS_down',
                #'var_psscale_weight_isr_X2XG_cNS_up',    'var_psscale_weight_isr_X2XG_cNS_down',
                'var_bfrag_peterson', 'var_top_pt']
   for x in range(1, 52):
      systNames.append('var_pdf_up_'+str(x))
   for x in range(1, 52):
      systNames.append('var_pdf_down_'+str(x))
   systNames.append('var_pdf_central_0')   

   return systNames.copy()

def getListOfWeights(df):
   w = {}
   w['weights'] =           df['weight']*df['btagSF']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']
   w['var_ele_id_down'] =   df['weight']*df['var_btag_ele_id_down']*df['var_ele_id_down']        *df['pileupSF']*df['prefiringWeight']*df['triggerSF']
   w['var_ele_id_up'] =     df['weight']*df['var_btag_ele_id_up']*df['var_ele_id_up']          *df['pileupSF']*df['prefiringWeight']*df['triggerSF']
   w['var_ele_reco_down'] = df['weight']*df['var_btag_ele_reco_down']*df['var_ele_reco_down']      *df['pileupSF']*df['prefiringWeight']*df['triggerSF']
   w['var_ele_reco_up'] =   df['weight']*df['var_btag_ele_reco_up']*df['var_ele_reco_up']        *df['pileupSF']*df['prefiringWeight']*df['triggerSF']
   w['var_muon_id_down'] =  df['weight']*df['var_btag_muon_id_down']*df['var_muon_id_down']       *df['pileupSF']*df['prefiringWeight']*df['triggerSF']
   w['var_muon_id_up'] =    df['weight']*df['var_btag_muon_id_up']*df['var_muon_id_up']         *df['pileupSF']*df['prefiringWeight']*df['triggerSF']
   w['var_muon_iso_down'] =  df['weight']*df['var_btag_muon_iso_down']*df['var_muon_iso_down']      *df['pileupSF']*df['prefiringWeight']*df['triggerSF']
   w['var_muon_iso_up']   =  df['weight']*df['var_btag_muon_iso_up']*df['var_muon_iso_up']        *df['pileupSF']*df['prefiringWeight']*df['triggerSF'] 
   w['var_pu_down']      =  df['weight']*df['var_btag_pu_down']*df['leptonSF']*df['var_pu_down']*df['prefiringWeight']*df['triggerSF']
   w['var_pu_up']        =  df['weight']*df['var_btag_pu_up']*df['leptonSF']*df['var_pu_up']*df['prefiringWeight']*df['triggerSF']
   w['var_trig_down']    =  df['weight']*df['var_btag_trig_down']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['var_trig_down']
   w['var_trig_up']      =  df['weight']*df['var_btag_trig_up']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['var_trig_up']


   w['var_btag_down']     = df['weight']*df['var_btag_down']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']
   w['var_btag_up']       = df['weight']*df['var_btag_up']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']
   w['var_btag_ljet_down']= df['weight']*df['var_btag_ljet_down']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']
   w['var_btag_ljet_up']  = df['weight']*df['var_btag_ljet_up']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']
   w['var_btag_statistic_down']    = df['weight']*df['var_btag_statistic_down']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']
   w['var_btag_statistic_up']      = df['weight']*df['var_btag_statistic_up']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']
   w['var_btag_pileup_down']    = df['weight']*df['var_btag_pileup_down']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']
   w['var_btag_pileup_up']      = df['weight']*df['var_btag_pileup_up']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']
   w['var_btag_jes_down']    = df['weight']*df['var_btag_jes_down']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']
   w['var_btag_jes_up']      = df['weight']*df['var_btag_jes_up']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']
   w['var_btag_type3_down']     = df['weight']*df['var_btag_type3_down']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']
   w['var_btag_type3_up']    = df['weight']*df['var_btag_type3_up']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']

   w['var_l1prefiring_down']     = df['weight']*df['var_btag_l1prefiring_down']*df['leptonSF']*df['pileupSF']*df['var_l1prefiring_down']*df['triggerSF']
   w['var_l1prefiring_up']    = df['weight']*df['var_btag_l1prefiring_up']*df['leptonSF']*df['pileupSF']*df['var_l1prefiring_up']*df['triggerSF']
   w['var_merenscale_down']      = df['weight']*df['var_btag_merenscale_down']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_merenscale_down']
   w['var_merenscale_up']     = df['weight']*df['var_btag_merenscale_up']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_merenscale_up']
   w['var_mefacscale_down']      = df['weight']*df['var_btag_mefacscale_down']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_mefacscale_down']
   w['var_mefacscale_up']     = df['weight']*df['var_btag_mefacscale_up']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_mefacscale_up']
   w['var_mescale_down']      = df['weight']*df['var_btag_mescale_down']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_mescale_down']
   w['var_mescale_up']     = df['weight']*df['var_btag_mescale_up']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_mescale_up']

   w['var_bsemilep_down']     = df['weight']*df['var_btag_bsemilep_down']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_bsemilep_down']
   w['var_bsemilep_up']    = df['weight']*df['var_btag_bsemilep_up']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_bsemilep_up']
   w['var_ml_hdamp_down']     = df['weight']*df['var_btag_ml_hdamp_down']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_ml_hdamp_down']
   w['var_ml_hdamp_up']    = df['weight']*df['var_btag_ml_hdamp_up']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_ml_hdamp_up']
   w['var_pdf_alphas_down']      = df['weight']*df['var_btag_pdf_alphas_down']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_pdf_alphas_down']
   w['var_pdf_alphas_up']     = df['weight']*df['var_btag_pdf_alphas_up']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_pdf_alphas_up']
   w['var_bfrag_down']     = df['weight']*df['var_btag_bfrag_down']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_bfrag_down']/df['var_bfrag_central']
   w['var_bfrag_up']    = df['weight']*df['var_btag_bfrag_up']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_bfrag_up']/df['var_bfrag_central']
   w['var_bfrag_central']     = df['weight']*df['var_btag_bfrag_central']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']/df['var_bfrag_central']

   #w['var_psscale_weight_up_3']   = df['weight']*df['var_btag_psscale_weight_isr_up']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_psscale_weight_isr_up']
   #w['var_psscale_weight_down_3'] = df['weight']*df['var_btag_psscale_weight_isr_down']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_psscale_weight_isr_down']
   #w['var_psscale_weight_up_14']   = df['weight']*df['var_btag_psscale_weight_fsr_up']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_psscale_weight_fsr_up']
   #w['var_psscale_weight_down_14'] = df['weight']*df['var_btag_psscale_weight_fsr_down']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_psscale_weight_fsr_down']
   #w['var_psscale_weight_up_5']   = df['weight']*df['var_btag_psscale_weight_fsr_G2GG_muR_up']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_psscale_weight_fsr_G2GG_muR_up']
   #w['var_psscale_weight_down_5'] = df['weight']*df['var_btag_psscale_weight_fsr_G2GG_muR_down']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_psscale_weight_fsr_G2GG_muR_down']
   #w['var_psscale_weight_up_6']   = df['weight']*df['var_btag_psscale_weight_fsr_G2QQ_muR_up']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_psscale_weight_fsr_G2QQ_muR_up']
   #w['var_psscale_weight_down_6'] = df['weight']*df['var_btag_psscale_weight_fsr_G2QQ_muR_down']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_psscale_weight_fsr_G2QQ_muR_down']
   #w['var_psscale_weight_up_7']   = df['weight']*df['var_btag_psscale_weight_fsr_Q2QG_muR_up']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_psscale_weight_fsr_Q2QG_muR_up']
   #w['var_psscale_weight_down_7'] = df['weight']*df['var_btag_psscale_weight_fsr_Q2QG_muR_down']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_psscale_weight_fsr_Q2QG_muR_down']
   #w['var_psscale_weight_up_8']   = df['weight']*df['var_btag_psscale_weight_fsr_X2XG_muR_up']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_psscale_weight_fsr_X2XG_muR_up']
   #w['var_psscale_weight_down_8'] = df['weight']*df['var_btag_psscale_weight_fsr_X2XG_muR_down']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_psscale_weight_fsr_X2XG_muR_down']
   #w['var_psscale_weight_up_9']   = df['weight']*df['var_btag_psscale_weight_fsr_G2GG_cNS_up']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_psscale_weight_fsr_G2GG_cNS_up']
   #w['var_psscale_weight_down_9'] = df['weight']*df['var_btag_psscale_weight_fsr_G2GG_cNS_down']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_psscale_weight_fsr_G2GG_cNS_down']
   #w['var_psscale_weight_up_10']   = df['weight']*df['var_btag_psscale_weight_fsr_G2QQ_cNS_up']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_psscale_weight_fsr_G2QQ_cNS_up']
   #w['var_psscale_weight_down_10'] = df['weight']*df['var_btag_psscale_weight_fsr_G2QQ_cNS_down']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_psscale_weight_fsr_G2QQ_cNS_down']
   #w['var_psscale_weight_up_11']   = df['weight']*df['var_btag_psscale_weight_fsr_Q2QG_cNS_up']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_psscale_weight_fsr_Q2QG_cNS_up']
   #w['var_psscale_weight_down_11'] = df['weight']*df['var_btag_psscale_weight_fsr_Q2QG_cNS_down']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_psscale_weight_fsr_Q2QG_cNS_down']
   #w['var_psscale_weight_up_12']   = df['weight']*df['var_btag_psscale_weight_fsr_X2XG_cNS_up']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_psscale_weight_fsr_X2XG_cNS_up']
   #w['var_psscale_weight_down_12'] = df['weight']*df['var_btag_psscale_weight_fsr_X2XG_cNS_down']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_psscale_weight_fsr_X2XG_cNS_down']
#
   #w['var_psscale_weight_isr_up_16']   = df['weight']*df['var_btag_psscale_weight_isr_G2GG_muR_up']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_psscale_weight_isr_G2GG_muR_up']
   #w['var_psscale_weight_isr_down_16'] = df['weight']*df['var_btag_psscale_weight_isr_G2GG_muR_down']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_psscale_weight_isr_G2GG_muR_down']
   #w['var_psscale_weight_isr_up_17']   = df['weight']*df['var_btag_psscale_weight_isr_G2QQ_muR_up']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_psscale_weight_isr_G2QQ_muR_up']
   #w['var_psscale_weight_isr_down_17'] = df['weight']*df['var_btag_psscale_weight_isr_G2QQ_muR_down']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_psscale_weight_isr_G2QQ_muR_down']
   #w['var_psscale_weight_isr_up_18']   = df['weight']*df['var_btag_psscale_weight_isr_Q2QG_muR_up']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_psscale_weight_isr_Q2QG_muR_up']
   #w['var_psscale_weight_isr_down_18'] = df['weight']*df['var_btag_psscale_weight_isr_Q2QG_muR_down']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_psscale_weight_isr_Q2QG_muR_down']
   #w['var_psscale_weight_isr_up_19']   = df['weight']*df['var_btag_psscale_weight_isr_X2XG_muR_up']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_psscale_weight_isr_X2XG_muR_up']
   #w['var_psscale_weight_isr_down_19'] = df['weight']*df['var_btag_psscale_weight_isr_X2XG_muR_down']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_psscale_weight_isr_X2XG_muR_down']
   #w['var_psscale_weight_isr_up_20']   = df['weight']*df['var_btag_psscale_weight_isr_G2GG_cNS_up']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_psscale_weight_isr_G2GG_cNS_up']
   #w['var_psscale_weight_isr_down_20'] = df['weight']*df['var_btag_psscale_weight_isr_G2GG_cNS_down']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_psscale_weight_isr_G2GG_cNS_down']
   #w['var_psscale_weight_isr_up_21']   = df['weight']*df['var_btag_psscale_weight_isr_G2QQ_cNS_up']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_psscale_weight_isr_G2QQ_cNS_up']
   #w['var_psscale_weight_isr_down_21'] = df['weight']*df['var_btag_psscale_weight_isr_G2QQ_cNS_down']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_psscale_weight_isr_G2QQ_cNS_down']
   #w['var_psscale_weight_isr_up_22']   = df['weight']*df['var_btag_psscale_weight_isr_Q2QG_cNS_up']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_psscale_weight_isr_Q2QG_cNS_up']
   #w['var_psscale_weight_isr_down_22'] = df['weight']*df['var_btag_psscale_weight_isr_Q2QG_cNS_down']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_psscale_weight_isr_Q2QG_cNS_down']
   #w['var_psscale_weight_isr_up_23']   = df['weight']*df['var_btag_psscale_weight_isr_X2XG_cNS_up']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_psscale_weight_isr_X2XG_cNS_up']
   #w['var_psscale_weight_isr_down_23'] = df['weight']*df['var_btag_psscale_weight_isr_X2XG_cNS_down']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_psscale_weight_isr_X2XG_cNS_down']
   w['var_psscale_weight_isr_up'] =   df['weight']*df['var_btag_psscale_weight_isr_up']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_psscale_weight_isr_up']
   w['var_psscale_weight_isr_down'] =   df['weight']*df['var_btag_psscale_weight_isr_down']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_psscale_weight_isr_down']
   w['var_psscale_weight_fsr_up'] =   df['weight']*df['var_btag_psscale_weight_fsr_up']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_psscale_weight_fsr_up']
   w['var_psscale_weight_fsr_down'] =   df['weight']*df['var_btag_psscale_weight_fsr_down']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_psscale_weight_fsr_down']

   w['var_bfrag_peterson'] = df['weight']*df['var_btag_bfrag_peterson']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_bfrag_peterson']/df['var_bfrag_central']
   w['var_top_pt']   = df['weight']*df['btagSF']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_top_pt']

   for idx in range(1, 52):
      w['var_pdf_up_'+str(idx)] = df['weight']*df['btagSF']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*(df['var_pdf_up_'+str(idx)])
      w['var_pdf_down_'+str(idx)] = df['weight']*df['btagSF']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*(df['var_pdf_down_'+str(idx)])
   w['var_pdf_central_0'] = df['weight']*df['btagSF']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*(df['var_pdf_central_0'])
   
   
   return w.copy()