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
                'var_mescale_down',      'var_mescale_up',
                'var_bsemilep_down',    'var_bsemilep_up',      'var_ml_hdamp_down', 'var_ml_hdamp_up',
                'var_pdf_alphas_down',          'var_pdf_alphas_up',
                'var_bfrag_down',       'var_bfrag_up',          'var_bfrag_central',           #'pdf',
                'var_psscale_weight_4_down', 'var_psscale_weight_4_up',
                'var_psscale_weight_5_down', 'var_psscale_weight_5_up', 'var_bfrag_peterson', 'var_top_pt']
   for x in range(1, 52):
      systNames.append('var_pdf_up_'+str(x))
   for x in range(1, 52):
      systNames.append('var_pdf_down_'+str(x))
   systNames.append('var_pdf_central_0')   

   return systNames.copy()

def getListOfWeights(df):
   w = [df['weight']*df['btagSF']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF'],
   # Electron
     df['weight']*df['var_btag_ele_id_down']*df['var_ele_id_down']        *df['pileupSF']*df['prefiringWeight']*df['triggerSF'],
     df['weight']*df['var_btag_ele_id_up']*df['var_ele_id_up']          *df['pileupSF']*df['prefiringWeight']*df['triggerSF'],
     df['weight']*df['btagSF']*df['var_ele_reco_down']      *df['pileupSF']*df['prefiringWeight']*df['triggerSF'],
     df['weight']*df['btagSF']*df['var_ele_reco_up']        *df['pileupSF']*df['prefiringWeight']*df['triggerSF'],
   # Muon Id and Iso
     df['weight']*df['btagSF']*df['var_muon_id_down']       *df['pileupSF']*df['prefiringWeight']*df['triggerSF'],
     df['weight']*df['btagSF']*df['var_muon_id_up']         *df['pileupSF']*df['prefiringWeight']*df['triggerSF'],
     #df['weight']*df['btagSF']*df['var_muon_id_syst_down']  *df['pileupSF']*df['prefiringWeight']*df['triggerSF'],
     #df['weight']*df['btagSF']*df['var_muon_id_syst_up']    *df['pileupSF']*df['prefiringWeight']*df['triggerSF'],
     #df['weight']*df['btagSF']*df['var_muon_id_stat_down']  *df['pileupSF']*df['prefiringWeight']*df['triggerSF'],
     #df['weight']*df['btagSF']*df['var_muon_id_stat_up']    *df['pileupSF']*df['prefiringWeight']*df['triggerSF'],
     df['weight']*df['btagSF']*df['var_muon_iso_down']      *df['pileupSF']*df['prefiringWeight']*df['triggerSF'],
     df['weight']*df['btagSF']*df['var_muon_iso_up']        *df['pileupSF']*df['prefiringWeight']*df['triggerSF'],
     #df['weight']*df['btagSF']*df['var_muon_iso_syst_down'] *df['pileupSF']*df['prefiringWeight']*df['triggerSF'],
     #df['weight']*df['btagSF']*df['var_muon_iso_syst_up']   *df['pileupSF']*df['prefiringWeight']*df['triggerSF'],
     #df['weight']*df['btagSF']*df['var_muon_iso_stat_down'] *df['pileupSF']*df['prefiringWeight']*df['triggerSF'],
     #df['weight']*df['btagSF']*df['var_muon_iso_stat_up']   *df['pileupSF']*df['prefiringWeight']*df['triggerSF'],

  # Pileup
     df['weight']*df['var_btag_pu_down']*df['leptonSF']*df['var_pu_down']*df['prefiringWeight']*df['triggerSF'],
     df['weight']*df['var_btag_pu_up']*df['leptonSF']*df['var_pu_up']*df['prefiringWeight']*df['triggerSF'],
  # Trigger
     df['weight']*df['btagSF']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['var_trig_down'],
     df['weight']*df['btagSF']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['var_trig_up'],
  # btag only
     df['weight']*df['var_btag_down']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF'],
     df['weight']*df['var_btag_up']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF'],
     df['weight']*df['var_btag_ljet_down']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF'],
     df['weight']*df['var_btag_ljet_up']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF'],
     df['weight']*df['var_btag_statistic_down']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF'],
     df['weight']*df['var_btag_statistic_up']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF'],
     df['weight']*df['var_btag_pileup_down']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF'],
     df['weight']*df['var_btag_pileup_up']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF'],
     df['weight']*df['var_btag_jes_down']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF'],
     df['weight']*df['var_btag_jes_up']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF'],
     df['weight']*df['var_btag_type3_down']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF'],
     df['weight']*df['var_btag_type3_up']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF'],
  # prefiring and energy scales
     df['weight']*df['var_btag_l1prefiring_down']*df['leptonSF']*df['pileupSF']*df['var_l1prefiring_down']*df['triggerSF'],
     df['weight']*df['var_btag_l1prefiring_up']*df['leptonSF']*df['pileupSF']*df['var_l1prefiring_up']*df['triggerSF'] ,
     df['weight']*df['var_btag_merenscale_down']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_merenscale_down'],
     df['weight']*df['var_btag_merenscale_up']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_merenscale_up'],
     df['weight']*df['var_btag_mefacscale_down']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_mefacscale_down'],
     df['weight']*df['var_btag_mefacscale_up']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_mefacscale_up'],
     df['weight']*df['var_btag_mescale_down']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_mescale_down'],
     df['weight']*df['var_btag_mescale_up']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_mescale_up'],
  
  # bsemilep pdf_alphas, b-fragmentation
     df['weight']*df['var_btag_bsemilep_down']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_bsemilep_down'],
     df['weight']*df['var_btag_bsemilep_up']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_bsemilep_up'], 
     df['weight']*df['var_btag_ml_hdamp_down']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_ml_hdamp_down'],
     df['weight']*df['var_btag_ml_hdamp_up']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_ml_hdamp_up'], 
     df['weight']*df['var_btag_pdf_alphas_down']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_pdf_alphas_down'],
     df['weight']*df['var_btag_pdf_alphas_up']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_pdf_alphas_up'],
     df['weight']*df['var_btag_bfrag_down']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_bfrag_down']/df['var_bfrag_central'],
     df['weight']*df['var_btag_bfrag_up']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_bfrag_up']/df['var_bfrag_central'],
     df['weight']*df['btagSF']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']/df['var_bfrag_central'],     # division by bfrag central
  # psscale
     df['weight']*df['btagSF']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_psscale_weight_4_down'],
     df['weight']*df['btagSF']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_psscale_weight_4_up'],
     df['weight']*df['btagSF']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_psscale_weight_5_down'],
     df['weight']*df['btagSF']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_psscale_weight_5_up'],
  # bfrag peterson
     df['weight']*df['var_btag_bfrag_peterson']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_bfrag_peterson']/df['var_bfrag_central'],
  # top pt
     df['weight']*df['btagSF']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*df['var_top_pt'],
     ]

   for idx in range(1, 52):
      w.append(df['weight']*df['btagSF']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*(df['var_pdf_up_'+str(idx)]))
   for idx in range(1, 52):
      w.append(df['weight']*df['btagSF']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*(df['var_pdf_down_'+str(idx)]))
   w.append(df['weight']*df['btagSF']*df['leptonSF']*df['pileupSF']*df['prefiringWeight']*df['triggerSF']*(df['var_pdf_central_0']))
   
   
   return w.copy()