import numpy as np
import matplotlib.pyplot as plt
import pickle 

filepath = "/nfs/dust/cms/user/celottog/mttNN/systematics/responseMatrices/dnn/relVariations.pkl"
outFolder = "/nfs/dust/cms/user/celottog/mttNN/systematics/plotSystematics/dnn/"
with open(filepath, "rb") as file:
            myDict = pickle.load(file)
# Binning for the seven bins
bins = np.linspace(0.5, 7.5, 8)
colors=[     'C0', 'C1', 'C2', 'C3', 'mediumblue', 'lime', 'magenta', 'C7', 'C8', 'C9', '', '']


#for l in myDict.keys():
#      print(l, np.sqrt(np.diag(myDict[l])))

# Sort labels by their syst impact
labels = []
for l in myDict.keys():
      labels.append(l)
#values = []
#for l in range(len(labels)):
#       values.append(np.mean(myDict[labels[l]]))
#indices = np.argsort(values)
#labelsNew_ = [labels[i] for i in indices]
#labels=labelsNew_
def custom_sort_key(item):
    if "down" in item:
        corresponding_item = item.replace("down", "up")
        return (corresponding_item, item)
    else:
        return (item,)

labels = sorted(labels, key=custom_sort_key)
labels.remove('weights')
labels.remove('var_pdf_central_0')


#
label_mapping = {
    'mefacscale': '$\mu_F$',
    'merenscale': '$\mu_R$',
    'mescale': '$\mu_R$ and $\mu_F$',
    
    'trig': 'Trigger',
    'pu': 'Pileup',
    'muon_id': 'Muon ID',
    'ele_id': 'Ele ID',
    'pdf_alphas': r'PDF $\alpha_S$',
    'pdf_central_0': 'PDF Central'
    
}
#

def transformLabel(label):
    label = label.replace("var_", "")
    label = label.replace("psscale_weight_", "")
    label = label.replace("_", " ")
    if any(substring in label for substring in ['scale', 'damp']):
        label = label.replace("mefacscale", '$\mu_F$')
        label = label.replace("merenscale", '$\mu_R$')
        label = label.replace("mescale", '$\mu_R$ and $\mu_F$')
        label = label.replace('ml hdamp', '$h_\mathrm{damp}$')
        
    else:
        label = label.title()
    label = label.replace("Jes", "JES")
    label = label.replace("Btag", "b-tag")
    label = label.replace("Up", "UP")
    label = label.replace("Down", "DOWN")
    label = label.replace("Id", "ID")
    label = label.replace("Fsr", "FSR")
    label = label.replace("Isr", "ISR")
    label = label.replace("Pu ", "Pileup ")
    label = label.replace("Trig ", "Trigger")
    label = label.replace("Pdf", "PDF")
    label = label.replace("Alphas", "$\\alpha_S$")
    return label


index = 0
fig, ax = plt.subplots(1, 1)
for k in labels:
    if 'pdf' in k:
        continue

    label = transformLabel(k)
    ax.hist(bins[:-1], bins=bins, weights=myDict[k], label=label, histtype=u'step', color=colors[index%8])
    index = index+1
    
    if ((index%8==0) & (index>0)):
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]*1.5)
        ax.set_xlim(bins[0], bins[-1])
        ax.set_xlabel("Bin number", fontsize=16)
        ax.set_ylabel("Relative Variation", fontsize=16)
        ax.tick_params(labelsize=14)
        ax.legend(loc='upper center', ncols=2)
        ax.text(s="Private Work (CMS Data)", x=0., y=1.02, ha='left', fontsize=14,  transform=ax.transAxes,  **{'fontname':'Arial'})
        ax.text(s="59.8 fb$^{-1}$ (13 TeV)", x=1.00, y=1.02, ha='right', fontsize=14,  transform=ax.transAxes,  **{'fontname':'Arial'})
        fig.savefig(outFolder +"/DNNhierarchyUncertainty"+ str(index) +".pdf", bbox_inches='tight')
        plt.cla()
    
ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]*1.5)
ax.set_xlim(bins[0], bins[-1])
ax.set_xlabel("Bin number", fontsize=16)
ax.set_ylabel("Relative Variation", fontsize=16)
ax.tick_params(labelsize=14)
ax.legend(loc='upper center', ncols=2)
ax.text(s="Private Work (CMS Data)", x=0., y=1.02, ha='left', fontsize=14,  transform=ax.transAxes,  **{'fontname':'Arial'})
ax.text(s="59.8 fb$^{-1}$ (13 TeV)", x=1.00, y=1.02, ha='right', fontsize=14,  transform=ax.transAxes,  **{'fontname':'Arial'})
fig.savefig(outFolder +"/DNNhierarchyUncertainty"+ str(index) +".pdf", bbox_inches='tight')
plt.cla()

index = 0


for k in labels:
    if 'pdf' not in k:
        continue
    
    label = transformLabel(k)
    ax.hist(bins[:-1], bins=bins, weights=myDict[k], label=label, histtype=u'step', color=colors[index%8])
    index = index+1
    print(k, label)
    if (index==104):
        continue
    if ((index%8==0) & (index>0)):
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]*1.5)
        ax.set_xlim(bins[0], bins[-1])
        ax.set_xlabel("Bin number", fontsize=16)
        ax.set_ylabel("Relative Variation", fontsize=16)
        ax.tick_params(labelsize=14)
        ax.legend(loc='upper center', ncols=2)
        ax.text(s="Private Work (CMS Data)", x=0., y=1.02, ha='left', fontsize=14,  transform=ax.transAxes,  **{'fontname':'Arial'})
        ax.text(s="59.8 fb$^{-1}$ (13 TeV)", x=1.00, y=1.02, ha='right', fontsize=14,  transform=ax.transAxes,  **{'fontname':'Arial'})
        fig.savefig(outFolder +"/DNNhierarchyUncertaintyPDF"+ str(index) +".pdf", bbox_inches='tight')
        plt.cla()
    
ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]*1.5)
ax.set_xlim(bins[0], bins[-1])
ax.set_xlabel("Bin number", fontsize=16)
ax.set_ylabel("Relative Variation", fontsize=16)
ax.tick_params(labelsize=14)
ax.legend(loc='upper center', ncols=2)
ax.text(s="Private Work (CMS Data)", x=0., y=1.02, ha='left', fontsize=14,  transform=ax.transAxes,  **{'fontname':'Arial'})
ax.text(s="59.8 fb$^{-1}$ (13 TeV)", x=1.00, y=1.02, ha='right', fontsize=14,  transform=ax.transAxes,  **{'fontname':'Arial'})
fig.savefig(outFolder +"/DNNhierarchyUncertaintyPDF"+ str(index) +".pdf", bbox_inches='tight')


# **************** 
# *    LOOSE     *
# ****************
filepath = "/nfs/dust/cms/user/celottog/mttNN/systematics/responseMatrices/loose/relVariations.pkl"
outFolder = "/nfs/dust/cms/user/celottog/mttNN/systematics/plotSystematics/loose/"
with open(filepath, "rb") as file:
            myDict = pickle.load(file)


index = 0
fig, ax = plt.subplots(1, 1)
for k in labels:
    if 'pdf' in k:
        continue
    
    #if k[4:] in label_mapping:
    #    labelNew = label_mapping[k[4:]]
    #else:
    label = transformLabel(k)#    labelNew = k[4:].replace('_', ' ').title()
    ax.hist(bins[:-1], bins=bins, weights=myDict[k], label=label, histtype=u'step', color=colors[index%8])
    index = index+1
    
    if ((index%8==0) & (index>0)):
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]*1.5)
        ax.set_xlim(bins[0], bins[-1])
        ax.set_xlabel("Bin number", fontsize=16)
        ax.set_ylabel("Relative Variation", fontsize=16)
        ax.tick_params(labelsize=14)
        ax.legend(loc='upper center', ncols=2)
        ax.text(s="Private Work (CMS Data)", x=0., y=1.02, ha='left', fontsize=14,  transform=ax.transAxes,  **{'fontname':'Arial'})
        ax.text(s="59.8 fb$^{-1}$ (13 TeV)", x=1.00, y=1.02, ha='right', fontsize=14,  transform=ax.transAxes,  **{'fontname':'Arial'})
        fig.savefig(outFolder +"/LOOSEhierarchyUncertainty"+ str(index) +".pdf", bbox_inches='tight')
        plt.cla()
    
ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]*1.5)
ax.set_xlim(bins[0], bins[-1])
ax.set_xlabel("Bin number", fontsize=16)
ax.set_ylabel("Relative Variation", fontsize=16)
ax.tick_params(labelsize=14)
ax.legend(loc='upper center', ncols=2)
ax.text(s="Private Work (CMS Data)", x=0., y=1.02, ha='left', fontsize=14,  transform=ax.transAxes,  **{'fontname':'Arial'})
ax.text(s="59.8 fb$^{-1}$ (13 TeV)", x=1.00, y=1.02, ha='right', fontsize=14,  transform=ax.transAxes,  **{'fontname':'Arial'})
fig.savefig(outFolder +"/LOOSEhierarchyUncertainty"+ str(index) +".pdf", bbox_inches='tight')
plt.cla()

index = 0


for k in labels:
    if 'pdf' not in k:
        continue
    #if k[4:] in label_mapping:
    #    labelNew = label_mapping[k[4:]]
    #else:
    #    labelNew = k[4:].replace('_', ' ').upper()
    label = transformLabel(k)
    ax.hist(bins[:-1], bins=bins, weights=myDict[k], label=label, histtype=u'step', color=colors[index%8])
    index = index+1
    if (index==104):
        continue
    if ((index%8==0) & (index>0)):
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]*1.5)
        ax.set_xlim(bins[0], bins[-1])
        ax.set_xlabel("Bin number", fontsize=16)
        ax.set_ylabel("Relative Variation", fontsize=16)
        ax.tick_params(labelsize=14)
        ax.legend(loc='upper center', ncols=2)
        ax.text(s="Private Work (CMS Data)", x=0., y=1.02, ha='left', fontsize=14,  transform=ax.transAxes,  **{'fontname':'Arial'})
        ax.text(s="59.8 fb$^{-1}$ (13 TeV)", x=1.00, y=1.02, ha='right', fontsize=14,  transform=ax.transAxes,  **{'fontname':'Arial'})
        fig.savefig(outFolder +"/LOOSEhierarchyUncertaintyPDF"+ str(index) +".pdf", bbox_inches='tight')
        plt.cla()
    
ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]*1.5)
ax.set_xlim(bins[0], bins[-1])
ax.set_xlabel("Bin number", fontsize=16)
ax.set_ylabel("Relative Variation", fontsize=16)
ax.tick_params(labelsize=14)
ax.legend(loc='upper center', ncols=2)
ax.text(s="Private Work (CMS Data)", x=0., y=1.02, ha='left', fontsize=14,  transform=ax.transAxes,  **{'fontname':'Arial'})
ax.text(s="59.8 fb$^{-1}$ (13 TeV)", x=1.00, y=1.02, ha='right', fontsize=14,  transform=ax.transAxes,  **{'fontname':'Arial'})
fig.savefig(outFolder +"/LOOSEhierarchyUncertaintyPDF"+ str(index) +".pdf", bbox_inches='tight')


# **************** 
# *    KIN       *
# ****************
filepath = "/nfs/dust/cms/user/celottog/mttNN/systematics/responseMatrices/kin/relVariations.pkl"
outFolder = "/nfs/dust/cms/user/celottog/mttNN/systematics/plotSystematics/kin/"
with open(filepath, "rb") as file:
            myDict = pickle.load(file)


index = 0
fig, ax = plt.subplots(1, 1)
for k in labels:
    if 'pdf' in k:
        continue
    
    #if k[4:] in label_mapping:
    #    labelNew = label_mapping[k[4:]]
    #else:
    label = transformLabel(k)#    labelNew = k[4:].replace('_', ' ').title()
    ax.hist(bins[:-1], bins=bins, weights=myDict[k], label=label, histtype=u'step', color=colors[index%8])
    index = index+1
    
    if ((index%8==0) & (index>0)):
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]*1.5)
        ax.set_xlim(bins[0], bins[-1])
        ax.set_xlabel("Bin number", fontsize=16)
        ax.set_ylabel("Relative Variation", fontsize=16)
        ax.tick_params(labelsize=14)
        ax.legend(loc='upper center', ncols=2)
        ax.text(s="Private Work (CMS Data)", x=0., y=1.02, ha='left', fontsize=14,  transform=ax.transAxes,  **{'fontname':'Arial'})
        ax.text(s="59.8 fb$^{-1}$ (13 TeV)", x=1.00, y=1.02, ha='right', fontsize=14,  transform=ax.transAxes,  **{'fontname':'Arial'})
        fig.savefig(outFolder +"/KINhierarchyUncertainty"+ str(index) +".pdf", bbox_inches='tight')
        plt.cla()
    
ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]*1.5)
ax.set_xlim(bins[0], bins[-1])
ax.set_xlabel("Bin number", fontsize=16)
ax.set_ylabel("Relative Variation", fontsize=16)
ax.tick_params(labelsize=14)
ax.legend(loc='upper center', ncols=2)
ax.text(s="Private Work (CMS Data)", x=0., y=1.02, ha='left', fontsize=14,  transform=ax.transAxes,  **{'fontname':'Arial'})
ax.text(s="59.8 fb$^{-1}$ (13 TeV)", x=1.00, y=1.02, ha='right', fontsize=14,  transform=ax.transAxes,  **{'fontname':'Arial'})
fig.savefig(outFolder +"/KINhierarchyUncertainty"+ str(index) +".pdf", bbox_inches='tight')
plt.cla()

index = 0


for k in labels:
    if 'pdf' not in k:
        continue
    
    label = transformLabel(k)
    ax.hist(bins[:-1], bins=bins, weights=myDict[k], label=label, histtype=u'step', color=colors[index%8])
    index = index+1
    if (index==104):
        continue
    if ((index%8==0) & (index>0)):
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]*1.5)
        ax.set_xlim(bins[0], bins[-1])
        ax.set_xlabel("Bin number", fontsize=16)
        ax.set_ylabel("Relative Variation", fontsize=16)
        ax.tick_params(labelsize=14)
        ax.legend(loc='upper center', ncols=2)
        ax.text(s="Private Work (CMS Data)", x=0., y=1.02, ha='left', fontsize=14,  transform=ax.transAxes,  **{'fontname':'Arial'})
        ax.text(s="59.8 fb$^{-1}$ (13 TeV)", x=1.00, y=1.02, ha='right', fontsize=14,  transform=ax.transAxes,  **{'fontname':'Arial'})
        fig.savefig(outFolder +"/KINhierarchyUncertaintyPDF"+ str(index) +".pdf", bbox_inches='tight')
        plt.cla()
    
ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]*1.5)
ax.set_xlim(bins[0], bins[-1])
ax.set_xlabel("Bin number", fontsize=16)
ax.set_ylabel("Relative Variation", fontsize=16)
ax.tick_params(labelsize=14)
ax.legend(loc='upper center', ncols=2)
ax.text(s="Private Work (CMS Data)", x=0., y=1.02, ha='left', fontsize=14,  transform=ax.transAxes,  **{'fontname':'Arial'})
ax.text(s="59.8 fb$^{-1}$ (13 TeV)", x=1.00, y=1.02, ha='right', fontsize=14,  transform=ax.transAxes,  **{'fontname':'Arial'})
fig.savefig(outFolder +"/KINhierarchyUncertaintyPDF"+ str(index) +".pdf", bbox_inches='tight')