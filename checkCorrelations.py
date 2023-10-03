import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils.helpers import getFeatureNames

print("Program Starting")
dataFolder = '/nfs/dust/cms/user/celottog/mttNN/npyData/3*None/inX.npy'
howManyFeatures = 73
inX = np.load(dataFolder)
inX = inX[:1000, :]
inX = inX[inX[:, 0] > -998, :]
df = pd.DataFrame(inX[:20, :howManyFeatures])
labels = getFeatureNames()[:howManyFeatures]

# Calculate the correlation matrix
correlation_matrix = df.corr()

fig, ax = plt.subplots(1, 1, figsize=(40, 40))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", ax=ax, annot_kws={"size": 8}, cmap="coolwarm",
            mask=~((correlation_matrix > 0.7) | (correlation_matrix < -0.7)))

ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.set_xticklabels(ax.get_yticklabels(), rotation=90)
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)


fig.savefig("/nfs/dust/cms/user/celottog/mttNN/corr.pdf", bbox_inches='tight')
