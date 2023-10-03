import numpy as np
from utils.plot import getRecoBin
y_predicted = np.load("/nfs/dust/cms/user/celottog/mttNN/npyData/1*None/testing/y_predicted_test.npy")
totGen = np.load("/nfs/dust/cms/user/celottog/mttNN/npyData/1*None/testing/totGen_test.npy")
Ry_predicted = np.load("/nfs/dust/cms/user/celottog/mttNN/outputs/comparisons/y_predicted_1f.npy")
RtotGen = np.load("/nfs/dust/cms/user/celottog/mttNN/outputs/comparisons/totGen_test_1f.npy")
weights = np.load("/nfs/dust/cms/user/celottog/mttNN/npyData/1*None/testing/weights_test.npy")


print((totGen==RtotGen).all())
# RESPONSE MATRIX
recoBin = getRecoBin()
nbin = len(recoBin)-1
matrix = np.empty((nbin, nbin))
for i in range(nbin):
    maskGen = (totGen >= recoBin[i]) & (totGen < recoBin[i+1])
    
    normalization = np.sum(weights[maskGen])
    for j in range(nbin):
        
        maskRecoGen = (y_predicted >= recoBin[j]) & (y_predicted < recoBin[j+1]) & maskGen
        entries = np.sum(weights[maskRecoGen])
        
        if normalization == 0:
            matrix[j, i] = 0
        else:
            matrix[j, i] = entries/normalization

# UNFOLDING
dnnMask_test = y_predicted>-998
dnnCounts, b_ = np.histogram(y_predicted[dnnMask_test] , bins=getRecoBin(), weights=weights[dnnMask_test], density=False)

# COVARIANCE MATRIX
dy = np.diag(dnnCounts)
try:
    A_inv = np.linalg.inv(matrix)
    cov = (A_inv.dot(dy)).dot(A_inv.T)
    res = np.linalg.det(cov)

except:
    res = -999

    print("det not computed")

print(res)