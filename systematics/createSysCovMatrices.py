import glob
import numpy as np
import sys
folder_path = '/nfs/dust/cms/user/celottog/mttNN'
sys.path.append(folder_path)
from utils.helpers4P import getXSections
import pickle
from utils.plot import getRecoBin


def main(createDict):
    methods = ['dnn', 'loose', 'kin']
    skip = [67, 69, 67] #number of characters to skip (path)
    for id in range(len(methods)):
        outFolder = "/nfs/dust/cms/user/celottog/mttNN/systematics/responseMatrices/"+methods[id]
        if createDict:
            matrixPath = "/nfs/dust/cms/user/celottog/mttNN/systematics/responseMatrices/"+methods[id]
            
            xsections = getXSections()
            matrixPaths = glob.glob(matrixPath+"/*.npy")
            
            matrixNames = []
            for i in range(len(matrixPaths)):
                # Load the paths of the response matrices skipping the first part of the path
                matrixNames.append(matrixPaths[i])
                matrixNames[-1] = matrixNames[-1][skip[id]:]
                
            assert len(matrixNames)==len(matrixPaths)
            # Load nominal values of bkg and data
            bkgCounts   = np.load("/nfs/dust/cms/user/celottog/realData/outputs/Counts/bkgRecoNorm.npy", allow_pickle=True).item()
            dataCounts  = np.load("/nfs/dust/cms/user/celottog/realData/outputs/Counts/dataCounts.npy")[id]
            for i in xsections.keys():
                dataCounts   = dataCounts   - bkgCounts[i][id]
            # bin-by-bin difference in order to unfold
            A_nom = np.load(matrixPath+"/weights.npy")
            A_nomInv = np.linalg.inv(A_nom)
            cNonVar = A_nomInv.dot(dataCounts)
            variations = {}
            relativeVariations = {}
            for idx in range(len(matrixPaths)):
                
                #if matrixNames[idx]=='weights.npy':
                #    print("Skipped 1/2")
                #    continue
                if matrixNames[idx]=='y_predicted.npy':
                    print("Skipped 1/2")
                    continue
                if matrixNames[idx]=='systCovMatrix.npy':
                    print("Skipped 2/2")
                    continue
                else:
                    A_var = np.load(matrixPaths[idx])
                    A_varInv = np.linalg.inv(A_var)
                    
                    cVar    = A_varInv.dot(dataCounts)
                    diff    = (cVar - cNonVar)
                    relativeDiff = (cVar - cNonVar)/cNonVar
                    variations[matrixNames[idx].replace('.npy', '')] = diff #remove .npy from the name
                    relativeVariations[matrixNames[idx].replace('.npy', '')] = relativeDiff #remove .npy from the name

                    #print(matrixNames[idx], relativeDiff)

                    with open(outFolder+"/variations.pkl", "wb") as file:
                        pickle.dump(variations, file)
                    with open(outFolder+"/relVariations.pkl", "wb") as file:
                        pickle.dump(relativeVariations, file)
        with open(outFolder+"/variations.pkl", "rb") as file:
            variations = pickle.load(file)
        print(len(variations.keys()))
        
        
        index=1
        covSyst = {}
        for key in variations:
            
            if key[-2:] == 'up':
                
                if 'bfrag' in key:
                    covSyst[key.replace('_up', '')] = (np.outer(variations[key], variations[key]) + np.outer(variations[key.replace('_up', '_down')], variations[key.replace('_up', '_down')]) + np.outer(variations[key.replace('_up', '_central')], variations[key.replace('_up', '_central')]))/3
                if key.replace('_up', '_down') in variations.keys():
                    
                    print(index, key.replace('_up', ''))
                    index=index+1
                    
                    covSyst[key.replace('_up', '')] = (np.outer(variations[key], variations[key]) +np.outer(variations[key.replace('_up', '_down')], variations[key.replace('_up', '_down')]))/2
            elif key[-4:] == 'down':
                continue
            elif 'var_pdf_up' in key:
                
                covSyst[key.replace('up_', '')] = (np.outer(variations[key], variations[key]) + np.outer(variations[key.replace('up', 'down')], variations[key.replace('up', 'down')]))/2
            elif ('var_top_pt' in key) | ('pdf_central' in key)  | ('var_bfrag_peterson' in key):
                
                index=index+1
                covSyst[key] = np.outer(variations[key], variations[key])
            

        covMatrix = np.zeros((len(getRecoBin())-1, len(getRecoBin())-1))
        for k in covSyst.keys():
            covMatrix = covMatrix + covSyst[k]
        print(np.sqrt(np.diag(covMatrix)))
        np.save(outFolder+"/systCovMatrix.npy", covMatrix)

    return

if __name__ =='__main__':
    print(sys.argv )
    if len(sys.argv) > 1:
        createDict = bool(int(sys.argv[1]))
    else:
        createDict = False
    main(createDict)