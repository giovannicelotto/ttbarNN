def getNames(nFiles, maxEvents, nDense, nNodes):
    inputName           = str(nFiles)+"*"+((str(int(maxEvents/1000))+"k") if maxEvents is not None else 'None')
    additionalName     = str(nFiles)+"*"+((str(int(maxEvents/1000))+"k") if maxEvents is not None else 'None')+"_"+str(nDense)+"*"+str(nNodes).replace(", ", "_")
    dataPathFolder="/nfs/dust/cms/user/celottog/mttNN/npyData/"+inputName
    outFolder="/nfs/dust/cms/user/celottog/mttNN/outputs/"+additionalName

    return inputName, dataPathFolder, outFolder

def getNamesForBayes(nFiles, maxEvents):
    inputName           = str(nFiles)+"*"+((str(int(maxEvents/1000))+"k") if maxEvents is not None else 'None')
    dataPathFolder="/nfs/dust/cms/user/celottog/mttNN/npyData/"+inputName
    #outFolder="/nfs/dust/cms/user/celottog/mttNN/outputs/"+additionalName

    return dataPathFolder

def getNamesForHyperSearch(nFiles, maxEvents, nDense, nNodes):
    inputName           = str(nFiles)+"*"+((str(int(maxEvents/1000))+"k") if maxEvents is not None else 'None')
    outFolder="/nfs/dust/cms/user/celottog/mttNN/outputs/gridSearch/"
    dataPathFolder="/nfs/dust/cms/user/celottog/mttNN/npyData/"+inputName

    return inputName, dataPathFolder, outFolder


def printStatus(hp, outFolder=None):
        #print ("LoadData from "+dataPathFolder+"/flat_[..].npy")
        #print ("\t nFiles       = "+str(nFiles_), file=f)
        #print ("\t maxEvents    = "+str(maxEvents_), file=f)
        #print ("\t testFraction = "+str(testFraction_), file=f)
        #print ("\t valid split  = "+str(validation_split_), file=f)
        #print ("\t Epochs       = "+str(epochs_), file=f)
        if (outFolder==None):
            print ("\t learningRate = "+str(hp['learningRate']))  
            print ("\t batchSize    = "+str(hp['batchSize']))
            print ("\t validBatch   = "+str(hp['validBatchSize']))
            print ("\t nDense       = "+str(hp['nDense']))
            print ("\t nNodes       = "+str(hp['nNodes']))
            print ("\t regRate      = "+str(hp['regRate']))
            print ("\t activation   = "+str(hp['activation']))
        else:
            with open(outFolder+"/Info.txt", "w+") as f:
                print ("\t learningRate = "+str(hp['learningRate']), file=f)  
                print ("\t batchSize    = "+str(hp['batchSize']), file=f)
                print ("\t validBatch   = "+str(hp['validBatchSize']), file=f)
                print ("\t nDense       = "+str(hp['nDense']), file=f)
                print ("\t nNodes       = "+str(hp['nNodes']), file=f)
                print ("\t regRate      = "+str(hp['regRate']), file=f)
                print ("\t activation   = "+str(hp['activation']), file=f)
                #print ("\t patiencelR   = "+str(patiencelR_), file=f)
        #print ("\t patienceeS   = "+str(patienceeS_), file=f)
        #print ("\t dropout      = "+str(dropout_), file=f)
        #print ("\t expTau       = "+str(exp_tau_), file=f)'''
        return
 #frozen_graph = freeze_session(tf.compat.v1.keras.backend.get_session(), output_names=[out.op.name for out in model.outputs])
        #tf.compat.v1.train.write_graph(frozen_graph, outFolder+ modelName+'.pbtxt', as_text=True)
        #tf.compat.v1.train.write_graph(frozen_graph, outFolder+ modelName+'.pb', as_text=False)
        #print ("Saved model to",outFolder+'/'+year+'/'+modelName+'.pbtxt/.pb/.h5')