from tabulate import tabulate
def getNames(nFiles, maxEvents, nDense, nNodes):
    inputName           = str(nFiles)+"*"+((str(int(maxEvents/1000))+"k") if maxEvents is not None else 'None')
    additionalName     = str(nFiles)+"*"+((str(int(maxEvents/1000))+"k") if maxEvents is not None else 'None')+"_"+str(nNodes).replace(", ", "_")
    dataPathFolder="/nfs/dust/cms/user/celottog/mttNN/npyData/"+inputName
    outFolder="/nfs/dust/cms/user/celottog/mttNN/outputs/"+additionalName

    return dataPathFolder, outFolder

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
    headers = ["Hyperparameter", "Value"]
    table = []

    for key, value in hp.items():
        table.append([key, value])

    if outFolder is None:
        print(tabulate(table, headers=headers))
    else:
        print(tabulate(table, headers=headers))
        with open(outFolder + "/Info.txt", "w+") as f:
            f.write(tabulate(table, headers=headers))

    return

def printStatusOld(hp, outFolder=None):
        if (outFolder==None):
            for key, value in hp.items():
                print("{}\t{}".format(key, value))
        else:
            with open(outFolder+"/Info.txt", "w+") as f:
                for key, value in hp.items():
                    print("{}\t{}".format(key, value), file=f)
                
        return
 #frozen_graph = freeze_session(tf.compat.v1.keras.backend.get_session(), output_names=[out.op.name for out in model.outputs])
        #tf.compat.v1.train.write_graph(frozen_graph, outFolder+ modelName+'.pbtxt', as_text=True)
        #tf.compat.v1.train.write_graph(frozen_graph, outFolder+ modelName+'.pb', as_text=False)
        #print ("Saved model to",outFolder+'/'+year+'/'+modelName+'.pbtxt/.pb/.h5')