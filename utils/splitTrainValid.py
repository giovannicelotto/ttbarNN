import numpy as np
def splitTrainValid(inX_train, outY_train, weights_train, lkrM_train, krM_train, totGen_train, mask_train, validation_split_, dataPathFolder):
    print("Splitting between training and validation...")
    divisor = int ((1-validation_split_)*len(inX_train))
    outY_valid      = outY_train[divisor:]
    inX_valid       = inX_train[divisor:]
    weights_valid   = weights_train[divisor:]
    lkrM_valid      = lkrM_train[divisor:]
    krM_valid       = krM_train[divisor:]
    totGen_valid    = totGen_train[divisor:]
    mask_valid      = mask_train[divisor:]
    
    inX_train       = inX_train[:divisor]
    outY_train      = outY_train[:divisor]
    weights_train   = weights_train[:divisor]
    lkrM_train      = lkrM_train[:divisor]
    krM_train       = krM_train[:divisor]
    totGen_train    = totGen_train[:divisor]
    mask_train      = mask_train[:divisor]
    #assert (outY_train[mask_train, 0] == totGen_train[mask_train]).all(), "Mask does not match after splitting training and validation"
    #assert (outY_valid[mask_valid, 0] == totGen_valid[mask_valid]).all(), "Mask does not match after splitting training and validation"
    assert len(inX_train)==len(outY_train)==len(weights_train)==len(mask_train), "Check lengths after splitting training and validation"
    assert len(inX_valid)==len(outY_valid)==len(weights_valid)==len(mask_valid), "Check lengths after splitting training and validation %d %d %d %d" %(len(inX_valid), len(outY_valid), len(weights_valid), len(mask_valid))    
    print("Number train events masked / total):", len(inX_train[mask_train,:]), len(inX_train))
    print("Number valid events masked / total):", len(inX_valid[mask_valid,:]), len(inX_valid))
 
    #jsTV = JSdist(outY_train[:len(outY_valid)-1,0], outY_valid[:len(outY_valid)-1,0])
    #print ("JS between KinR training and validation", str(jsTV[0])[:6], str(jsTV[1])[:6])       
    '''if (saveData):
        np.save(dataPathFolder+"/testing/inX"    + "_train.npy", inX_train)
        np.save(dataPathFolder+"/testing/outY"   + "_train.npy", outY_train)
        np.save(dataPathFolder+"/testing/weights"+ "_train.npy", weights_train)
        np.save(dataPathFolder+"/testing/inX"    + "_valid.npy", inX_valid)
        np.save(dataPathFolder+"/testing/outY"   + "_valid.npy", outY_valid)
        np.save(dataPathFolder+"/testing/weights"+ "_valid.npy", weights_valid)'''
    return inX_train, outY_train, weights_train, lkrM_train, krM_train, mask_train, inX_valid, outY_valid, weights_valid, lkrM_valid, krM_valid, mask_valid

def computePredicted(inX_train, inX_valid, inX_test, dataPathFolder, model):
    print("Computing predictions...")
    y_predicted = model.predict(inX_test)
    y_predicted_train = model.predict(inX_train)
    y_predicted_valid = model.predict(inX_valid)
    print("Saving predictions...")
    np.save(dataPathFolder+"/testing/regY"   + "_test.npy", y_predicted)
    np.save(dataPathFolder+"/testing/regY"   + "_train.npy", y_predicted_train)
    np.save(dataPathFolder+"/testing/regY"   + "_valid.npy", y_predicted_valid)
    return y_predicted_train, y_predicted_valid, y_predicted

def saveTrainValidTest(dataPathFolder, inX_train, outY_train, weights_train, inX_valid, outY_valid, weights_valid,inX_test, outY_test, weights_test, totGen_test):
    np.save(dataPathFolder+"/testing/inX"    + "_train.npy", inX_train)
    np.save(dataPathFolder+"/testing/outY"   + "_train.npy", outY_train)
    np.save(dataPathFolder+"/testing/weights"+ "_train.npy", weights_train)
    np.save(dataPathFolder+"/testing/inX"    + "_valid.npy", inX_valid)
    np.save(dataPathFolder+"/testing/outY"   + "_valid.npy", outY_valid)
    np.save(dataPathFolder+"/testing/weights"+ "_valid.npy", weights_valid)
    np.save(dataPathFolder+"/testing/inX"    + "_test.npy", inX_test)
    np.save(dataPathFolder+"/testing/outY"   + "_test.npy", outY_test)
    np.save(dataPathFolder+"/testing/weights"+ "_test.npy", weights_test)
