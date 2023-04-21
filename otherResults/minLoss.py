import numpy as np
# Open the file for reading
with open('/nfs/dust/cms/user/celottog/mttNN/otherResults/randomResults1904.txt', 'r') as file:
    # Initialize variables to store column totals
    
    lines = file.readlines()
    
    mse = []
    corr = []
    model = []
    separators = "\t"
    # Loop through each line in the file
    for line in lines:
        # Split the line by tab or whitespace
        
        values = line.replace('\t\t', '\t').strip(separators).split()
        if (len(values)<2):
            print("line skipped")
            continue

        # Extract values for each column
        mse.append(float(values[0]))
        corr.append(float(values[1]))
        model.append(values[2:])
        
    mse = np.array(mse)
    corr = np.array(corr)
    model = np.array(model)
    # first 5 minima
    min_ind = np.argsort(mse)[:10]
    for i in min_ind:
        print(mse[i], corr[i], model[i])


