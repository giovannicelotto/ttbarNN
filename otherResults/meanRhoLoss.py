import numpy as np
import matplotlib.pyplot as plt
# Open the file for reading
with open('/nfs/dust/cms/user/celottog/mttNN/noRotRes.txt', 'r') as file:
    # Initialize variables to store column totals
    loss = []
    rho = []
    
    lines = file.readlines()
    

    # Loop through each line in the file
    for line in lines:
        # Split the line by tab or whitespace
        
        values = line.replace('\t\t', '\t').strip().split('\t')
        
        # Skip empty lines
        if len(values) != 2:
            continue
        

        # Extract values for each column
        loss.append(float(values[0]))
        rho.append(float(values[1]))


    lossNR = np.array(loss)
    rhoNR  = np.array(rho)

    errLossNR =  np.std(lossNR)/np.sqrt(len(lossNR))
    errRhoNR  = np.std(rhoNR)/np.sqrt(len(rhoNR))

    # Print the averages
    print("WITHOUT ROTATION")
    print("Loss Average:", np.mean(lossNR), "\t\t+-\t",errLossNR)
    print("Rho  Average:", np.mean(rhoNR), "\t+-\t", errRhoNR)


# Open the file for reading
with open('/nfs/dust/cms/user/celottog/mttNN/rotRes.txt', 'r') as file:
    # Initialize variables to store column totals
    loss = [ ]
    rho  = []
    lines = file.readlines()
    

    # Loop through each line in the file
    for line in lines:
        # Split the line by tab or whitespace
        
        values = line.replace('\t\t', '\t').strip().split('\t')
        
        # Skip empty lines
        if len(values) != 2:
            continue
        

       # Extract values for each column
        loss.append(float(values[0]))
        rho.append(float(values[1]))


    lossR = np.array(loss)
    rhoR  = np.array(rho)
    errLossR =  np.std(lossR)/np.sqrt(len(lossR))
    errRhoR  = np.std(rhoR)/np.sqrt(len(rhoR))
    # Print the averages
    print("WITH ROTATION")
    print("Loss Average:", np.mean(lossR), "\t\t+-\t",errLossR)
    print("Rho  Average:", np.mean(rhoR), "\t+-\t", errRhoR)

    fig, ax = plt.subplots(1, 2, figsize=(10, 2), constrained_layout=True)
    ax[0].errorbar(x=1, y=np.mean(lossR), yerr= errLossR, marker = 'o', markersize=8)
    ax[0].errorbar(x=2, y=np.mean(lossNR), yerr= errLossNR, marker = 'o', markersize=8)

    ax[0].set_ylabel("MSE", fontsize=18)
    tick_labels = ['', 'With Rotation', 'Without Rotation', '']  # List of custom tick labels
    x = [0.5, 1, 2, 2.5]
    ax[0].set_xticks(x, tick_labels)

    ax[1].errorbar(x=1, y=np.mean(rhoR), yerr= errRhoR, marker = 'o', markersize=8)
    ax[1].errorbar(x=2, y=np.mean(rhoNR), yerr= errRhoNR, marker = 'o', markersize=8)

    ax[1].set_ylabel(r"$\rho$", fontsize=18)
    ax[1].set_xticks(x, tick_labels)
    fig.savefig("/nfs/dust/cms/user/celottog/mttNN/otherResults/rotationLossRho.pdf", bbox_inches='tight')