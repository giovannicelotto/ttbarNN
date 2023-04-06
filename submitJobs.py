import numpy as np
import itertools
import os
def main():
    lr = ['0', '1', '2']
    bs = ['0', '1', '2']
    vbs = ['0', '1', '2']
    nN = ['0', '1']
    rR = ['0', '1', '2']
    af = ['0', '1']

    combinations = list(itertools.product(lr, bs, vbs, nN, rR, af))

# Submit a Condor job for each combination
    for combination in combinations:
        # Convert the combination to a string of command line arguments
        args = ' '.join(combination)
        
        # Construct the Condor submit file for this job
        submit_file = '''
            executable = /nfs/dust/cms/user/sewuchte/newDNN/myConda/envs/hepML/bin/python
            arguments = /nfs/dust/cms/user/celottog/mttNN/hyperSearch.py {}
            output = output.$(Process)
            error = error.$(Process)
            log = log.$(Process)
            request_cpus = 1
            request_memory = 10GB
            request_disk = 10GB
            queue
        '''.format(args)
        
        # Write the submit file to disk
        with open('submit_file.submit', 'w') as f:
            f.write(submit_file)
        
        # Submit the job to Condor
        os.system('condor_submit submit_file.submit')
        




    return

if __name__ == "__main__":
    main()