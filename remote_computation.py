"""
Created on Mon Jan 152017

@author: Deb
"""


import numpy as np
from tsneFunctions import normalize_columns, tsne
from local1 import local_site1
from local2 import local_site2
import json
import argparse


def updateS(Y1,Y2, iY_Site_1, iY_Site_2):
    ''' It will collect Y and IY from local sites and return the updated Y

    args:
        Y1: low dimensional shared data from site 1
        Y2: low dimensional shared data from site 2
        iY_Site_1: Comes from site 1
        iY_Site_2: Comes from site 2

    Returns:
        Y: Updated shared value
    '''


    Y= (Y1 + Y2 )/2
    iY = (iY_Site_1 + iY_Site_2) / 2
    Y = Y + iY

    return Y

def demeanS(Y, average_Y):
    ''' Subtract Y(low dimensional shared value )by the average_Y and return the updated Y) '''


    return Y - np.tile(average_Y, (Y.shape[0], 1))


def remote_operations(args, computation_phase):
    '''
    args(dictionary):  {
        "shared_X"(str): remote site data,
        "shared_Label"(str): remote site labels
        "no_dims"(int): Final plotting dimensions,
        "initial_dims"(int): number of dimensions that PCA should produce
        "perplexity"(int): initial guess for nearest neighbor
        "shared_Y": the low - dimensional remote site data

    Returns:
        Y: the final computed low dimensional remote site data
        local1Yvalues: Final low dimensional local site 1 data
        local2Yvalues: Final low dimensional local site 2 data
    }
    '''

    parser = argparse.ArgumentParser(description='''read in coinstac args for remote computation''')
    parser.add_argument('--run', type=json.loads, help='grab coinstac args')
    Y = np.loadtxt(args["shared_Y"])
    average_Y = (np.mean(Y, 0));
    average_Y[0] =0; average_Y[1] =0; C =0;
    compAvgError = {'output': {'avgX': average_Y[0], 'avgY': average_Y[1], 'error': C}}
    localSite1SharedY = local_site1(args, json.dumps(compAvgError, sort_keys=True,indent=4,separators=(',' ,':')), computation_phase='local')
    localSite2SharedY = local_site2(args, json.dumps(compAvgError, sort_keys=True,indent=4,separators=(',' ,':')), computation_phase='local')


    for i in range(args["max_iter"]):
        numOfSites =0;
        # local site 1 computation
        localSite1SharedY, localSite1SharedIY, ExtractMeanErrorSite1 = local_site1(args, json.dumps(compAvgError, sort_keys=True,indent=4,separators=(',' ,':')), computation_phase='computation')
        Y1 = np.loadtxt(localSite1SharedY["localSite1SharedY"])
        IY1 = np.loadtxt(localSite1SharedIY["localSite1SharedIY"])
        meanError1 = parser.parse_args(['--run', ExtractMeanErrorSite1])
        average_Y = (np.mean(Y1, 0));
        average_Y[0] = meanError1.run['output']['MeanX']
        average_Y[1] = meanError1.run['output']['MeanY']
        C = meanError1.run['output']['error']
        numOfSites += 1



        # local site 2 computation
        localSite2SharedY, localSite2SharedIY, ExtractMeanErrorSite2 = local_site2(args, json.dumps(compAvgError, sort_keys=True,indent=4,separators=(',' ,':')), computation_phase='computation')
        Y2 = np.loadtxt(localSite2SharedY["localSite2SharedY"])
        IY2 = np.loadtxt(localSite2SharedIY["localSite2SharedIY"])

        meanError2 = parser.parse_args(['--run', ExtractMeanErrorSite2])
        average_Y[0] = average_Y[0] + meanError2.run['output']['MeanX']
        average_Y[1] = average_Y[0] + meanError2.run['output']['MeanY']
        C = C + (meanError2.run['output']['error'])
        numOfSites += 1;

        # Here two local sites are considered. That's why it is divided by 2
        average_Y/=2
        C/=2


        Y = updateS(Y1,Y2, IY1, IY2)

        Y = demeanS(Y, average_Y)
            
        

        with open("Y_values.txt", "w") as f:
            for i in range(0, len(Y)):
                f.write(str(Y[i][0]) + '\t')
                f.write(str(Y[i][1]) + '\n')

        args["shared_Y"] = "Y_values.txt"
    # call local site 1 and collect low dimensional shared value of Y
    Y1 = local_site1(args, json.dumps(compAvgError, sort_keys=True,indent=4,separators=(',' ,':')), computation_phase='final')
    local1Yvalues = np.loadtxt(Y1["localSite1"])

    # call local site 2 and collect low dimensional shared value of Y
    Y2 = local_site2(args, json.dumps(compAvgError, sort_keys=True,indent=4,separators=(',' ,':')), computation_phase='final')
    local2Yvalues = np.loadtxt(Y2["localSite2"])

    return Y, local1Yvalues, local2Yvalues
