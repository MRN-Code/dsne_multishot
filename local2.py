"""
Created on Mon Jan 152017

@author: Deb
"""


import numpy as np
import argparse
import json
from tsneFunctions import normalize_columns, tsne, master_child



def updateL(Y, G):
    ''' It will take Y and IY of only local site data and return the updated Y'''
    return Y + G


def demeanL(Y, average_Y):
    ''' It will take Y and average_Y of only local site data and return the updated Y by subtracting IY'''

    pp = np.tile(np.mean(Y, 0), (Y.shape[0], 1))
    return Y - np.tile(average_Y, (Y.shape[0], 1))


def local_site2(args, compAvgError, computation_phase):
    ''' It will load local data and download remote data and place it on top. Then it will run tsne on combined data(shared + local) and return low dimensional shared Y and IY

           args (dictionary): {
               "shared_X" (str): file path to remote site data,
               "shared_Label" (str): file path to remote site labels
               "no_dims" (int): Final plotting dimensions,
               "initial_dims" (int): number of dimensions that PCA should produce
               "perplexity" (int): initial guess for nearest neighbor
               "shared_Y" (str):  the low-dimensional remote site data
               }


           Returns:
               computation_phase(local): It will return only low dimensional shared data from local site
               computation_phase(final): It will return only low dimensional local site data
               computation_phase(computation): It will return only low dimensional shared data Y and corresponding IY
           '''

    C =0
    if computation_phase is 'local':
        shared_X = np.loadtxt(args["shared_X"])
        shared_Y = np.loadtxt(args["shared_Y"])
        no_dims = args["no_dims"]
        initial_dims = args["initial_dims"]
        perplexity = args["perplexity"]
        local_site2.sharedRows, local_site2.sharedColumns = shared_X.shape


        # load high dimensional site 1 data
        parser = argparse.ArgumentParser(
            description='''read in coinstac args for local computation''')
        parser.add_argument('--run', type=json.loads, help='grab coinstac args')
        localSite2_Data = ''' {
            "site2_Data": "Site_2_Mnist_X.txt",
            "site2_Label": "Site_2_Label.txt"
        } '''
        site2args = parser.parse_args(['--run', localSite2_Data])
        Site2Data = np.loadtxt(site2args.run["site2_Data"])

        # create combinded list by local and remote data
        combined_X = np.concatenate((shared_X, Site2Data), axis=0)
        combined_X = normalize_columns(combined_X)

        # create low dimensional position
        combined_Y = np.random.randn(combined_X.shape[0], no_dims)
        combined_Y[:shared_Y.shape[0], :] = shared_Y

        local_site2.Y, local_site2.dY, local_site2.iY, local_site2.gains, local_site2.P, local_site2.n = tsne(
            combined_X,
            combined_Y,
            local_site2.sharedRows,
            no_dims=no_dims,
            initial_dims=initial_dims,
            perplexity=perplexity,
            computation_phase=computation_phase)

        # save local site sharedIY data into file
        with open("site2SharedY.txt", "w") as f1:
            for i in range(0,  local_site2.sharedRows):
                f1.write(str(local_site2.Y[i][0]) + '\t')
                f1.write(str(local_site2.Y[i][1]) + '\n')

        # pass data to remote in json format
        localJsonY = ''' {"localSite1SharedIY": "site2SharedY.txt"} '''
        sharedY = parser.parse_args(['--run', localJsonY])

        return (sharedY.run)


    if computation_phase is 'computation':
        parser = argparse.ArgumentParser(description='''read in coinstac args for local computation''')
        parser.add_argument('--run', type=json.loads, help='grab coinstac args')
        shared_Y = np.loadtxt(args["shared_Y"])
        local_site2.Y[:local_site2.sharedRows, :] = shared_Y;
        compAvgError1 = parser.parse_args(['--run', compAvgError])
        C = compAvgError1.run['output']['error']
        demeanAvg = (np.mean(local_site2.Y, 0));
        demeanAvg[0] = compAvgError1.run['output']['avgX']
        demeanAvg[1] = compAvgError1.run['output']['avgY']
        local_site2.Y = demeanL(local_site2.Y, demeanAvg)

        local_site2.Y, iY, local_site2.Q, C, local_site2.P = master_child(local_site2.Y, local_site2.dY, local_site2.iY,
                                                                          local_site2.gains, local_site2.n,
                                                                          local_site2.sharedRows, local_site2.P, iter,
                                                                          C)
        local_site2.Y[local_site2.sharedRows:, :] = updateL(local_site2.Y[local_site2.sharedRows:, :],
                                                            local_site2.iY[local_site2.sharedRows:, :])

        # save local site sharedY data into file
        with open("site2SharedY.txt", "w") as f1:
            for i in range(0, local_site2.sharedRows):
                f1.write(str(local_site2.Y[i][0]) + '\t')
                f1.write(str(local_site2.Y[i][1]) + '\n')

        # pass data to remote in json format
        localJson = ''' {"localSite2SharedY": "site2SharedY.txt"} '''
        sharedY = parser.parse_args(['--run', localJson])

        # save local site sharedIY data into file
        with open("site2SharedIY.txt", "w") as f1:
            for i in range(0, local_site2.sharedRows):
                f1.write(str(local_site2.iY[i][0]) + '\t')
                f1.write(str(local_site2.iY[i][1]) + '\n')

        # pass data to remote in json format
        localJsonIY = ''' {"localSite2SharedIY": "site2SharedIY.txt"} '''
        sharedIY = parser.parse_args(['--run', localJsonIY])

        meanValue = (np.mean(local_site2.Y, 0));
        comp = {'output': {'MeanX': meanValue[0], 'MeanY': meanValue[1], 'error': C}}

        return (sharedY.run, sharedIY.run, json.dumps(comp, sort_keys=True, indent=4, separators=(',', ':')))


    if computation_phase is 'final':
        parser = argparse.ArgumentParser(description='''read in coinstac args for local computation''')
        parser.add_argument('--run', type=json.loads, help='grab coinstac args')
        with open("local_site2Y.txt", "w") as f1:
            for i in range(local_site2.sharedRows, len(local_site2.Y)):
                f1.write(str(local_site2.Y[i][0]) + '\t')
                f1.write(str(local_site2.Y[i][1]) + '\n')

            localJsonY = ''' {"localSite2": "local_site2Y.txt"} '''
            sharedY = parser.parse_args(['--run', localJsonY])

        return (sharedY.run) 

