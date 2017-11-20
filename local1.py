#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 20:49:04 2017

@author: gazula
"""
import numpy as np
import argparse
import json
from tsneFunctions import normalize_columns, tsne, master_child




def updateL(Y, G):
    return Y + G


def demeanL(Y, average_Y):
    pp = np.tile(np.mean(Y, 0), (Y.shape[0], 1))
    return Y - np.tile(average_Y, (Y.shape[0], 1))


def local_site1(args, compAvgError, computation_phase):
    C= 0
    if computation_phase is 'local':
        shared_X = np.loadtxt(args["shared_X"])
        shared_Y = np.loadtxt(args["shared_Y"])
        no_dims = args["no_dims"]
        initial_dims = args["initial_dims"]
        perplexity = args["perplexity"]
        local_site1.sharedRows, local_site1.sharedColumns = shared_X.shape

        parser = argparse.ArgumentParser(
            description='''read in coinstac args for local computation''')
        parser.add_argument('--run', type=json.loads, help='grab coinstac args')
        localSite1_Data = ''' {
            "site1_Data": "Site_1_Mnist_X.txt",
            "site1_Label": "Site_1_Label.txt"
        } '''
        site1args = parser.parse_args(['--run', localSite1_Data])
        Site1Data = np.loadtxt(site1args.run["site1_Data"])

        # create combinded list by local and remote data
        combined_X = np.concatenate((shared_X, Site1Data), axis=0)
        combined_X = normalize_columns(combined_X)

        # create low dimensional position
        combined_Y = np.random.randn(combined_X.shape[0], no_dims)
        combined_Y[:shared_Y.shape[0], :] = shared_Y

        local_site1.Y, local_site1.dY, local_site1.iY, local_site1.gains, local_site1.P, local_site1.n = tsne(
            combined_X,
            combined_Y,
            local_site1.sharedRows,
            no_dims=no_dims,
            initial_dims=initial_dims,
            perplexity=perplexity,
            computation_phase=computation_phase)

        # save local site sharedIY data into file
        with open("site1SharedY.txt", "w") as f1:
            for i in range(0,  local_site1.sharedRows):
                f1.write(str(local_site1.Y[i][0]) + '\t')
                f1.write(str(local_site1.Y[i][1]) + '\n')

        # pass data to remote in json format
        localJsonY = ''' {"localSite1SharedIY": "site1SharedY.txt"} '''
        sharedY = parser.parse_args(['--run', localJsonY])

        return (sharedY.run)


    if computation_phase is 'computation':
        parser = argparse.ArgumentParser(description='''read in coinstac args for local computation''')
        parser.add_argument('--run', type=json.loads, help='grab coinstac args')
        shared_Y = np.loadtxt(args["shared_Y"])
        local_site1.Y[:local_site1.sharedRows, :] = shared_Y;
        compAvgError1 = parser.parse_args(['--run', compAvgError])
        C = compAvgError1.run['output']['error']
        demeanAvg = (np.mean(local_site1.Y, 0));
        demeanAvg[0]= compAvgError1.run['output']['avgX']
        demeanAvg[1] = compAvgError1.run['output']['avgY']
        local_site1.Y = demeanL(local_site1.Y, demeanAvg)

        local_site1.Y, iY, local_site1.Q, C, local_site1.P = master_child(local_site1.Y, local_site1.dY, local_site1.iY, local_site1.gains, local_site1.n, local_site1.sharedRows, local_site1.P, iter, C)
        local_site1.Y[local_site1.sharedRows:, :] = updateL(local_site1.Y[local_site1.sharedRows:, :], local_site1.iY[local_site1.sharedRows:, :])

        # save local site sharedY data into file
        with open("site1SharedY.txt", "w") as f1:
            for i in range(0,local_site1.sharedRows):
                f1.write(str(local_site1.Y[i][0]) + '\t')
                f1.write(str(local_site1.Y[i][1]) + '\n')

        # pass data to remote in json format
        localJson = ''' {"localSite1SharedY": "site1SharedY.txt"} '''
        sharedY = parser.parse_args(['--run', localJson])

        # save local site sharedIY data into file
        with open("site1SharedIY.txt", "w") as f1:
            for i in range(0, local_site1.sharedRows):
                f1.write(str(local_site1.iY[i][0]) + '\t')
                f1.write(str(local_site1.iY[i][1]) + '\n')

        # pass data to remote in json format
        localJsonIY = ''' {"localSite1SharedIY": "site1SharedIY.txt"} '''
        sharedIY = parser.parse_args(['--run', localJsonIY])


        meanValue = (np.mean(local_site1.Y, 0));
        comp = {'output': {'MeanX' : meanValue[0], 'MeanY' : meanValue[1], 'error': C}}

        return (sharedY.run, sharedIY.run, json.dumps(comp, sort_keys=True,indent=4,separators=(',' ,':')))

    if computation_phase is 'final':
        parser = argparse.ArgumentParser(description='''read in coinstac args for local computation''')
        parser.add_argument('--run', type=json.loads, help='grab coinstac args')
        with open("local_site1Y.txt", "w") as f1:
            for i in range(local_site1.sharedRows, len(local_site1.Y)):
                f1.write(str(local_site1.Y[i][0]) + '\t')
                f1.write(str(local_site1.Y[i][1]) + '\n')

            localJsonY = ''' {"localSite1": "local_site1Y.txt"} '''
            sharedY = parser.parse_args(['--run', localJsonY])

        return(sharedY.run)
