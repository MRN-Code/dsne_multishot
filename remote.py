"""
Created on Mon Jan 152017

@author: Deb
"""


import numpy as np
from tsneFunctions import normalize_columns, tsne


def remote_site(args, computation_phase):
    ''' It will receive parameters from dsne_multi_shot. After receiving parameters it will compute tsne on high dimensional remote data and pass low dimensional values of remote site data


       args (dictionary): {
            "shared_X" (str):  remote site data
            "shared_Label" (str): remote site labels
            "no_dims" (int): Final plotting dimensions
            "initial_dims" (int): number of dimensions that PCA should produce
            "perplexity" (int): initial guess for nearest neighbor
            "max_iter" (str):  maximum number of iterations during tsne computation
            }
       computation_phase (string): remote

       normalize_columns:
           Shared data is normalized through this function

       Returns:
           Return args will contain previous args value in addition of Y[low dimensional Y values] values of shared_Y.
       args(dictionary):  {
           "shared_X" (str):  remote site data,
           "shared_Label" (str):  remote site labels
           "no_dims" (int): Final plotting dimensions,
           "initial_dims" (int): number of dimensions that PCA should produce
           "perplexity" (int): initial guess for nearest neighbor
           "shared_Y" : the low-dimensional remote site data
           }
       '''

    shared_X = np.loadtxt(args["shared_X"])
    #    sharedLabel = np.loadtxt(args.run["shared_Label"])
    no_dims = args["no_dims"]
    initial_dims = args["initial_dims"]
    perplexity = args["perplexity"]

    shared_X = normalize_columns(shared_X)
    (sharedRows, sharedColumns) = shared_X.shape

    init_Y = np.random.randn(sharedRows, no_dims)

    # shared data computation in tsne
    shared_Y = tsne(
        shared_X,
        init_Y,
        sharedRows,
        no_dims,
        initial_dims,
        perplexity,
        computation_phase=computation_phase)

    with open("Y_values.txt", "w") as f:
        for i in range(0, len(shared_Y)):
            f.write(str(shared_Y[i][0]) + '\t')
            f.write(str(shared_Y[i][1]) + '\n')

    args["shared_Y"] = "Y_values.txt"

    return args
