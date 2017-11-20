#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 08:59:22 2017

@author: dsaha
"""
import numpy as np
import json
import argparse

from remote import remote_site
#from local import local_site
from remote_computation import remote_operations

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='''read in coinstac args for remote computation''')
    parser.add_argument('--run', type=json.loads, help='grab coinstac args')


    sharedData = ''' {
        "shared_X": "Shared_Mnist_X.txt",
        "shared_Label": "Shared_Label.txt",
        "no_dims": 2,
        "initial_dims": 50,
        "perplexity" : 20.0,
        "max_iter" : 5
    } '''


    args = parser.parse_args(['--run', sharedData])

    #Remote output actually the "sharedData" with remote Y
    remote_output = remote_site(args.run, computation_phase='remote')

    #for iter in range(sharedData["max_iter"]):
    Y, local1Y,local2Y = remote_operations(remote_output, computation_phase='local')



