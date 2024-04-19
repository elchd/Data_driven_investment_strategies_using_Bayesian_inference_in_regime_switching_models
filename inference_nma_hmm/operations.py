#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 16:48:09 2023

@author: eleonore
"""

import os
if os.getcwd()[-len('inference_nma_hmm'):] != 'inference_nma_hmm': os.chdir(os.getcwd() + '/inference_nma_hmm/')

from prob_distrib import *
from random_sample_jit import *
from lma_hmm import *
from gibbs import *
from data import * 
from appli_recup import *
