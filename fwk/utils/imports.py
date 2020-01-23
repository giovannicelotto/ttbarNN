import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import ROOT

import os
# import tensorflow as tf
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_WARNINGS'] = 'off'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import keras

from keras import backend as K
# K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads =32, inter_op_parallelism_threads=32)))
import tensorflow as tf
import keras.backend as K
K.tensorflow_backend.set_session(tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))))


import uproot
import numpy as np
from lbn import LBN,LBNLayer
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
import six
import random
