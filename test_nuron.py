'''
file documentation
'''
###############################################
###############################################
#
# File / Package Import
#
###############################################
###############################################

from time import time
from matplotlib import pyplot
from collections import Counter
import pandas
import os

import warnings
warnings.simplefilter('ignore')

###############################################
###############################################
#
# path variables
#
###############################################
###############################################

string_path = 'C:\\Code\\\Development\\Python\\Seyi_Class\\data'
string_file_test_raw = 'test_technidus_clf.csv'
string_file_train_raw = 'train_technidus_clf.csv'

###############################################
###############################################
#
# load data
#
###############################################
###############################################

df_test_raw = pandas.read_csv(os.path.join(string_path, string_file_test_raw))
df_train_raw = pandas.read_csv(os.path.join(string_path, string_file_train_raw))

type(df_test_raw.dtypes)