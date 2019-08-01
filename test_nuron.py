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
import numpy
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

###############################################
###############################################
#
# basic data exploration
#
###############################################
###############################################

df_test_raw.dtypes
df_train_raw.dtypes

set(df_test_raw.columns) & set(df_train_raw.columns)

type(df_train_raw['TotalChildren'].dtype)
df_train_raw['TotalChildren'].dtype == numpy.dtype('int64')
str(df_train_raw['TotalChildren'].dtype)

type(df_train_raw['MaritalStatus'].dtype)
df_train_raw['MaritalStatus'].dtype == numpy.dtype('object')