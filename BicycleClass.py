'''
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
# Classes
#
###############################################
###############################################

class BicycleAnalysis(object):
    '''
    This class analyzes of a Bicycle purchase dataset.

    Requirements:
    package pandas
    package scikit-learn

    Methods:
    ??

    Attributes:
    ??
    '''

    #--------------------------------------------------------------------------#
    # constructor
    #--------------------------------------------------------------------------#

    def __init__(self):
        '''
        class constructor

        Requirements:
        package pymssql

        Inputs:
        None
        Type: n/a
        Desc: n/a

        Important Info:
        1. ??

        Objects and Properties:
        bool_is_connected
        Type: boolean
        Desc: flag to help the user to determine if the connection is generated
        '''

        #--------------------------------------------------------------------------#
        # file and path variables
        #--------------------------------------------------------------------------#

        self.string_data_path = os.path.abspath('./data')
        self.string_model_path = os.path.abspath('./models')
        self.string_file_test = 'test_technidus_clf.csv'
        self.string_file_train = 'train_technidus_clf.csv'
        self.set_source_files = set([self.string_file_test, self.string_file_train])

        #--------------------------------------------------------------------------#
        # data containers
        #--------------------------------------------------------------------------#

        self.df_test_raw = None
        self.df_train_raw = None

    #--------------------------------------------------------------------------#
    # main method
    #--------------------------------------------------------------------------#

    def main(self):
        '''
        this method is the main method for the analysis

        Requirements:
        None

        Inputs:
        None
        Type: n/a
        Desc: n/a

        Important Info:
        None

        Return:
        None
        Type: n/a
        Desc:
        '''

        #--------------------------------------------------------------------------------#
        # objects declarations
        #--------------------------------------------------------------------------------#

        #--------------------------------------------------------------------------------#
        # time declarations
        #--------------------------------------------------------------------------------#

        #--------------------------------------------------------------------------------#
        # lists declarations
        #--------------------------------------------------------------------------------#

        #--------------------------------------------------------------------------------#
        # variables declarations
        #--------------------------------------------------------------------------------#

        ###############################################
        ###############################################
        #
        # Load Data
        #
        ###############################################
        ###############################################

        bool_missing_data = self.load_data()

        #--------------------------------------------------------------------------#
        # sub-section comment
        #--------------------------------------------------------------------------#

        ###############################################
        ###############################################
        #
        # sectional comment
        #
        ###############################################
        ###############################################

        #--------------------------------------------------------------------------#
        # variable / object cleanup
        #--------------------------------------------------------------------------#

        #--------------------------------------------------------------------------#
        # return value
        #--------------------------------------------------------------------------#

        return

    #--------------------------------------------------------------------------#
    # callable methods
    #--------------------------------------------------------------------------#

    def load_data(self, m_bool_filter_columns = False):
        '''
        this method loads the data from the data directory

        Requirements:
        package pandas
        package os

        Inputs:
        None
        Type: n/a
        Desc: n/a

        Important Info:
        None

        Return:
        None
        Type: n/a
        Desc:
        '''
        
        #--------------------------------------------------------------------------------#
        # objects declarations
        #--------------------------------------------------------------------------------#

        #--------------------------------------------------------------------------------#
        # time declarations
        #--------------------------------------------------------------------------------#

        #--------------------------------------------------------------------------------#
        # lists declarations
        #--------------------------------------------------------------------------------#

        #--------------------------------------------------------------------------------#
        # variables declarations
        #--------------------------------------------------------------------------------#

        ###############################################
        ###############################################
        #
        # walk through the data directory
        #
        ###############################################
        ###############################################

        for string_root, list_dir, list_files in os.walk(self.string_data_path):
            set_files_in_data_dir = set(list_files)
            break

        bool_missing_data_file = False
        list_missing_files = list()
        for string_file in self.set_source_files:
            if string_file not in set_files_in_data_dir:
                bool_missing_data_file = True
                list_missing_files.append(string_file)

        ###############################################
        ###############################################
        #
        # load data or print out missing files
        #
        ###############################################
        ###############################################

        if bool_missing_data_file:
            string_mf_templ = 'missing {0} data file'
            for string_df in list_missing_files:
                print(string_mf_templ.format(string_df))
        else:
            self.df_test_raw = pandas.read_csv(
                os.path.join(self.string_data_path, self.string_file_test))
            self.df_train_raw = pandas.read_csv(
                os.path.join(self.string_data_path, self.string_file_train))

        ###############################################
        ###############################################
        #
        # filter columns
        #
        ###############################################
        ###############################################

        if not bool_missing_data_file and m_bool_filter_columns:
            # find the common columns to both sets
            set_train_cols = set(self.df_train_raw.columns)
            set_test_cols = set(self.df_test_raw.columns)
            list_common_cols = list(set_train_cols & set_test_cols)
            
            # filter dataframes
            self.df_test_raw = self.df_test_raw[list_common_cols]
            self.df_train_raw = self.df_train_raw[list_common_cols]
        
        #--------------------------------------------------------------------------#
        # return value
        #--------------------------------------------------------------------------#

        return not bool_missing_data_file

    def compare_train_test(self, m_list_flags = ['categorical_columns']):
        '''
        plots both train and test data based on the flags passed

        Requirements:
        package pandas
        package os
        package matplotlib.pyplot

        Inputs:
        m_string_flags
        Type: list
        Desc: flag for the type of analysis to do
            'categorical_columns' -> bar charts of categorical columns
            'prediction_column' -> bar chart of prediction column

        Important Info:
        None

        Return:
        None
        Type: n/a
        Desc:
        '''
        
        #--------------------------------------------------------------------------------#
        # objects declarations
        #--------------------------------------------------------------------------------#

        #--------------------------------------------------------------------------------#
        # time declarations
        #--------------------------------------------------------------------------------#

        #--------------------------------------------------------------------------------#
        # lists declarations
        #--------------------------------------------------------------------------------#

        list_test_cols = self.df_test_raw.columns.values.tolist()

        #--------------------------------------------------------------------------------#
        # variables declarations
        #--------------------------------------------------------------------------------#

        string_status_templ = 'Comparing train and test data sets.\n'

        ###############################################
        ###############################################
        #
        # start the data exploration
        #
        ###############################################
        ###############################################

        print(string_status_templ)

        #--------------------------------------------------------------------------#
        # loop through analyses
        #--------------------------------------------------------------------------#

        for string_analysis in m_list_flags:
            # categorical analysis        
            if string_analysis == 'categorical_columns':
                for string_column in list_test_cols:
                    if self.df_test_raw[string_column].dtype == 'object' and \
                        self.df_train_raw[string_column].dtype == 'object':
                        # get plot values
                        series_test = self._ctt_calc_cat_values(
                            self.df_test_raw[string_column])
                        series_train = self._ctt_calc_cat_values(
                            self.df_train_raw[string_column])
                        
                        # create figure and axes
                        int_max = 10
                        fig, axes = pyplot.subplots(1, 2)
                        fig.suptitle(string_column)
                        
                        # plot train / test data
                        axes[0] = self._ctt_plot_train_test(axes[0], series_train,
                            int_max, 'Train')
                        axes[1] = self._ctt_plot_train_test(axes[1], series_test,
                            int_max, 'Test')

                        # show plots
                        pyplot.show()
            
            # predicted column
            if string_analysis == 'prediction_column':
                # get train / test values
                string_bb_column = 'BikeBuyer'
                series_bb_train = self._ctt_calc_cat_values(
                    self.df_train_raw[string_bb_column])

                # plot predicted column
                fig, ax = pyplot.subplots()
                ax = self._ctt_plot_train_test(ax, series_bb_train,
                    2, 'Train')
                fig.suptitle('BikeBuyer Yes (1) or No (0)')
                
                # show plots
                pyplot.show()
    
        #--------------------------------------------------------------------------#
        # return value
        #--------------------------------------------------------------------------#

        return

    def basic_exploration(self):
        '''
        this method compares the common columns of the train and test sets

        Requirements:
        package pandas
        package matplotlib.pyplot

        Inputs:
        None
        Type: n/a
        Desc: n/a

        Important Info:
        None

        Return:
        None
        Type: n/a
        Desc: n/a
        '''

        #--------------------------------------------------------------------------------#
        # objects declarations
        #--------------------------------------------------------------------------------#

        #--------------------------------------------------------------------------------#
        # time declarations
        #--------------------------------------------------------------------------------#

        #--------------------------------------------------------------------------------#
        # lists declarations
        #--------------------------------------------------------------------------------#

        set_train_cols = set(self.df_train_raw.columns)
        set_test_cols = set(self.df_test_raw.columns)
        list_common_cols = list(set_train_cols & set_test_cols)
        if 'BikeBuyer' in list_common_cols:
            list_common_cols.remove('BikeBuyer')

        #--------------------------------------------------------------------------------#
        # variables declarations
        #--------------------------------------------------------------------------------#

        ###############################################
        ###############################################
        #
        # start
        #
        ###############################################
        ###############################################

        #--------------------------------------------------------------------------#
        # dataframe info
        #--------------------------------------------------------------------------#

        print('Training dataset info')
        self.df_train_raw[list_common_cols].info()

        print('\nTest dataset info')
        self.df_test_raw[list_common_cols].info()

        #--------------------------------------------------------------------------#
        # first three records of train and test set
        #--------------------------------------------------------------------------#

        print('\nFirst three records of training data')
        print(self.df_train_raw[list_common_cols].head(3))

        print('\nFirst three records of test data')
        print(self.df_test_raw[list_common_cols].head(3))

        #--------------------------------------------------------------------------#
        # return value
        #--------------------------------------------------------------------------#

        return
    
    def pre_process_data(self, m_df):
        '''
        this method pre-processes the data for modeling

        Requirements:
        package pandas

        Inputs:
        m_df
        Type: pandas.DataFrame
        Desc: data for either train or test to process

        Important Info:
        None

        Return:
        object
        Type: pandas.DataFrame
        Desc: processed data for modeling
        '''

        #--------------------------------------------------------------------------------#
        # objects declarations
        #--------------------------------------------------------------------------------#

        #--------------------------------------------------------------------------------#
        # time declarations
        #--------------------------------------------------------------------------------#

        #--------------------------------------------------------------------------------#
        # sequence declarations
        #--------------------------------------------------------------------------------#

        set_train_cols = set(self.df_train_raw.columns.values.tolist())
        set_test_cols = set(self.df_test_raw.columns.values.tolist())
        list_common_cols = list(set_train_cols & set_test_cols)

        #--------------------------------------------------------------------------------#
        # variables declarations
        #--------------------------------------------------------------------------------#

        string_bb_col = 'BikeBuyer'
        
        ###############################################
        ###############################################
        #
        # Start
        #
        ###############################################
        ###############################################

        #--------------------------------------------------------------------------#
        # set-up for processing
        #--------------------------------------------------------------------------#

        df_prep = m_df[list_common_cols]
        if string_bb_col in list_common_cols:
            series_bb = df_prep[string_bb_col]
            df_prep = df_prep.drop(string_bb_col)

        ###############################################
        ###############################################
        #
        # sectional comment
        #
        ###############################################
        ###############################################

        #--------------------------------------------------------------------------#
        # variable / object cleanup
        #--------------------------------------------------------------------------#

        #--------------------------------------------------------------------------#
        # return value
        #--------------------------------------------------------------------------#

        return    

        '''
        '''

        return
    
    #--------------------------------------------------------------------------#
    # supportive methods
    #--------------------------------------------------------------------------#

    def _ctt_calc_cat_values(self, m_series):
        '''
        calculates the categorical values in each series

        Requirements:
        package pandas

        Inputs:
        m_series
        Type: pandas.Series
        Desc: categorical values

        Important Info:
        None

        Return:
        object
        Type: pandas.Series
        Desc: values by category in descending order
        '''
        
        # set-up to count categorical values
        list_y_values = list()
        list_x_values = m_series.unique().tolist()
        
        # loop through x-values
        for string_value in list_x_values:
            array_x_bool = m_series == string_value
            list_y_values.append(array_x_bool.sum())
        
        # create series
        series_values = pandas.Series(data = list_y_values, index = list_x_values)
        
        return series_values.sort_values(ascending = False)
    
    def _ctt_plot_train_test(self, m_plot, m_series_data, m_int_max_cat = 10,
        m_string_data = 'Train'):
        '''
        plot train / test values

        Requirements:
        package matplotlib.pyplot
        package pandas

        Inputs:
        m_plot
        Type: matplotlib.pyplot.axes
        Desc: plot to plot data onto

        m_series_data
        Type: pandas.Series
        Desc: values and categories to plot

        m_int_max_cat
        Type: integer
        Desc: max amount of categories to plot

        Important Info:
        None

        Return:
        object
        Type: matplotlib.pyplot.axes
        Desc: plot of train or test data
        '''
        
        # plot the data
        m_plot.bar(
            x = m_series_data.index.values.tolist()[:m_int_max_cat],
            height  = m_series_data[:m_int_max_cat].values)
        m_plot.tick_params(axis = 'x', rotation = 35)
        m_plot.set_ylabel('Count')
        m_plot.set_title(m_string_data)
        
        return m_plot
    
    def def_Methods(self, list_cluster_results, array_sparse_matrix):
        '''
        below is an example of a good method comment

        ---------------------------------------------------------------------------------------------------------------------------------------------------

        this method implements the evauluation criterea for the clusters of each clutering algorithms
        criterea:
               - 1/2 of the clusters for each result need to be:
                   - the average silhouette score of the cluster needs to be higher then the silhouette score of all the clusters
                     combined
                   - the standard deviation of the clusters need to be lower than the standard deviation of all the clusters
                     combined
               - silhouette value for the dataset must be greater than 0.5

        Requirements:
        package time
        package numpy
        package statistics
        package sklearn.metrics

        Inputs:
        list_cluster_results
        Type: list
        Desc: the list of parameters for the clustering object
        list[x][0] -> type: array; of cluster results by sample in the order of the sample row passed as indicated by the sparse
                         or dense array
        list[x][1] -> type: string; the cluster ID with the parameters

        array_sparse_matrix
        Type: numpy array
        Desc: a sparse matrix of the samples used for clustering

        Important Info:
        None

        Return:
        object
        Type: list
        Desc: this of the clusters that meet the evaluation criterea
        list[x][0] -> type: array; of cluster results by sample in the order of the sample row passed as indicated by the sparse
                        or dense array
        list[x][1] -> type: string; the cluster ID with the parameters
        list[x][2] -> type: float; silhouette average value for the entire set of data
        list[x][3] -> type: array; 1 dimensional array of silhouette values for each data sample
        list[x][4] -> type: list; list of lists, the cluster and the average silhoutte value for each cluster, the orders is sorted 
                            highest to lowest silhoutte value
                            list[x][4][x][0] -> int; cluster label
                            list[x][4][x][1] -> float; cluster silhoutte value
        list[x][5] -> type: list; a list that contains the cluster label and the number of samples in each cluster
                           list[x][5][x][0] -> int; cluster label
                           list[x][5][x][1] -> int; number of samples in cluster list[x][5][x][0]
        '''

        #--------------------------------------------------------------------------------#
        # objects declarations
        #--------------------------------------------------------------------------------#

        #--------------------------------------------------------------------------------#
        # time declarations
        #--------------------------------------------------------------------------------#

        #--------------------------------------------------------------------------------#
        # lists declarations
        #--------------------------------------------------------------------------------#

        #--------------------------------------------------------------------------------#
        # variables declarations
        #--------------------------------------------------------------------------------#

        ###############################################
        ###############################################
        #
        # Start
        #
        ###############################################
        ###############################################

        #--------------------------------------------------------------------------#
        # sub-section comment
        #--------------------------------------------------------------------------#

        ###############################################
        ###############################################
        #
        # sectional comment
        #
        ###############################################
        ###############################################

        #--------------------------------------------------------------------------#
        # variable / object cleanup
        #--------------------------------------------------------------------------#

        #--------------------------------------------------------------------------#
        # return value
        #--------------------------------------------------------------------------#

        return    

    #--------------------------------------------------------------------------#
    # example
    #--------------------------------------------------------------------------#