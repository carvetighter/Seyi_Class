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
import pandas
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

    def explore(self, m_bool_train = True, m_string_flag = 'basic'):
        '''
        this method explores a pandas dataframe

        Requirements:
        package pandas
        package os

        Inputs:
        m_bool_train
        Type: boolean
        Desc: determine which set to load and explore

        m_string_flag
        Type: string
        Desc: flag for the type of analysis to do

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

        if m_bool_train:
            df_data = self.df_train_raw
            string_test_train = 'training'
        else:
            df_data = self.df_test_raw
            string_test_train = 'test'

        #--------------------------------------------------------------------------------#
        # time declarations
        #--------------------------------------------------------------------------------#

        #--------------------------------------------------------------------------------#
        # lists declarations
        #--------------------------------------------------------------------------------#

        #--------------------------------------------------------------------------------#
        # variables declarations
        #--------------------------------------------------------------------------------#

        string_status_templ = 'Exploring {0} data set.\n'

        ###############################################
        ###############################################
        #
        # start the data exploration
        #
        ###############################################
        ###############################################

        print(string_status_templ.format(string_test_train))

        if m_string_flag == 'basic':
            print(df_data.info(verbose = True))
            print(df_data.head(3))

        #--------------------------------------------------------------------------#
        # loop through the columns and plot
        #--------------------------------------------------------------------------#

        if m_string_flag == 'categorical_columns':
            for string_column in df_data.columns:
                if df_data[string_column].dtype == 'object':
                    list_y_values = list()
                    list_x_values = list(df_data[string_column].unique())
                    for string_value in list_x_values:
                        array_x_bool = df_data[string_column] == string_value
                        list_y_values.append(array_x_bool.sum())
                    series_values = pandas.Series(data = list_y_values, index = list_x_values)
                    series_values = series_values.sort_values(ascending = False)
                    int_max = 10
                    fig, ax = pyplot.subplots()
                    ax.bar(
                        x = series_values.index.values.tolist()[:int_max],
                        height  = series_values[:int_max].values)
                    ax.tick_params(axis = 'x', rotation = 35)
                    ax.set_ylabel('Count')
                    ax.set_title(string_column)
                    pyplot.show()
        
        #--------------------------------------------------------------------------#
        # look at the distribution of the prediction column
        #--------------------------------------------------------------------------#

        if m_string_flag == 'prediction_column':
            list_y_values = list()
            list_x_values = list(df_data['BikeBuyer'].unique())
            for string_value in list_x_values:
                array_x_bool = df_data['BikeBuyer'] == string_value
                list_y_values.append(array_x_bool.sum())
            series_values = pandas.Series(data = list_y_values, index = list_x_values)
            series_values = series_values.sort_values(ascending = False)
            fig, ax = pyplot.subplots()
            ax.bar(
                x = series_values.index.values.tolist(),
                height  = series_values.values)
            ax.set_ylabel('Count')
            ax.set_xticks([0, 1])
            ax.set_title('BikeBuyer Yes (1) or No (0)')
            pyplot.show()

        #--------------------------------------------------------------------------#
        # return value
        #--------------------------------------------------------------------------#

        return 

    
    #--------------------------------------------------------------------------#
    # supportive methods
    #--------------------------------------------------------------------------#

    # def def_Methods(self, list_cluster_results, array_sparse_matrix):
    #     '''
    #     below is an example of a good method comment

    #     ---------------------------------------------------------------------------------------------------------------------------------------------------

    #     this method implements the evauluation criterea for the clusters of each clutering algorithms
    #     criterea:
    #            - 1/2 of the clusters for each result need to be:
    #                - the average silhouette score of the cluster needs to be higher then the silhouette score of all the clusters
    #                  combined
    #                - the standard deviation of the clusters need to be lower than the standard deviation of all the clusters
    #                  combined
    #            - silhouette value for the dataset must be greater than 0.5

    #     Requirements:
    #     package time
    #     package numpy
    #     package statistics
    #     package sklearn.metrics

    #     Inputs:
    #     list_cluster_results
    #     Type: list
    #     Desc: the list of parameters for the clustering object
    #     list[x][0] -> type: array; of cluster results by sample in the order of the sample row passed as indicated by the sparse
    #                      or dense array
    #     list[x][1] -> type: string; the cluster ID with the parameters

    #     array_sparse_matrix
    #     Type: numpy array
    #     Desc: a sparse matrix of the samples used for clustering

    #     Important Info:
    #     None

    #     Return:
    #     object
    #     Type: list
    #     Desc: this of the clusters that meet the evaluation criterea
    #     list[x][0] -> type: array; of cluster results by sample in the order of the sample row passed as indicated by the sparse
    #                     or dense array
    #     list[x][1] -> type: string; the cluster ID with the parameters
    #     list[x][2] -> type: float; silhouette average value for the entire set of data
    #     list[x][3] -> type: array; 1 dimensional array of silhouette values for each data sample
    #     list[x][4] -> type: list; list of lists, the cluster and the average silhoutte value for each cluster, the orders is sorted 
    #                         highest to lowest silhoutte value
    #                         list[x][4][x][0] -> int; cluster label
    #                         list[x][4][x][1] -> float; cluster silhoutte value
    #     list[x][5] -> type: list; a list that contains the cluster label and the number of samples in each cluster
    #                        list[x][5][x][0] -> int; cluster label
    #                        list[x][5][x][1] -> int; number of samples in cluster list[x][5][x][0]
    #     '''

    #     #--------------------------------------------------------------------------------#
    #     # objects declarations
    #     #--------------------------------------------------------------------------------#

    #     #--------------------------------------------------------------------------------#
    #     # time declarations
    #     #--------------------------------------------------------------------------------#

    #     #--------------------------------------------------------------------------------#
    #     # lists declarations
    #     #--------------------------------------------------------------------------------#

    #     #--------------------------------------------------------------------------------#
    #     # variables declarations
    #     #--------------------------------------------------------------------------------#

    #     ###############################################
    #     ###############################################
    #     #
    #     # Start
    #     #
    #     ###############################################
    #     ###############################################

    #     #--------------------------------------------------------------------------#
    #     # sub-section comment
    #     #--------------------------------------------------------------------------#

    #     ###############################################
    #     ###############################################
    #     #
    #     # sectional comment
    #     #
    #     ###############################################
    #     ###############################################

    #     #--------------------------------------------------------------------------#
    #     # variable / object cleanup
    #     #--------------------------------------------------------------------------#

    #     #--------------------------------------------------------------------------#
    #     # return value
    #     #--------------------------------------------------------------------------#

    #     return    

    #--------------------------------------------------------------------------#
    # example
    #--------------------------------------------------------------------------#