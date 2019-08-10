'''
'''

###############################################
###############################################
#
# File / Package Import
#
###############################################
###############################################

from matplotlib import pyplot
from collections import Counter
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectPercentile

import pandas
import numpy
import os
import seaborn

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
        self.string_y_col = 'BikeBuyer'

        #--------------------------------------------------------------------------#
        # data containers
        #--------------------------------------------------------------------------#

        self.df_test_raw = None
        self.df_train_raw = None
        self.df_test_common = None
        self.df_train_common = None
        self.df_train_ohe = None
        self.df_test_ohe = None
        self.series_train_y = None
        self.series_test_y = None

        #--------------------------------------------------------------------------#
        # others
        #--------------------------------------------------------------------------#

        self.list_common_cols = None
        self.dict_feat_imp_flags = {
            'all':False,
            'anova':False,
            'chi':False,
            'heatmap':False,
            'feat_imp':False}

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
            self.series_test_y = self.df_test_raw[self.string_y_col]
            self.series_train_y = self.df_train_raw[self.string_y_col]
            self.df_test_raw = self.df_test_raw.drop(self.string_y_col, axis = 1)
            self.df_train_raw = self.df_train_raw.drop(self.string_y_col, axis = 1)

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
            self.list_common_cols = list(set_train_cols & set_test_cols)
            
            # filter dataframes
            self.df_test_common = self.df_test_raw[self.list_common_cols]
            self.df_train_common = self.df_train_raw[self.list_common_cols]
        
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
                for string_column in self.list_common_cols:
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
                series_bb_train = self._ctt_calc_cat_values(self.series_train_y)

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

    def basic_exploration(self, m_bool_train = True):
        '''
        this method compares the common columns of the train and test sets

        Requirements:
        package pandas
        package matplotlib.pyplot

        Inputs:
        m_bool_train
        Type: boolean
        Desc: flag to flip between training and test sets

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

        if m_bool_train:
            df_expl = self.df_train_common
            string_set = 'train'
        else:
            df_expl = self.df_test_common
            string_set = 'test'

        #--------------------------------------------------------------------------------#
        # time declarations
        #--------------------------------------------------------------------------------#

        #--------------------------------------------------------------------------------#
        # lists declarations
        #--------------------------------------------------------------------------------#
        
        set_npdt_num = {numpy.dtype('int16'), numpy.dtype('int32'), numpy.dtype('int64'), 
            numpy.dtype('float16'), numpy.dtype('float32'), numpy.dtype('float64')}

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

        print('{} dataset info'.format(string_set))
        df_expl.info()

        #--------------------------------------------------------------------------#
        # first three records of train and test set
        #--------------------------------------------------------------------------#

        print('\nFirst three records of {} data set'.format(string_set))
        print(df_expl.head(3), '\n')

        #--------------------------------------------------------------------------#
        # loop through columns for column info
        #--------------------------------------------------------------------------#

        for string_col in self.list_common_cols:
            np_dtype = df_expl[string_col].dtype
            int_nulls = df_expl[string_col].isnull().sum()
            if np_dtype == numpy.dtype(object):
                var_vals = df_expl[string_col].unique().tolist()
                string_vals = 'unique values are'

            print('# of nulls in column:', int_nulls)
            print(df_expl[string_col].describe())
            if np_dtype == numpy.dtype(object):
                print(string_vals, var_vals, '\n')
            else:
                print('\n')


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
        # one-hot encode categorical columns
        #--------------------------------------------------------------------------#

        list_ohe_cols = list()
        df_prep = m_df.copy()
        for string_col in self.list_common_cols:
            if df_prep[string_col].dtype == 'object':
                list_ohe_cols.append(string_col)
        
        ohe = OneHotEncoder(sparse = False)
        array_ohe = ohe.fit_transform(df_prep[list_ohe_cols])

        #--------------------------------------------------------------------------#
        # create column names
        #--------------------------------------------------------------------------#

        list_ohe_col_names = list()
        for int_idx in range(0, len(list_ohe_cols)):
            string_col = list_ohe_cols[int_idx]
            array_cats = ohe.categories_[int_idx]
            for int_cat_idx in range(0, len(array_cats)):
                list_ohe_col_names.append(string_col + '_' + array_cats[int_cat_idx])
        
        #--------------------------------------------------------------------------#
        # create dataframes to concat
        #--------------------------------------------------------------------------#
        
        df_prep = df_prep.drop(labels = list_ohe_cols, axis = 1)
        df_ohe = pandas.DataFrame(array_ohe, columns = list_ohe_col_names)

        #--------------------------------------------------------------------------#
        # return value
        #--------------------------------------------------------------------------#

        return pandas.concat([df_prep, df_ohe], axis = 1)
    
    def feature_importance(self, *args, m_df_train, m_series_y):
        '''
        calculates the feature importance using several different methods

        Requirements:
        package pandas

        Inputs:
        m_df_train
        Type: pandas.DataFrame
        Desc: data to use to calculate feature importance

        m_series_y
        Type: pandas.Series
        Desc: contains the class of the prediction column

        *args
        Type: list
        Desc: flags passed as strings and they could be one or all of the following
            'all' -> do all the exploration
            'anova' -> analysis of variance
            'chi' -> chi squared analysis
            'heatmap' -> head map of correlation
            'feat_imp' -> feature importance of a model; in this case random forest

        Important Info:
        1. must have a classifaction / result to compare to

        Return:
        ??
        Type: ??
        Desc: ??
        '''

        # set the arguements
        for string_arg in args:
            if string_arg in self.dict_feat_imp_flags.keys():
                self.dict_feat_imp_flags[string_arg] = True
        
        # test for all analysees
        bool_all = False
        if self.dict_feat_imp_flags.get('all', False):
            bool_all = True
        
        # anova analysis
        if bool_all or self.dict_feat_imp_flags.get('anova', False):
            df_anova = self._fi_selector(m_df_train, m_series_y, 'anova')
        else:
            df_anova = None
        
        # chi-squared analyis
        if bool_all or self.dict_feat_imp_flags.get('chi', False):
            df_chi2 = self._fi_selector(m_df_train, m_series_y, 'chi')
        else:
            df_chi2 = None
        
        # feature importance by model
        if bool_all or self.dict_feat_imp_flags.get('feat_imp', False):
            df_fi = self._fi_model(m_df_train, m_series_y, 'feat_imp')
        else:
            pass
        
        # debug code
        print(df_anova[:20], '\n')
        print(df_chi2[:20], '\n')
        print(df_fi[:20])
        anova_ax = df_anova['cum_perc'].plot.line()

        return
    
    def generic_models(self, m_df_train, m_dict_models = None):
        '''
        this method tests generic models based on the f1 score will a prioritized list of
        models by f1 score

        Requirements:
        package pandas
        package sklearn

        Inputs:
        m_df_train
        Type: pandas.DataFrame
        Desc: train data

        Important Info:
        None

        Return:
        object
        Type: dictionary
        Desc: the average of the cross-validation results
            dict['model_name'] = {
                'precision'; float
                'recall'; float
                'f1'; float
                'support'; float
                'accuracy'; float
            }
        '''

        # model dictionary set-up
        if m_dict_models:
            dict_models = m_dict_models
        else:
            dict_models = {
                'AdaBoost':AdaBoostClassifier(),
                'GradBoost':GradientBoostingClassifier(),
                'ExtraTrees':ExtraTreesClassifier(),
                'RandForest':RandomForestClassifier(),
                'LogReg':LogisticRegression(),
                'PassAgg':PassiveAggressiveClassifier(),
                'Perc':Perceptron(),
                'Ridge':RidgeClassifier(),
                'SGD':SGDClassifier(),
                'BernNB':BernoulliNB(),
                'GaussNB':GaussianNB(),
                'KnnC':KNeighborsClassifier(),
                'LinSvc':LinearSVC(),
                'NuSvc':NuSVC(),
                'Svc':SVC(),
                'DecTree':DecisionTreeClassifier(),
                'ExtraTC':ExtraTreeClassifier()
            }
        
        # loop through generic models
        dict_return = dict()
        for string_model, model_classifier in dict_models.items():
            print('starting %s' %string_model)

            # cross-validation
            cv_sss = StratifiedShuffleSplit(n_splits = 5, test_size=0.15)
            
            # loop through train, test splits to fit and predict
            list_cv_results = list()
            for array_train_idx, array_test_idx in cv_sss.split(
                m_df_train, self.series_train_y):
                model_classifier.fit(
                    m_df_train.loc[array_train_idx], self.series_train_y.loc[array_train_idx])
                array_y_pred = model_classifier.predict(m_df_train.loc[array_test_idx])
                tup_prfs = precision_recall_fscore_support(
                    y_true = self.series_train_y.loc[array_test_idx].values,
                    y_pred = array_y_pred,
                    average = 'weighted')
                float_acc = accuracy_score(
                    y_true = self.series_train_y.loc[array_test_idx].values,
                    y_pred = array_y_pred)
                list_record = list(tup_prfs[:-1])
                list_record.append(float_acc)
                list_cv_results.append(list_record)
            
            # dataframe of cv results
            df_cv_results = pandas.DataFrame(
                data = list_cv_results,
                columns = ['precision', 'recall', 'f1', 'accuracy'])

            # average the results ov the cross-validation
            dict_return[string_model] = {
                'precision':df_cv_results['precision'].mean(),
                'recall':df_cv_results['recall'].mean(),
                'f1':df_cv_results['f1'].mean(),
                'accuracy':df_cv_results['accuracy'].mean()}
        
        # dataframe of results
        df_results = pandas.DataFrame(dict_return)
        df_results = df_results.transpose()
        
        return df_results.sort_values(by = 'f1', ascending = False)
    
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
        m_plot.tick_params(axis = 'x', rotation = 90)
        if m_string_data == 'Test':
            m_plot.tick_params(axis = 'y', right = True, left = False, 
                labelright = True, labelleft = False, direction = 'out')
            m_plot.yaxis.set_label_position('right')
        m_plot.set_ylabel('Count')
        m_plot.set_title(m_string_data)
        
        return m_plot
    
    def _cast_cols_to_object(self, m_df):
        '''
        this method automates the casting of some columns in the training and test sets
        that is done in the notebook example

        Requirements:
        package pandas

        Inputs:
        m_df
        Type: pandas.DataFrame
        Desc: train or test data

        Important Info:
        1. assumption is columns are in DataFrame

        Return:
        None
        Type: n/a
        Desc: n/a
        '''
        
        list_cols_to_cast = ['NumberChildrenAtHome', 'NumberCarsOwned', 'TotalChildren']
        m_df[list_cols_to_cast] = m_df[list_cols_to_cast].astype(str)
        
        return m_df
    
    def _fi_selector(self, m_fi_x, m_fi_y, m_string_func):
        '''
        generates feature analysis dataframe

        Requirements:
        package pandas
        package sklearn.feature_selection

        Inputs:
        m_fi_x
        Type: pandas.DataFrame
        Desc: data to fit selector on

        m_fi_y
        Type: pandas.Series
        Desc: y-values to fit selector on

        m_string_func
        Type: string
        Desc: string to indicate the function

        Important Info:
        None

        Return:
        object
        Type: pandas.DataFrame
        Desc: feature analysis dataframe
            columns:
            <string_evaluation> -> the score
            'feature' -> feature for the score
            'cum_sum' -> cummlative sum
            'cum_perc' -> cummlative percentage

        '''
        
        # set-up for feature selection
        dict_sort_col = {
            'chi':('chi2', chi2),
            'anova':('f_value', f_classif)}
        tup_sc = dict_sort_col.get(m_string_func, 'anova')

        # feature selection
        fi_selector = SelectPercentile(tup_sc[1], percentile = 100)
        fi_selector.fit(m_fi_x, m_fi_y)

        # create DataFrame
        dict_chi2_data = {
            tup_sc[0]:fi_selector.scores_,
            'feature':m_fi_x.columns.values}
        df_fa = pandas.DataFrame(data = dict_chi2_data)
        df_fa = df_fa.sort_values(by = tup_sc[0], ascending = False)

        # add additional information
        float_max = df_fa[tup_sc[0]].sum()
        series_cum_sum = df_fa[tup_sc[0]].cumsum()
        series_perc = (series_cum_sum / float_max) * 100
        dict_new_cols = {'cum_sum':series_cum_sum, 'cum_perc':series_perc}
        df_fa = df_fa.assign(**dict_new_cols)
        
        return df_fa
    
    def _fi_model(self, m_fi_x, m_fi_y, m_string_func):
        '''
        generates feature analysis dataframe

        Requirements:
        package pandas
        package sklearn.feature_selection

        Inputs:
        m_fi_x
        Type: pandas.DataFrame
        Desc: data to fit selector on

        m_fi_y
        Type: pandas.Series
        Desc: y-values to fit selector on

        m_string_func
        Type: string
        Desc: string to indicate the function

        Important Info:
        None

        Return:
        object
        Type: pandas.DataFrame
        Desc: feature analysis dataframe
            columns:
            'feat_imp' -> the score
            'feature' -> feature for the score
            'cum_sum' -> cummlative sum
            'cum_perc' -> cummlative percentage

        '''
        
        # model for feature importances
        rf_model = RandomForestClassifier()
        rf_model.fit(m_fi_x, m_fi_y)

        # create values for series
        series_fi = pandas.Series(rf_model.feature_importances_)
        df_fi = pandas.DataFrame(
            {
                m_string_func:series_fi.values,
                'feature':m_fi_x.columns.values.tolist()
            }
        )
        df_fi = df_fi.sort_values(by = m_string_func, ascending = False)

        # create additional columns
        series_cum_sum = df_fi[m_string_func].cumsum()
        float_max = series_cum_sum.max()
        series_cum_perc = (series_cum_sum / float_max) * 100
        dict_data = {'cum_sum':series_cum_sum.values, 'cum_perc':series_cum_perc.values}

        return df_fi.assign(**dict_data)

    #--------------------------------------------------------------------------#
    # example
    #--------------------------------------------------------------------------#

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
