'''
this script analyses the BikeBuyer data set
'''

'''
package import
'''

from matplotlib import pyplot
from datetime import datetime

# models
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier

# metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

# pre-processing
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectPercentile

# model tuning
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from numpy.random import uniform
from numpy.random import randint

import pandas
import numpy
import os
import seaborn
import pickle

import warnings
warnings.simplefilter('ignore')

'''
classes
'''

class BicycleAnalysis(object):
    '''
    This class analyzes of a Bicycle purchase dataset.

    Requirements:
    package pandas
    package numpy
    package os
    package seaborn
    package pickle
    package matplotlib
    package datetime
    package warnings
    package scikit-learn

    Example:
    ba = BicycleAnalysis()
    ba.main()

    Methods:
    main -> wrapper for the analysis
    load_data -> loads the analysis data
    compare_train_test -> plots train and test data to compare in preperation for the
        analysis
    basic_exploration -> basic metrics of the data sets
    pre_process_data -> processes the data for modeling
    feature_importance -> explores the data for feature engineering
    generic_models -> tests the data on generic models
    model_tuning -> tunes a subset of models
    predict_on_test -> predices the best model from model tuning on the test set

    Attributes:
    $$$ file and path variables $$$

    string_data_path -> directory that olds the data
    string_model_path -> directory that holds the models
    string_plots_path -> directory that holds the plots
    string_file_test -> test data file
    string_file_train -> training data file
    set_source_files -> container of the data files
    string_y_col -> name of the column to predict, in train and test sets

    $$$ data containers $$$

    df_test_raw -> raw test data
    df_train_raw -> raw training data
    df_test_common -> test data with common columns of traning set; without predict column
    df_train_common -> training data with common columsn of test set; without predict
        column
    df_train_ohe -> one-hot-encoded training set; without predict column
    df_test_ohe -> one-hot-endodeded test set; without predict column
    series_train_y -> training data predicted column
    series_test_y -> test data predicted column

    $$$ others $$$

    list_common_cols -> common columns in train and test set
    dict_feat_imp_flags -> feature importance flags which produces plots
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
        self.string_plots_path = os.path.abspath('./plots')
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
        # Start
        #
        ###############################################
        ###############################################

        #--------------------------------------------------------------------------------#
        # load data
        #--------------------------------------------------------------------------------#
        
        print('\nLOADING DATA', '\n')
        self.load_data(m_bool_filter_columns = True)
        
        #--------------------------------------------------------------------------------#
        # below is the exploration of the data
        #--------------------------------------------------------------------------------#
        
        print('DATA EXPLORATION', '\n')
        self.compare_train_test(['categorical_columns'])
        self.compare_train_test(['prediction_column'])
        self.basic_exploration()
        
        #--------------------------------------------------------------------------------#
        # below is the set-up for modeling
        #--------------------------------------------------------------------------------#
        
        print('PRE-PROCESSING DATA', '\n')
        self.df_test_ohe = self.pre_process_data(self.df_test_common, False)
        self.df_train_ohe = self.pre_process_data(self.df_train_common, True)
        
        #--------------------------------------------------------------------------------#
        # feature engineering
        #--------------------------------------------------------------------------------#
        
        print('FEATURE ENGINEERING', '\n')
        tup_dfs = self.feature_importance('all',
            m_df_train = self.df_train_ohe,
            m_series_y = self.series_train_y)
        
        #--------------------------------------------------------------------------------#
        # below is the modeling
        #--------------------------------------------------------------------------------#
        
        print('GENERIC MODEL TESTING', '\n')
        df_gen_models = self.generic_models(self.df_train_ohe)
        print()
        print(df_gen_models, '\n')

        #--------------------------------------------------------------------------------#
        # below is the model tuning for the top two generic models
        #--------------------------------------------------------------------------------#

        print('TUNING TWO MODELS', '\n')
        dict_model_tuning = self.model_tuning()
        for string_clf in dict_model_tuning:
            print('\n' + string_clf)
            print(dict_model_tuning[string_clf]['best_est'])
            print(dict_model_tuning[string_clf]['best_score'], '\n')

        #--------------------------------------------------------------------------------#
        # prediction on test set
        #--------------------------------------------------------------------------------#

        print('PREDICT ON THE TEST SET', '\n')
        series_y_hat = self.predict_on_test('Ridge')
        print('BikeBuyer analysis complete')

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

            # pickle tain & test y-values
            string_y_test = os.path.join(self.string_data_path, 'series_y_test.pckl')
            string_y_train = os.path.join(self.string_data_path, 'series_y_train.pckl')
            pickle.dump(self.series_test_y, open(string_y_test, 'wb'))
            pickle.dump(self.series_train_y, open(string_y_train, 'wb'))

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
                            int_max, 'Train', 90)
                        axes[1] = self._ctt_plot_train_test(axes[1], series_test,
                            int_max, 'Test', 90)

                        # save plot
                        string_plot_name = 'cc_' + string_column + '.png'
                        fig.savefig(os.path.join(
                            self.string_plots_path, string_plot_name))
            
            # predicted column
            if string_analysis == 'prediction_column':
                # get train / test values
                series_bb_train = self._ctt_calc_cat_values(self.series_train_y)

                # plot predicted column
                fig, ax = pyplot.subplots()
                ax = self._ctt_plot_train_test(ax, series_bb_train,
                    2, 'Train', 0)
                fig.suptitle('BikeBuyer Yes (1) or No (0)')
                
                # save plot
                string_plot_name = 'pd_bikebuyer.png'
                fig.savefig(os.path.join(self.string_plots_path, string_plot_name))
    
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
    
    def pre_process_data(self, m_df, m_bool_train):
        '''
        this method pre-processes the data for modeling

        Requirements:
        package pandas

        Inputs:
        m_df
        Type: pandas.DataFrame
        Desc: data for either train or test to process

        m_bool_train
        Type: boolean
        Desc: flag if training or test set

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
        df_prep = self._cast_cols_to_object(df_prep)

        for string_col in self.list_common_cols:
            if df_prep[string_col].dtype == 'object':
                list_ohe_cols.append(string_col)
        
        ohe = OneHotEncoder(sparse = False)
        array_ohe = ohe.fit_transform(df_prep[list_ohe_cols])

        #--------------------------------------------------------------------------#
        # create column names
        #--------------------------------------------------------------------------#

        list_ohe_col_names = list()
        for int_idx, string_col in enumerate(list_ohe_cols):
            array_cats = ohe.categories_[int_idx]
            for int_cat_idx, string_cat in enumerate(array_cats):
                list_ohe_col_names.append(string_col + '_' + string_cat)
        
        #--------------------------------------------------------------------------#
        # create dataframes to concat
        #--------------------------------------------------------------------------#
        
        df_prep = df_prep.drop(labels = list_ohe_cols, axis = 1)
        df_ohe = pandas.DataFrame(array_ohe, columns = list_ohe_col_names)
        df_pp = pandas.concat([df_prep, df_ohe], axis = 1)
        del df_prep, df_ohe

        #--------------------------------------------------------------------------#
        # safe dataframe
        #--------------------------------------------------------------------------#

        if m_bool_train:
            string_file = 'df_ohe_train.pckl'
        else:
            string_file = 'df_ohe_test.pckl'
        string_pp_save = os.path.join(self.string_data_path, string_file)
        pickle.dump(df_pp, open(string_pp_save, 'wb'))

        #--------------------------------------------------------------------------#
        # return value
        #--------------------------------------------------------------------------#

        return df_pp
    
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
        object
        Type: tuple
        Desc: the result of the feature analysis bases on the inputs; each DataFrame
            could be None if not executed
            tuple[0] -> df; anova
            tuple[1] -> df; chi-squared 
            tuple[2] -> df; feature imporance based on random forest
            tuple[3] -> df; correlation heatmap from seaborn
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
            df_anova['cum_perc'].plot.line()
            self._create_line(df_anova[:20], 'cum_perc', 'anova.png', 'anova')
        else:
            df_anova = None
        
        # chi-squared analyis
        if bool_all or self.dict_feat_imp_flags.get('chi', False):
            df_chi2 = self._fi_selector(m_df_train, m_series_y, 'chi')
            self._create_line(df_chi2[:20], 'cum_perc', 'chi2.png', 'chi^2')
        else:
            df_chi2 = None
        
        # feature importance by model
        if bool_all or self.dict_feat_imp_flags.get('feat_imp', False):
            df_fi = self._fi_model(m_df_train, m_series_y, 'feat_imp')
            self._create_line(df_anova[:20], 'cum_perc', 'model_fi.png',
                'model feature importance')
        else:
            df_fi = None

        # correlation matrix
        if bool_all or self.dict_feat_imp_flags.get('heatmap', False):
            df_corr = m_df_train.corr()
            idx_top_corr_feat = df_corr.index
            fig = pyplot.figure()
            heatmap = seaborn.heatmap(
                m_df_train[idx_top_corr_feat].corr(), annot = False, cmap = 'RdYlGn')
            heatmap.set_title('correlation heatmap')
            fig.add_axes(heatmap)
            fig.savefig(os.path.join(self.string_plots_path, 'heatmap.png'))
        else:
            heatmap = None

        return df_anova, df_chi2, df_fi, df_corr
    
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
            dt_start = datetime.now()
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
            td_cv = datetime.now() - dt_start
            
            # dataframe of cv results
            df_cv_results = pandas.DataFrame(
                data = list_cv_results,
                columns = ['precision', 'recall', 'f1', 'accuracy'])

            # average the results ov the cross-validation
            dict_return[string_model] = {
                'precision':df_cv_results['precision'].mean(),
                'recall':df_cv_results['recall'].mean(),
                'f1':df_cv_results['f1'].mean(),
                'accuracy':df_cv_results['accuracy'].mean(),
                'time':td_cv.total_seconds()}
        
        # dataframe of results
        df_results = pandas.DataFrame(dict_return)
        df_results = df_results.transpose()
        df_results = df_results.sort_values(by = 'f1', ascending = False)

        # save dataframe
        string_gen_models = os.path.join(self.string_data_path, 'df_gen_models.pckl')
        pickle.dump(df_results, open(string_gen_models, 'wb'))
        
        return df_results
    
    def model_tuning(self, m_int_iterations = 20, m_int_cv = 5):
        '''
        this method tunes the top 'n' models; right now set up for gradboost and ridge
        classification

        Requirements:
        package pandas
        package sklearn

        Inputs:
        m_int_iterations
        Type: integer
        Desc: the number of times the model will pull paramaters for the random search

        m_int_cv
        Type: integer
        Desc: the number of cross validations for the random search

        Important Info:
        1. in the pramater dictionary the classifer is the address to the classifier
           and not the classifier istself; that is why '()' need to be at the end
           'estimator' section of the random search object

        Return:
        object
        Type: dictionary
        Desc: the best estimator and other varriables of the random search
            key -> string, name of model; value -> dictionary of estimator info
            dict[string_model] = {
                'best_est':best_estimator,
                'best_score':best_score,
                'best_params':best_params,
                'cv_results':cv_results,
                'generic_model':address of generic estimator
            }
        '''
        # list_top_models = ['Ridge', 'GradBoost']
        list_top_models = ['Ridge']

        # make scorer
        dict_scorer = {'f1':make_scorer(f1_score), 'roc_auc':make_scorer(roc_auc_score)}

        # create model tuning diciontary
        dict_model_tuning_params = {
            'Ridge':{
                'clf':RidgeClassifier,
                'params':{
                    'alpha':uniform(low = 0.01, high = 2, size = 10),
                    'fit_intercept':[True, False],
                    'tol':uniform(low = 0.001, high = 0.7, size = 10),
                    'solver':['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
                },
                'scorer':dict_scorer.get('f1', 'f1')
            },
            'GradBoost':{
                'clf':GradientBoostingClassifier,
                'params':{
                    # 'loss':['deviance', 'exponential'],
                    'loss':['exponential'],
                    'learning_rate':uniform(low = 0.01, high = 0.7, size = 10),
                    'n_estimators':randint(low = 20, high = 500, size = 10),
                    'criterion':['friedman_mse', 'mse', 'mae'],
                    'min_samples_split':[x for x in range(2, 5)],
                    'min_samples_leaf':[x for x in range(1, 4)],
                    'max_depth':[2, 3, 4, 5]
                },
                'scorer':dict_scorer.get('f1', 'f1')
            }
        }

        # load train data
        print('loading traning data')
        string_path_x = os.path.join(self.string_data_path, 'df_ohe_train.pckl')
        string_path_y = os.path.join(self.string_data_path, 'series_y_train.pckl')
        df_x_train = pickle.load(open(string_path_x, 'rb'))
        series_y_train = pickle.load(open(string_path_y, 'rb'))

        # loop through models to tune
        dict_best_tuning = dict()
        for string_model in list_top_models:
            # get params
            dict_model_params = dict_model_tuning_params.get(string_model, dict())

            # conduct random search
            random_search = RandomizedSearchCV(
                estimator = dict_model_params.get('clf')(),
                param_distributions = dict_model_params.get('params', dict()),
                scoring = dict_model_params.get('scorer', 'f1'),
                n_iter = m_int_iterations,
                cv = m_int_cv,
                return_train_score = True
            )

            # fit model and start timer
            print('randomly searching variables for {}'.format(string_model))
            dt_start = datetime.now()
            random_search.fit(df_x_train.values, series_y_train.values)
            td_rs = datetime.now() - dt_start

            # get best results
            best_estimator = random_search.best_estimator_
            best_score = random_search.best_score_
            best_params = random_search.best_params_
            cv_results = random_search.cv_results_

            # add to return dictionary
            dict_best_tuning[string_model] = {
                'best_est':best_estimator,
                'best_score':best_score,
                'best_params':best_params,
                'cv_results':cv_results,
                'generic_model':dict_model_params.get('clf'),
                'time':td_rs.total_seconds()
            }

        # pickle model dictionary
        string_bt = os.path.join(self.string_data_path, 'dict_best_estimator.pckl')
        pickle.dump(dict_best_tuning, open(string_bt, 'wb'))
        
        return dict_best_tuning

    def predict_on_test(self, m_string_classifier):
        '''
        this method predicts on the test set from the best estimator from the dictionary
        saved as a pickle file

        Requirements:
        package pandas
        package sklearn

        Inputs:
        m_string_classifier
        Type: string
        Desc: key to pull the classifier from the best estimator dictionary

        Important Info:
        None

        Return:
        object
        Type: pandas.Series
        Desc: predicted value for the bicycle buyer
        '''
        # load test data
        string_test_x = os.path.join(self.string_data_path, 'df_ohe_test.pckl')
        string_train_x = os.path.join(self.string_data_path, 'df_ohe_train.pckl')
        df_test_x = pickle.load(open(string_test_x, 'rb'))
        df_train_x = pickle.load(open(string_train_x, 'rb'))

        # ensure same columns training and test sets
        df_test_x = self._pot_same_columns(df_test_x, df_train_x.columns.values.tolist(),
            df_train_x.dtypes.values.tolist())

        # load classifier
        string_tm = os.path.join(self.string_data_path, 'dict_best_estimator.pckl')
        dict_tuned_models = pickle.load(open(string_tm, 'rb'))
        if isinstance(dict_tuned_models, dict) and m_string_classifier in dict_tuned_models.keys():            
            best_clf = dict_tuned_models[m_string_classifier]['best_est']
        else:
            best_clf = None
        
        # predict on test set
        if best_clf is None:
            raise ValueError(
                '{} is not in the tuned models dictionary'.format(m_string_classifier))
        else:
            array_y_hat = best_clf.predict(df_test_x.values)
            series_y_hat = pandas.Series(data = array_y_hat, name = 'BicyleBuyer')
            del array_y_hat
            string_y_hat_dump = os.path.join(self.string_data_path, 'series_y_test.pckl')
            pickle.dump(series_y_hat, open(string_y_hat_dump, 'wb'))
        
        # create plot of prediction
        fig, ax = pyplot.subplots()
        string_plot_title = 'BicycleBuyer Predicted'
        series_yhat_counts = self._ctt_calc_cat_values(series_y_hat)
        ax = self._ctt_plot_train_test(ax, series_yhat_counts,
            m_string_data = string_plot_title, m_int_x_tick_rotation = 0)
        string_save_yhat_plot = os.path.join(self.string_plots_path, 'bb_predicted.png')
        fig.savefig(string_save_yhat_plot)
        
        return series_y_hat

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
        m_string_data = 'Train', m_int_x_tick_rotation = 90):
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

        m_int_x_tick_rotation
        Type: integer
        Desc: number of degrees to rotate the xtick label

        Important Info:
        None

        Return:
        object
        Type: matplotlib.pyplot.axes
        Desc: plot of train or test data
        '''
        
        # plot the data
        series_plot = m_series_data[:m_int_max_cat]
        m_plot.bar(
            x = series_plot.index.values.tolist(),
            height  = series_plot.values)
        m_plot.set_xticks([x for x in range(0, len(series_plot))])
        m_plot.set_xticklabels(series_plot.index.values.tolist(), 
            rotation = m_int_x_tick_rotation)
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
        dict_data = {
            tup_sc[0]:fi_selector.scores_,
            'feature':m_fi_x.columns.values}
        df_fa = pandas.DataFrame(data = dict_data)
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

    def _create_line(self, m_df, m_string_col, m_string_plot_name, m_string_title):
        '''
        '''

        fig, ax = pyplot.subplots()
        ax.plot(m_df[m_string_col].values)
        ax.set_xlabel('feature')
        ax.set_ylabel(m_string_col)
        ax.set_xticks([x for x in range(0, len(m_df['feature']))])
        ax.set_xticklabels(m_df['feature'].values, rotation = 90)
        ax.set_title(m_string_title)
        string_fig_path = os.path.join(self.string_plots_path, m_string_plot_name)
        fig.savefig(string_fig_path)

        return

    def _pot_same_columns(self, m_df_test, m_list_train_cols, m_list_train_dtypes):
        '''
        ensures the same columns are in the train and test sets

        Requirements:
        package pandas
        package numpy

        Inputs:
        m_df_test
        Type: pandas.DataFrame
        Desc: dataframe of test data

        m_list_train_cols
        Type: list
        Desc: training data set columns

        m_list_train_dtypes
        Type: list
        Desc: data types of training dataset

        Important Info:
        None

        Return:
        object
        Type: pandas.DataFrame
        Desc: test data with the same columns at the training data set
        '''
        # get columns needed and columns to drop
        df_test = m_df_test.copy()
        set_train_cols = set(m_list_train_cols)
        set_test_cols = set(df_test.columns.values.tolist())
        set_cols_needed = set_train_cols - set_test_cols
        set_cols_to_drop = set_test_cols - set_train_cols

        # add columns
        dict_data_to_add = dict()
        for string_col_to_add in set_cols_needed:
            # get index and dtype from lists
            int_index = m_list_train_cols.index(string_col_to_add)
            dtype = m_list_train_dtypes[int_index]

            # generate fill value
            if dtype == numpy.dtype('int64'):
                fill_value = 0
            else:
                fill_value = 0.
            
            # create series & add to data dictionary
            series_fill = pandas.Series([fill_value for x in range(0, len(df_test))])
            dict_data_to_add[string_col_to_add] = series_fill

        # create dataframe
        df_to_add = pandas.DataFrame(data = dict_data_to_add)

        # cols to drop
        if len(set_cols_to_drop) > 0:
            list_cols_to_drop = list(set_cols_to_drop)
            df_test = df_test.drop(list_cols_to_drop, axis = 1)
        
        # return dataframe
        df_return = pandas.concat([df_test, df_to_add], axis = 1)

        return df_return[m_list_train_cols]
