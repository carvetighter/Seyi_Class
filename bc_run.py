'''
'''

###############################################
###############################################
#
# File / Package Import
#
###############################################
###############################################

from BicycleClass import BicycleAnalysis
import os
import 
import pandas

###############################################
###############################################
#
# main method
#
###############################################
###############################################

def main():
    '''
    load data
    '''
    
    bicycle_analysis = BicycleAnalysis()
    # print('\nLOADING DATA', '\n')
    # bicycle_analysis.load_data(m_bool_filter_columns = True)
    
    '''
    below is the exploration of the data
    '''
    
    # print('DATA EXPLORATION', '\n')
    # bicycle_analysis.compare_train_test(['categorical_columns'])
    # bicycle_analysis.compare_train_test(['prediction_column'])
    # bicycle_analysis.basic_exploration()
    
    '''
    below is the set-up for modeling
    '''
    
    # print('PRE-PROCESSING DATA', '\n')
    # bicycle_analysis.df_test_ohe = bicycle_analysis.pre_process_data(
    #     bicycle_analysis.df_test_common, False)
    # bicycle_analysis.df_train_ohe = bicycle_analysis.pre_process_data(
    #     bicycle_analysis.df_train_common, True)
    
    '''
    feature engineering
    '''
    
    # print('FEATURE ENGINEERING', '\n')
    # var_obj = bicycle_analysis.feature_importance(
    #     'all',
    #     m_df_train = bicycle_analysis.df_train_ohe,
    #     m_series_y = bicycle_analysis.series_train_y)
    
    '''
    below is the modeling
    '''
    
    # print('GENERIC MODEL TESTING', '\n')
    # df_gen_models = bicycle_analysis.generic_models(bicycle_analysis.df_train_ohe)
    # print()
    # print(df_gen_models)

    '''
    below is the model tuning for the top two generic models
    '''

    print('TUNING TWO MODELS', '\n')
    dict_model_tuning = bicycle_analysis.model_tuning()
    for string_clf in dict_model_tuning:
        print(string_clf)
        print(dict_model_tuning[string_clf]['best_estimator'])
        print(dict_model_tuning[string_clf]['best_score'], '\n')
    
    '''
    prediction on test set
    '''

    print('PREDICT ON THE TEST SET', '\n')
    series_y_hat = bicycle_analysis.predict_on_test('Ridge')


if __name__ == '__main__':
    main()
    