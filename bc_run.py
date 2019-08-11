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

if __name__ == '__main__':
    '''
    load data
    '''
    
    print('LOADING DATA', '\n')
    bicycle_analysis = BicycleAnalysis()
    bicycle_analysis.load_data(m_bool_filter_columns = True)
    
    '''
    below is the exploration of the data
    '''
    
    print('DATA EXPLORATION', '\n')
    bicycle_analysis.compare_train_test(['categorical_columns'])
    bicycle_analysis.compare_train_test(['prediction_column'])
    bicycle_analysis.basic_exploration()
    
    '''
    below is the set-up for modeling
    '''
    
    print('PRE-PROCESSING DATA', '\n')
    bicycle_analysis.df_test_ohe = bicycle_analysis.pre_process_data(
        bicycle_analysis.df_test_common)
    bicycle_analysis.df_train_ohe = bicycle_analysis.pre_process_data(
        bicycle_analysis.df_train_common)
    
    '''
    feature engineering
    '''
    
    print('FEATURE ENGINEERING', '\n')
    var_obj = bicycle_analysis.feature_importance(
        'heatmap',
        m_df_train = bicycle_analysis.df_train_ohe,
        m_series_y = bicycle_analysis.series_train_y)
    
    '''
    below is the modeling
    '''
    
    print('GENERIC MODEL TESTING', '\n')
    df_gen_models = bicycle_analysis.generic_models(bicycle_analysis.df_train_ohe)
    print()
    print(df_gen_models)