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
    bicycle_analysis = BicycleAnalysis()
    bicycle_analysis.load_data(m_bool_filter_columns = True)
    
    '''
    below is the exploration of the data
    '''

    # bicycle_analysis.compare_train_test(['categorical_columns'])
    # bicycle_analysis.compare_train_test(['prediction_column'])
    # bicycle_analysis.basic_exploration()

    '''
    below is the set-up for modeling
    '''
    
    # df_test_ohe = bicycle_analysis._cast_cols_to_object(bicycle_analysis.df_test_raw)
    # df_train_ohe = bicycle_analysis._cast_cols_to_object(bicycle_analysis.df_train_common)
    
    # df_test_ohe = bicycle_analysis.pre_process_data(df_test_ohe)
    # df_train_ohe = bicycle_analysis.pre_process_data(df_train_ohe)
    
    var_obj = bicycle_analysis.feature_importance(
        'chi', 'anova', 'all',
        m_df_train = bicycle_analysis.df_train_common,
        m_series_y = bicycle_analysis.series_train_y)
    
    '''
    below is the modeling
    '''

    # print(bicycle_analysis.generic_models(df_train_ohe))