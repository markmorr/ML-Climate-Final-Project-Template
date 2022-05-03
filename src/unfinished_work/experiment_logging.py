# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 16:56:45 2022

@author: 16028
"""


def runTest():
    """
    Still need to implement properly

    Returns
    -------
    None.

    """

def input_specs():
        
    in_specs = dict()
    in_specs['model_type'] = 'random_forest
    in_specs['hyperparameters'] = 'max_depth=35'
    in_specs['features'] = 'full'
    in_specs['normalization'] = 'none'
    in_specs['imputation'] = 'none'
    
    
        
    trained_model, specs = runTest(network, X_train, y_train, X_test, y_test, in_specs)
        
     
    # new_df = pd.read_csv(r'experiment_results.csv')
    # # specs = [ 1e-4', 'blue and green only', 'none', 'none' ]
    # res = {category_list[i]: specs[i] for i in range(len(category_list))}
    # res = {category_list[i]: specs[i] for i in range(len(category_list))}
    #  #okay so weight decacy 1e-4 is pretty good
    # new_df = new_df.append(res, ignore_index=True)
    # new_df.to_csv(r'experiment_results.csv', index=False)