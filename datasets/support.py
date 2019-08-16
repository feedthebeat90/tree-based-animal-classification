#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 12:25:40 2019

@author: lnisenbaum
"""

import pandas as pd


def get_results(y_test, y_pred, df, class_dict):
       
    # Organize our true labels and predictions in a DataFrame
    results = pd.DataFrame({'label': y_test, 'prediction': y_pred})

    # Collect animal_name and add them to results
    y_names = df.iloc[results.index]['animal_name']
    results.insert(0, 'animal_name', y_names)

    # Map labels and predictions to a readable format
    results['true_class'] = results.label.map(class_dict)
    results['predicted_class'] = results.prediction.map(class_dict)
    
    return results

