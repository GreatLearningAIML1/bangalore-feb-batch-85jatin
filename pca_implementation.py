# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 01:45:27 2019

@author: Jatin_Thakkar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
import os
import sys
import datetime


def pca_implementation():
    """
    ---------------------------------------------------------------------------------------
    ================================ IDENTIFY AND TRAIN PCA ===============================
    ---------------------------------------------------------------------------------------
    DATAFRAME TYPE: Preprocessed, structured data
    ---------------------------------------------------------------------------------------
    FUNCTION: PCA implementation, Correlation metrics, Dataframe modifications
    ---------------------------------------------------------------------------------------
    INPUTS REQUIRED: Information in pandas dataframe
    ---------------------------------------------------------------------------------------
    """
    print(pca_implementation.__doc__)
    
    print("PFB is the list of databases to choose from...")
    # This is the path where you want to search
    path = r'C:/Users/Jatin_Thakkar/Downloads/Reference codes/Lab/Featurization_Model_selection_and_Tuning_R5_Project'
    # this is the extension you want to detect
    extension = '.csv'
    for root, dirs_list, files_list in os.walk(path):
        for file_name in files_list:
            if os.path.splitext(file_name)[-1] == extension:
                file_name_path = os.path.join(root, file_name)
                print(file_name)
    #            print(file_name_path)   # This is the full path of the filter file
    print("")
    print("---------------------------------------------------------------------------------------")
    identified_db = input("Please select a dataframe for exploratory data analysis: ")
    print("---------------------------------------------------------------------------------------")
    db = identified_db + ".csv"
    df = pd.read_csv(db)
    print(df.info())
    
    target = input("Identify target column name (NA if unsupervised): ")
    
    if target == "NA":
        df_pca_f = df.copy()
    else:
        df_pca_f = df.drop(target,1).copy()
    shape = df_pca_f.shape
    max_columns = shape[1]
    
    def get_redundant_pairs(df_pca_f):
        '''Get diagonal and lower triangular pairs of correlation matrix'''
        pairs_to_drop = set()
        cols = df_pca_f.columns
        for i in range(0, df_pca_f.shape[1]):
            for j in range(0, i+1):
                pairs_to_drop.add((cols[i], cols[j]))
        return pairs_to_drop

    def get_top_abs_correlations(df_pca_f):
        au_corr = df_pca_f.corr().abs().unstack()
        labels_to_drop = get_redundant_pairs(df_pca_f)
        a = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
        au_corr = pd.DataFrame(a)
        au_corr.columns = ["corr"]
        print(au_corr.info())
        corr_limit = float(input("Set correlation strength (in decimals): "))
        high_corr = au_corr.loc[au_corr['corr'] >= corr_limit]
        return high_corr
            
    df_corr = df_pca_f.select_dtypes(['number'])
    # Generate absolute correlations dataframe with top 10 correlations
    top_corr = pd.DataFrame(data = get_top_abs_correlations(df_corr))
    print("------------------------------------------------------------------------------------")
    print("#1 - Significant correlatons observed:")
    print("------------------------------------------------------------------------------------")
    print(top_corr)
        
    pca = PCA(n_components=max_columns)
    pca.fit(df_pca_f)
    print("PCA explained variance ratio report:")
    pca_var_ratio = pca.explained_variance_ratio_
    pca_variance_ratio = np.round(pca_var_ratio, 2)
    print(pca_variance_ratio)
    
    components_input = int(input("Choose the minimum number of principal components: "))
    pca_f = PCA(n_components = components_input)
    pca_f.fit(df_pca_f)
    
    # generate overall dataframe with PCA components for correlation evaluation
    X_PCA_components = pca_f.transform(df_pca_f)
    X_PCA_components = pd.DataFrame(X_PCA_components)
    # Generating PCA column names
    col_list = []
    n = components_input
    # iterating till the range 
    for i in range(0, n): 
#                colname = df_pca_f.columns[i]
        PCA_no = str(i)
        new_colname = "PCA_" + PCA_no
        col_list.append(new_colname)
    X_PCA_components.columns = col_list
    overall_corr_df = pd.concat([df_pca_f, X_PCA_components], axis=1, sort=False)
    corr = overall_corr_df.corr()
    corr_df = pd.DataFrame(corr)
#    print(corr_df)
    filter_col = [col for col in corr_df if col.startswith('PCA_')]
    pca_corr = corr_df[filter_col].abs()
    pca_corr = pca_corr.round(2)
    col_list = pca_corr.columns
    print("------------------------------------------------------------------------------------")
    print("#2 - PCA correlations with other metrics for selected components:")
    print("------------------------------------------------------------------------------------")
    print(col_list)
    n = len(col_list)
    for i in range(0,n):
        colname = col_list[i]
        print(colname)
        single_pca = pca_corr.copy()
        single_pca[colname] = single_pca[colname].abs()
        single_pca = single_pca.sort_values([colname], ascending=False)
        corr_limit = float(input("Set correlation strength (in decimals): "))
        single_pca = single_pca[single_pca[colname] > corr_limit]
        single_pca = single_pca.ix[1:]
        print(single_pca[colname])
    implement_trigger = input("Implement PCA on dataset (Y/N)? ")
    if implement_trigger == "Y":
        components_input = int(input("Set number of principal components (int): "))
        pca_f = PCA(n_components = components_input)
        pca_f.fit(df_pca_f)
        
        # generate overall dataframe with PCA components for correlation evaluation
        X_PCA_components = pca_f.transform(df_pca_f)
        X_PCA_components = pd.DataFrame(X_PCA_components)
        # Generating PCA column names
        col_list = []
        n = components_input
        # iterating till the range 
        for i in range(0, n): 
    #                colname = df_pca_f.columns[i]
            PCA_no = str(i)
            new_colname = "PCA_" + PCA_no
            col_list.append(new_colname)
        X_PCA_components.columns = col_list
        df_out = pd.concat([df, X_PCA_components], axis=1, sort=False)
    else:
        df_out = df.copy()
        
    def drop_check(df):
        print(df.dtypes)
        print("---------------------------------------------------------------------------------------")
        drop_check = input("QUESTION: Any columns that need to be excluded (Y/N)? ")
        print("---------------------------------------------------------------------------------------")
        if drop_check == "Y":
            # creating an empty list 
            cols = [] 
            # number of elemetns as input 
            print("---------------------------------------------------------------------------------------")
            n = int(input("Enter number of columns to be dropped : ")) 
            print("---------------------------------------------------------------------------------------")
            # iterating till the range 
            for i in range(0, n): 
                print("---------------------------------------------------------------------------------------")
                ele = input("Enter column name/s (one at a time): ")
                print("---------------------------------------------------------------------------------------")
                cols.append(ele) # adding the element 
            # drop identified columns
            print(cols)
            df_1 = df.drop(cols, axis=1)           
        else:
            df_1 = df.copy()
        export_df = input("Save copy of modified df for further steps (Y/N)? ")
        if export_df == "Y":
            print("Exporting following DF for future steps...")
            time_executed = datetime.datetime.now()
            string = identified_db + str(time_executed)
            name_for_logging = ''.join(e for e in string if e.isalnum())
            name_for_logging_2 = name_for_logging + "_pca.csv"
            df_1.to_csv(name_for_logging_2, index=False)
            print("File exported: '%s'" % (name_for_logging_2))
            print("")
        else:
            pass
#        print("")
#        print("Exporting following DF for future steps...")
#        time_executed = datetime.datetime.now()
#        string = identified_db + str(time_executed)
#        name_for_logging = ''.join(e for e in string if e.isalnum())
#        name_for_logging_2 = name_for_logging + "_pcaImplementation" + ".csv"
#        df_1.to_csv(name_for_logging_2, index=False)
#        print("File exported: '%s'" % (name_for_logging_2))
#        print("")
#        return df_1
    
    drop_check(df_out)
#    return df_f
#    print(pca_f.explained_variance_ratio_)
#
#df_f = pca_implementation(df)