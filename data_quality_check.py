# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 02:40:14 2019
Modified on Mon Jul 8 11:07:00 2019
Version: 1.0 (released on Fri Jul 12 10:08:00 2019)
Version: 1.0 Post-release patch (Sat Jul 13 2019)

@author: Jatin_Thakkar
"""

import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import Binarizer
import warnings
warnings.filterwarnings('ignore')
import os
import sys
import datetime


def data_quality_check():
    """
    ---------------------------------------------------------------------------------------
    ================================== DATA QUALITY CHECK =================================
    ---------------------------------------------------------------------------------------
    DATAFRAME TYPE: Structured data
    ---------------------------------------------------------------------------------------
    FUNCTION: Provide information overview, drop columns, alter dtypes, one-hot encoding &
    missing data summary
    ---------------------------------------------------------------------------------------
    INPUTS REQUIRED: Information in pandas dataframe
    ---------------------------------------------------------------------------------------
    """
    print(data_quality_check.__doc__)
    
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
    
    
    time.sleep(3)   
    
    report_sections = ["#1 Data overview", "#2 Dtypes modifications", "#3 Missing values treatment", 
                       "#4 Outliers detection", "#5 Numerical data scaling", 
                       "#6 Categorical variables treatment", "#7 Dataframe finalization"]
    print(report_sections)                   
    observation_list = []
    
    print("------------------------------------------------------------------------------------")
    print("STEP 0: Information summary")
    print("------------------------------------------------------------------------------------")
    # Eye ball the imported dataset
    print("Dataframe shape:")
    print(df.shape)
    print("")
    print("Description:")
    print(df.describe(include="all").T)
    print("")
#    print("CHECK: Required dtypes manipulations and treating categorical data")
#    print("------------------------------------------------------------------------------------")
#    print("Data type identified for each column:")
#    print(df.dtypes)
#    print("")
    print("====================================================================================")    
    observation_list += [input("OBSERVATIONS/NOTES: ")]
    print("====================================================================================")
    
    print("------------------------------------------------------------------------------------")
    print("STEP 1: Dtypes modifications")
    print("------------------------------------------------------------------------------------")
    
    def numeric_check(df):
        print("---------------------------------------------------------------------------------------")
        numeric_check = input("QUESTION: Any numeric columns recorded as object/complex dtype (Y/N)? ")
        print("---------------------------------------------------------------------------------------")
        if numeric_check == 'Y':
            print("ACTION: Convert object to numeric")
            # creating an empty list 
            cols = [] 
            # number of elemetns as input 
            print("---------------------------------------------------------------------------------------")
            n = int(input("Enter number of columns to be modified : ")) 
            print("---------------------------------------------------------------------------------------")
            # iterating till the range 
            for i in range(0, n): 
                print("---------------------------------------------------------------------------------------")
                ele = input("Enter column name/s (one at a time): ")
                print("---------------------------------------------------------------------------------------")
            cols.append(ele) # adding the element 
            # replace 'any strings' with nan in pandas DataFrame
            mask = df[cols].applymap(lambda x: isinstance(x, (int, float)))
            df[cols] = df[cols].where(mask) 
        else:
            df = df.copy()
        return df
    
    # run numeric columns check
    df_1 = numeric_check(df)
    
    def datetime_check(df):
        print("---------------------------------------------------------------------------------------")
        datetime_check = input("QUESTION: Any datetime columns identified (Y/N)? ")
        print("---------------------------------------------------------------------------------------")
        if datetime_check == 'Y':
            print("ACTION: Convert to datetime")
            # creating an empty list 
            cols = [] 
            # number of elemetns as input 
            print("---------------------------------------------------------------------------------------")
            n = int(input("Enter number of columns to be modified : ")) 
            print("---------------------------------------------------------------------------------------")
            # iterating till the range 
            for i in range(0, n): 
                print("---------------------------------------------------------------------------------------")
                ele = input("Enter column name/s (one at a time): ")
                print("---------------------------------------------------------------------------------------")
            cols.append(ele) # adding the element 
            
            n = len(cols)
            for i in range(0, n):
                colname = cols[i]
                df[colname]= pd.to_datetime(df[colname])
                print("Date range for %s:" %(colname))
                print(min(df[colname]))
                print(max(df[colname]))
        else:
            df = df.copy()
        return df
    df_1o = datetime_check(df_1)
    
    def object_check(df):
        print("---------------------------------------------------------------------------------------")
        object_check = input("QUESTION: Any int/float dtype variable to convert to object (Y/N)? ")
        print("---------------------------------------------------------------------------------------")
        if object_check == "Y":
            print("ACTION: Convert numeric to object")
            # creating an empty list 
            cols = [] 
            # number of elemetns as input 
            print("---------------------------------------------------------------------------------------")
            n = int(input("Enter number of columns to be modified : ")) 
            print("---------------------------------------------------------------------------------------")
            # iterating till the range 
            for i in range(0, n): 
                print("---------------------------------------------------------------------------------------")
                ele = input("Enter column name/s (one at a time): ")
                print("---------------------------------------------------------------------------------------")
                cols.append(ele) # adding the element 
            # convert variable to object type
            df[cols] = df[cols].astype(object)
            df_1 = df.copy()
#            print(df_1.dtypes)
        else:
            df_1 = df.copy()
#            print(df_1.dtypes)
        return df_1
    
    # run object columns check
    df_12 = object_check(df_1o)
    print("====================================================================================")    
    observation_list += [input("OBSERVATIONS/NOTES: ")]
    print("====================================================================================")
    print("------------------------------------------------------------------------------------")
    print("STEP 2: Missing value treatment")
    print("------------------------------------------------------------------------------------")

#    print("Available columns for evaluating missing data:")
#    print(df_12.dtypes)
    
    #capture df information
    info = df_12.describe(include='all')
    print(df_12.info())
    # Get counts for each row
    count_column = pd.DataFrame(info.iloc[0])
#    print(count_column)
    # Get max rows count in df
    shape = df_12.shape
    max_rows = shape[0]
    # flag and count missing values information
    count_column['missing_flag'] = count_column.apply(lambda row: 0 if row['count'] == max_rows else 1, axis=1)
    count_column['missing_%'] = (max_rows - count_column['count']) / max_rows * 100
    missing_data = count_column['missing_flag'].sum()
#    print(missing_data)
    # Note observations
    if missing_data == 0:
        print("NOTE: No data missing!")
        df_123 = df_12.copy()
        print(df_123.info())
    else:
        # generate data loss evaluation if all rows with missing values
        # capture missing columns information
        missing_columns = count_column.loc[count_column['missing_flag'] == 1]
        missing_columns = missing_columns.reset_index()
        print("%s columns have missing data." % (missing_data))
        print("Missing data details:")
        missing_columns.columns = ['variable', 'count', 'missing_flag', 'missing_%']
        print(missing_columns)
#        print(missing_columns.dtypes)
#        print("NOTE: Refer missing_values_treatment function for further steps")
        shape = df_12.shape
        max_rows = shape[0]
        df_withoutNAN = df_12.dropna().copy()
        shape_f = df_withoutNAN.shape
        max_rows_f = shape_f[0]
        data_loss = round(100 - (max_rows_f / max_rows * 100),2)
        cols = list(missing_columns.variable)
#        print(cols)
        
        print("NOTE: %s percent data loss if all rows with missing values are dropped" % (data_loss))
        def missing_values_treatment(df, missing_columns, cols):
            print("---------------------------------------------------------------------------------------")
            treat_all = input("Delete all rows with missing values (Y/N)? ")
            print("---------------------------------------------------------------------------------------")
            if treat_all == "Y":
                result = df.dropna().reset_index()
            else:
                print("ACTION: Select appropriate treatment for each column with missing values")
#                cols = list(missing_columns.variable)
                n = len(cols)
                df_replace = pd.DataFrame([])
#                print(df_replace.shape)
                # iterating till the range 
                for i in range(0, n):
                    colname = cols[i]
                    print("Select Treatment for %s: " % (colname))
                    print("---------------------------------------------------------------------------------------")
                    action = int(input("Drop column(1) / Replace (2): "))
                    print("---------------------------------------------------------------------------------------")
                    if action == 1:
                        result = df.drop(colname, 1)
                    else:
                        specific_value_type = df[colname].dtype.kind
                        if specific_value_type == 'O':
                            print("---------------------------------------------------------------------------------------")
                            strategy = int(input("imputation strategy - most_frequent(1) / replace as 'missing'(2): "))
                            print("---------------------------------------------------------------------------------------")
                            if strategy == 1:
                                imp = SimpleImputer(strategy='most_frequent')
                                X = df[[colname]]
                                X = imp.fit_transform(X)
                                a = pd.DataFrame(X)
                                a.columns = [str(colname) + '_imputed' for col in a.columns]
                                df_replace = df_replace.append(a, ignore_index=True)
                            else:
                                #constant_value = input("input constant string to replace: ")
                                imp = SimpleImputer(strategy='constant', fill_value="missing")
                                X = df[[colname]]
                                X = imp.fit_transform(X)
                                a = pd.DataFrame(X)
                                a.columns = [str(colname) + '_imputed' for col in a.columns]
                                df_replace = df_replace.append(a, ignore_index=True)
                        else:
                            print("---------------------------------------------------------------------------------------")
                            strategy = int(input("imputation strategy - mean(1) / median(2): "))
                            print("---------------------------------------------------------------------------------------")
                            if strategy == 1:
                                imp = SimpleImputer(strategy='mean')
                                X = df[[colname]]
                                X = imp.fit_transform(X)
                                a = pd.DataFrame(X)
                                a.columns = [str(colname) + '_imputed' for col in a.columns]
                                df_replace = df_replace.append(a, ignore_index=True)
                                                                
                            else:
                                imp = SimpleImputer(strategy='median')
                                X = df[[colname]]
                                X = imp.fit_transform(X)
                                a = pd.DataFrame(X)
                                a.columns = [str(colname) + '_imputed' for col in a.columns]
                                df_replace = df_replace.append(a, ignore_index=True)
                            
                result = pd.concat([df, df_replace], axis=1)
            print(result.describe(include="all").T)
            return result, max_rows_f, max_rows
        df_123, max_rows_missing, max_rows = missing_values_treatment(df_12, missing_columns, cols)  
        print(df_123.info())
    print("====================================================================================")    
    observation_list += [input("OBSERVATIONS/NOTES: ")]
    print("====================================================================================")                      
    print("------------------------------------------------------------------------------------")
    print("STEP 3: Outliers detection to identify preprocessing steps")
    print("------------------------------------------------------------------------------------")   
    print("ACTION - Review significant outliers present and overall information loss if outliers excluded")
    print("(Note: Certain algorithms don't need outlier treatments - example SVM)")
    def outlier_detection(df):            
        df_iqr = df.describe().T
        df_iqr['outlier_min'] = df_iqr['25%'] - ((df_iqr['75%'] - df_iqr['25%']) * 1.5)
        df_iqr['outlier_max'] = df_iqr['75%'] + ((df_iqr['75%'] - df_iqr['25%']) * 1.5)
        df_iqr_reference = df_iqr.T
        df_outliers = df.select_dtypes(['number']).copy()
        print("Column-wise outliers range captured:")
        print(df_iqr_reference.T)
        # Get max columns count in df
        shape = df_outliers.shape
        max_columns = shape[1]
        # number of columns as input 
        col_list = []
        n = max_columns 
        # iterating till the range 
        for i in range(0, n): 
            colname = df_outliers.columns[i]
            flagname_min = colname + "_outlier_min"
            flagname_max = colname + "_outlier_max"
            min_v = round(float(df_iqr_reference.iloc[8, i]),1)
            max_v = round(float(df_iqr_reference.iloc[9, i]),1)
#                print(colname, min_v, max_v)
            df_outliers[flagname_min] = df_outliers[colname].apply(lambda x: 1 if x < min_v else 0)
            df_outliers[flagname_max] = df_outliers[colname].apply(lambda x: 1 if x > max_v else 0)
            col_list.append(flagname_min)
            col_list.append(flagname_max)
    
        df_outliers['outlier_filter'] = df_outliers[col_list].sum(axis=1)
        df_outliers['outlier_filter_flag'] = df_outliers['outlier_filter'].apply(lambda x: 1 if x > 0 else 0)
        # Get max rows count in df_outliers
        shape = df_outliers.shape
        max_rows = shape[0]
        df_flagged_outliers = df_outliers[df_outliers["outlier_filter_flag"] > 0]
        df["outlier_filter_flag"]= df_outliers["outlier_filter_flag"]
        shape_f = df_flagged_outliers.shape
        max_rows_f = shape_f[0]
        data_loss = round((max_rows_f / max_rows * 100),2)
        print("Column-wise outlier counts:")
        outliers_res = df_outliers[col_list].sum().reset_index()
        #print(outliers_res)
        outliers_res.columns = ["index","count"]
        outliers_report = outliers_res.loc[outliers_res['count'] > 0]
        print(outliers_report.sort_values('count', ascending=False))
        print("------------------------------------------------------------------------------------")
        print("NOTE: %s percent data loss if all outliers removed" % (data_loss))
        print("------------------------------------------------------------------------------------")
        outlier_treatment_input = input("Drop all outliers (Y/N)?: ")
        if outlier_treatment_input == "Y":
            df = df[df["outlier_filter_flag"] == 0].reset_index()
            df = df.drop("outlier_filter_flag", 1)
        else:
            df = df.copy()
        return df, max_rows_f
            
    df_123o, max_rows_outliers = outlier_detection(df_123)
    print(df_123o.shape)
    print("====================================================================================")    
    observation_list += [input("OBSERVATIONS/NOTES: ")]
    print("====================================================================================")
    print("------------------------------------------------------------------------------------")
    print("STEP 4: Numerical data scaling")
    print("------------------------------------------------------------------------------------")   
    
    def data_scaling(df):
        print("---------------------------------------------------------------------------------------")
        scaling_check = input("QUESTION: Any numeric variable that needs to binned or scaled (Y/N)? ")
        print("---------------------------------------------------------------------------------------")
        if scaling_check == "Y":
            # identifying dtypes
            numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
            newdf = df.select_dtypes(include=numerics)
            num_cols = newdf.columns.values
            
            #kinds = np.array([dt.kind for dt in df.dtypes])
            #all_columns = df.columns.values
            #is_num = kinds != 'O' and kinds!= 'D'
            #num_cols = all_columns[is_num]
            # choosing
            print("List of %s available numeric columns:" % (len(num_cols)))
            print(num_cols)
            # creating an empty dataframe
            df_replace = pd.DataFrame([])
            # number of elemetns as input 
            print("---------------------------------------------------------------------------------------")
            n = int(input("Enter number of columns to be modified (int, input 0 to select all numeric columns): ")) 
            print("---------------------------------------------------------------------------------------")
            if n == 0:
                n = len(num_cols)
                cols = num_cols
            else:
                cols = []
                # iterating till the range 
                for i in range(0, n): 
                    print("---------------------------------------------------------------------------------------")
                    ele = input("Enter column name/s (one at a time): ")
                    print("---------------------------------------------------------------------------------------")
                    cols.append(ele) # adding the element 
                n = len(cols)
            treat_all_same_input = input("Treat all variables with same scaler (Y/N)?" )
            print("Options available #1: Scaling the information with StandardScaler(0), MinMaxScaler(1) or RobustScaler(2)- opt.2 suggested if data has outliers")
            print("Options available #2: Binning the information with KBinsDiscretizer(3), Binarizer(4)")
            
            if treat_all_same_input == "Y":
                print("---------------------------------------------------------------------------------------")
                treament_selection = int(input("Select Treatment for all columns (int):" ))
                print("---------------------------------------------------------------------------------------")
                # treatment iterating till the range
                for i in range(0, n):
                    colname = cols[i]
                    
                    if treament_selection == 0:
                        sca = StandardScaler()
                        X = df[[colname]]
                        X = sca.fit_transform(X)
                        a = pd.DataFrame(X)
                        column_name = str(colname) + '_stdscaled'
                        a.columns = [column_name]
                        df_replace[column_name]= a[column_name]
    #                    df_replace = df_replace.append(a, ignore_index=True)
                    elif treament_selection == 1:
                        sca = MinMaxScaler()
                        X = df[[colname]]
                        X = sca.fit_transform(X)
                        a = pd.DataFrame(X)
                        column_name = str(colname) + '_minmaxscaled'
                        a.columns = [column_name]
                        df_replace[column_name]= a[column_name]
                    elif treament_selection == 2:
                        sca = RobustScaler()
                        X = df[[colname]]
                        X = sca.fit_transform(X)
                        a = pd.DataFrame(X)
                        column_name = str(colname) + '_robustscaled'
                        a.columns = [column_name]
                        df_replace[column_name]= a[column_name]
                    elif treament_selection == 3:
                        print("---------------------------------------------------------------------------------------")
                        n_bins_c = int(input("Set n_bins to categorize data in (ordinal): "))
                        print("---------------------------------------------------------------------------------------")
                        print("Available strategies: uniform, quantile, kmeans")
                        print("---------------------------------------------------------------------------------------")
                        strategy_t = input("Select strategy from above options: ")
                        print("---------------------------------------------------------------------------------------")
                        sca = KBinsDiscretizer(n_bins=n_bins_c, encode='ordinal', strategy=strategy_t)
                        X = df[[colname]]
                        X = sca.fit_transform(X)
                        a = pd.DataFrame(X)
                        column_name = str(colname) + '_kbinsscaled'
                        a.columns = [column_name]
                        df_replace[column_name]= a[column_name]
                    else:
                        print("---------------------------------------------------------------------------------------")
                        threshold_v = float(input("Set threshold for binary categorization (float): "))
                        print("---------------------------------------------------------------------------------------")
                        sca = Binarizer(threshold=threshold_v)
                        X = df[[colname]]
                        X = sca.fit_transform(X)
                        a = pd.DataFrame(X)
                        column_name = str(colname) + '_binaryscaled'
                        a.columns = [column_name]
                        df_replace[column_name]= a[column_name]
                
            else:                
                # treatment iterating till the range
                for i in range(0, n):
                    colname = cols[i]
                    print(df[colname].describe(include="all").T)
                    print("Options available #1: Scaling the information with StandardScaler(0), MinMaxScaler(1) or RobustScaler(2)- opt.2 suggested if data has outliers")
                    print("Options available #2: Binning the information with KBinsDiscretizer(3), Binarizer(4)")
                    print("---------------------------------------------------------------------------------------")
                    treament_selection = int(input("Select Treatment for %s (int): " % (colname)))
                    print("---------------------------------------------------------------------------------------")
                    if treament_selection == 0:
                        sca = StandardScaler()
                        X = df[[colname]]
                        X = sca.fit_transform(X)
                        a = pd.DataFrame(X)
                        column_name = str(colname) + '_stdscaled'
                        a.columns = [column_name]
                        df_replace[column_name]= a[column_name]
    #                    df_replace = df_replace.append(a, ignore_index=True)
                    elif treament_selection == 1:
                        sca = MinMaxScaler()
                        X = df[[colname]]
                        X = sca.fit_transform(X)
                        a = pd.DataFrame(X)
                        column_name = str(colname) + '_minmaxscaled'
                        a.columns = [column_name]
                        df_replace[column_name]= a[column_name]
                    elif treament_selection == 2:
                        sca = RobustScaler()
                        X = df[[colname]]
                        X = sca.fit_transform(X)
                        a = pd.DataFrame(X)
                        column_name = str(colname) + '_robustscaled'
                        a.columns = [column_name]
                        df_replace[column_name]= a[column_name]
                    elif treament_selection == 3:
                        print("---------------------------------------------------------------------------------------")
                        n_bins_c = int(input("Set n_bins to categorize data in (ordinal): "))
                        print("---------------------------------------------------------------------------------------")
                        print("Available strategies: uniform, quantile, kmeans")
                        print("---------------------------------------------------------------------------------------")
                        strategy_t = input("Select strategy from above options: ")
                        print("---------------------------------------------------------------------------------------")
                        sca = KBinsDiscretizer(n_bins=n_bins_c, encode='ordinal', strategy=strategy_t)
                        X = df[[colname]]
                        X = sca.fit_transform(X)
                        a = pd.DataFrame(X)
                        column_name = str(colname) + '_kbinsscaled'
                        a.columns = [column_name]
                        df_replace[column_name]= a[column_name]
                        print(a.describe())
                    else:
                        print("---------------------------------------------------------------------------------------")
                        threshold_v = float(input("Set threshold for binary categorization (float): "))
                        print("---------------------------------------------------------------------------------------")
                        sca = Binarizer(threshold=threshold_v)
                        X = df[[colname]]
                        X = sca.fit_transform(X)
                        a = pd.DataFrame(X)
                        column_name = str(colname) + '_binaryscaled'
                        a.columns = [column_name]
                        df_replace[column_name]= a[column_name]
                        print(a.describe())
            drop_org_input = input("Keep original columns (Y/N)?" )
            if drop_org_input == 'Y':
                result = pd.concat([df, df_replace], axis=1)
            else:
                result = pd.concat([df, df_replace], axis=1)
                result = result.drop(cols, axis=1)
            
        else:
            result = df.copy()
            
        print(result.describe(include="all").T)
        #print(result.info())
        return result
            
    df_1234 = data_scaling(df_123o) 
    #print(df_1234.info())
    print("====================================================================================")    
    observation_list += [input("OBSERVATIONS/NOTES: ")]
    print("====================================================================================")
    print("------------------------------------------------------------------------------------")
    print("STEP 5: Categorical variables treatment")
    print("------------------------------------------------------------------------------------")                        
    
    def categorical_check(df):
        print("---------------------------------------------------------------------------------------")
        categorical_check = input("QUESTION: Any object variable that needs one-hot(1) / label(2) encoding (N if not applicable(N))? ")
        print("---------------------------------------------------------------------------------------")
        if categorical_check == "1":
            print("ACTION: One-hot encoding for categorical variables")
            # creating an empty list 
            cols = [] 
            # number of elemetns as input 
            print("---------------------------------------------------------------------------------------")
            n = int(input("Enter number of columns to be modified : ")) 
            print("---------------------------------------------------------------------------------------")
            # iterating till the range 
            for i in range(0, n): 
                print("---------------------------------------------------------------------------------------")
                ele = input("Enter column name/s (one at a time): ")
                print("---------------------------------------------------------------------------------------")
                cols.append(ele) # adding the element 
            # replace 'any strings' with nan in pandas DataFrame
            df = pd.get_dummies(df, columns = cols, prefix = cols)
        elif categorical_check == "2":
            print("ACTION: Label encoding for categorical variables")
            le = LabelEncoder()
            # creating an empty list 
            cols = [] 
            # number of elemetns as input 
            print("---------------------------------------------------------------------------------------")
            n = int(input("Enter number of columns to be modified : ")) 
            print("---------------------------------------------------------------------------------------")
            # iterating till the range 
            for i in range(0, n): 
                print("---------------------------------------------------------------------------------------")
                ele = input("Enter column name/s (one at a time): ")
                print("---------------------------------------------------------------------------------------")
                cols.append(ele) # adding the element 
            # replace 'any strings' with nan in pandas DataFrame
            # number of elemetns as input 
            n = len(cols) 
            # iterating till the range 
            for i in range(0, n):
                col_name = cols[i]
                df_new = pd.DataFrame(data = df[col_name])
                le.fit(df_new.stack().unique())
                df[col_name] = le.transform(df[col_name])
            df = df.copy()
            
        else:
            df = df.copy()
        return df
    
    # run categorical columns check
    df_12345 = categorical_check(df_1234)
    print("====================================================================================")    
    observation_list += [input("OBSERVATIONS/NOTES: ")]
    print("====================================================================================")
    print("------------------------------------------------------------------------------------")
    print("STEP 6: Dataframe columns finalization")
    print("------------------------------------------------------------------------------------")   
    
    def drop_check(df):
#        print(df.dtypes)
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
        return df_1
    
    # run object columns check
    df_f = drop_check(df_12345)
    print(df_f.info())
    print("====================================================================================")    
    observation_list += [input("OBSERVATIONS/NOTES: ")]
    print("====================================================================================")
    export_df = input("Save copy of modified df for further steps (Y/N)? ")
    if export_df == "Y":
        print("Exporting following DF for future steps...")
        time_executed = datetime.datetime.now()
        string = identified_db + str(time_executed)
        name_for_logging = ''.join(e for e in string if e.isalnum())
        name_for_logging_2 = name_for_logging + "_dqm.csv"
        df_f.to_csv(name_for_logging_2, index=False)
        print("File exported: '%s'" % (name_for_logging_2))
        print("")
    else:
        pass
    print("============================ DATA QUALITY CHECK NOTES ==============================")
    
    print("------------------------------------------------------------------------------------")
#    print("Overall data loss during quality check: %s perc" % (data_loss))
    n = len(observation_list)
    
    # iterating till the range 
    for i in range(0, n):
        reference = report_sections[i]
        print("%s:" %(reference))
        print(observation_list[i])
        print("------------------------------------------------------------------------------------")


#df_1 = data_quality_check(df,identified_db)
#shape = df.shape
#max_rows = shape[0]
#shape_n = df_1.shape
#max_rows_n = shape_n[0]
#trigger_calc = max_rows_n/max_rows
#if trigger_calc != 1:
#    data_loss = round((1 - max_rows_n/max_rows)*100,2)
#    print("NOTE: Overall data loss during quality check: %s perc" % (data_loss))
#else:
#    pass
#print("=================================   END OF REPORT  =================================")
