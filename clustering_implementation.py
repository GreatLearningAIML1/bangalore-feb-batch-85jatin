# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 21:39:33 2019

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


def clustering_kmeans():
    
    from sklearn.cluster import KMeans
    """
    ---------------------------------------------------------------------------------------
    =============================== CLUSTERS IDENTIFICATION ===============================
    ---------------------------------------------------------------------------------------
    DATAFRAME TYPE: Preprocessed, structured data
    ---------------------------------------------------------------------------------------
    FUNCTION: Explore clusters, dataframe bifercations into identified clusters
    ---------------------------------------------------------------------------------------
    INPUTS REQUIRED: Information in pandas dataframe
    ---------------------------------------------------------------------------------------
    """
    print(clustering_kmeans.__doc__)
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
    df_cluster = df._get_numeric_data()
    # pair panel visual inspection restricted from 2 to 10
    cluster_range = range(2, 10)   
    cluster_errors = []
    for num_clusters in cluster_range:
        clusters = KMeans( num_clusters, n_init = 5)
        clusters.fit(df_cluster)
#                labels = clusters.labels_
#                centroids = clusters.cluster_centers_
        cluster_errors.append( clusters.inertia_ )
    clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors} )
    
    print("Elbow curve information (tabular):")
    print(clusters_df)
    # Elbow plot
    plt.figure(figsize=(8,6))
    plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )
    # fitting 4 clusters as per elbow plot
    clusters_identified = int(input("Clusters noted from elbow curve information above? "))
    cluster = KMeans( n_clusters = clusters_identified, random_state = 7)
    cluster.fit(df_cluster)
    # Creating a new column "GROUP" which will hold the cluster id of each record
    prediction = cluster.predict(df_cluster)
    df_cluster["GROUP_kmeans"] = prediction
    # Generate clustering summary
    print("Mean, SD per variable centroid noted for each cluster: ")
    cluster_details = df_cluster.groupby("GROUP_kmeans").agg(["mean","std"])
    cluster_details = cluster_details.T
    cluster_details.reset_index()
    n = clusters_identified
    cluster_names = []
    for i in range(0, n):
        cluster_no = str(i)
        new_colname = "Cluster_" + cluster_no
        cluster_names.append(new_colname)
        
    cluster_details.columns = cluster_names
    print(cluster_details)
    print("")
    print("Population for each cluster: ")
    print(df_cluster["GROUP_kmeans"].value_counts())
    
    print(df_cluster.info())
    
    trigger_export = input("Generate bifercated datasets based on identified clusters (Y/N)? ")
    if trigger_export == "Y":
        print("Exporting following DF for future steps...")
        time_executed = datetime.datetime.now()
        string = identified_db + str(time_executed)
        name_for_logging = ''.join(e for e in string if e.isalnum())
        n = clusters_identified
        cluster_names = []
        for i in range(0, n):
            cluster_no = str(i)
            new_colname = "Cluster_" + cluster_no
            name_for_logging_2 = name_for_logging + "_" + new_colname + ".csv"
            df_1 = df_cluster.loc[df_cluster['GROUP_kmeans'] == i]
            df_1.to_csv(name_for_logging_2, index=False)
            print("File exported: '%s'" % (name_for_logging_2))
            print("")
    else:
        pass

    


#clustering_kmeans(df)