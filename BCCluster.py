
# coding: utf-8

# # BCCluster
# ## Description
# A set of python methods to cluster Bottle Club user information. Designed to be used as a module, and by calling the RunCluster() method. By default, with no parameters, this function will return clusters built from the Bottle Club user information, export the information with clusters to an excel doc, and return a pandas dataframe of the information with clusters. 
# 
# ## Parameters
# The main function to call is the RunCluster() method, which has the following parameters and default values:
# 
# ### Types
# <i>excel_path</i>: A str path to the excel document to read information from<br><br>
# <i>excel_sheet</i>: A str name of the sheet on the excel doc to read from<br><br>
# <i>excel_output</i>: A str path to the excel document to write information to<br><br>
# <i>feature_names</i>: A str array of the names of the features to cluster from (More info in the New Features section)<br><br>
# <i>mini_batch</i>: A boolean that dictates if a batch method is used or not<br><br>
# <i>make_plot</i>: A boolean that dictates if plots of the data is printed to the console<br><br>
# <i>export_to_excel</i>: A boolean that dictates if the data is saved to an excel sheet<br><br>
# <i>num_clusters</i>: An int variable that dictates how many clusters to form the data into<br><br>
# <i>starting</i>: A numpy float array of shape (#Clusters, #Features) that holds the starting points to find clusters from<br><br>
# 
# ### Default Values
# <i>excel_path</i>: "AnonFullBottleClubInfo.xlsx"<br><br>
# <i>excel_sheet</i>: "Categories"<br><br>
# <i>excel_output</i>: "BottleClubClusters.xlsx"<br><br>
# <i>feature_names</i>: ["liquor_percentage", "beer_percentage", "wine_percentage","total_price", "supplies_percentage"]<br><br>
# <i>mini_batch</i>: True<br><br>
# <i>make_plot</i>: False<br><br>
# <i>export_to_excel</i>: True<br><br>
# <i>num_clusters</i>: 3<br><br>
# <i>starting</i>: np.array([[1,0,0,0,0], [0,0,1,0,0], [.5,.5,.5,.5,.5]])<br><br>
# 
# ## New Features
# This script is designed to become as general as possible, and as more possible features are found and are used for clustering the user information, more functions to clean and prepare the features will need to be made. Currently there are 5 functions for cleaning different features. The format for creating a function that creates a type of feature is to add an else if case in the ReadAndCleanDocument() function, and to create a GetFeatureName() function with a single parameter <i>data</i>. If your feature name is some feature_name, then the following will need to be added:
# 
#         elif(feature_names[x] == "feature_name"):
#             df[feature_names[x]] = GetFeatureName(data)
#         
#         def GetFeatureName(data):
#             return column_of_feature_name_data
# 
# The function will need to return a single column of data that is created in your function, using the information gained in <i>data</i>. The else if line should be added with the others, so that the function will know to check if the feature is known. If this is not done, the clustering will fail. To be sure that the function will not fail, follow this format exactly when adding new features.
# 
# This data should also be normalized between 0 and 1 if a numeric value, in order to best ensure that the added feature will function well when combined with the other features.
# 
# ## Usability
# ### Running as an imported module
# As it is now, this script is only to be used when clustering the specifically designated Bottle Club user information. By specifying a different excel document, this script can be used for other purposes, but note that new features will likely need to be created, as any formatting issues will throw off the existing feature creation methods. 
# 
# To use in another script, make sure you are running python 3, and have the modules listed in the import section installed. Then use the following commands to get data from the default script functionality:
# 
#         import BCCluster as bc
#         data = bc.RunCluster()
# 
# Note, this function can take some time to run, as the current Bottle Club user data has over 100k rows. This function call will return a pandas dataframe, which can then be used to do further procedures with. Alternatively, the data will also be written to an excel spreadsheet. If this is unneccessary, this can be disabled by passing in export_to_excel as false, demonstrated like so:
# 
#         data = bc.RunCluster(export_to_excel=False)
#         
# This should run substantially faster, but do note that if the data is not saved later on, it will be lost.
# 
# Additionally, note that when run with the default parameters, the data will be split into three clusters, with Cluster 0 denoting the liquor cluster, Cluster 1 denoting the wine cluster, and Cluster 2 being everyone else. This is assured by carefully selcting starting values, through the parameter <i>starting</i>. When customizing this script for uses other than its default, by specifying a starting point for each cluster, you can ensure that the general area of each cluster number remains the same, increasing the cluster number's usability.
# 
# ### Running on the command-line
# This script now supports running from the command line, with several flags as parameters to be passed into the RunCluster() method. All flags are optional, and when they aren't present, the default values for the parameters will be used instead. This section will be a guide to using this script from the command line and how to use the flags. A quicker guide can be found by using the -h flag when running the script from the command line.
# 
# <i>--einp</i>: Flag for str path to excel input file, 1 optional argument<br><br>
# <i>--esht</i>: Flag for str name of excel sheet, 1 optional argument<br><br>
# <i>--eotp</i>: Flag for str path to excel output file, 1 optional argument<br><br>
# <i>--feat</i>: Flag for str name of feature, >=1 optional arguments, if changed, --strt 'rand' must be active<br><br>
# <i>--mini</i>: Flag for bool to trigger batch mode, 1 optional argument<br><br>
# <i>--plot</i>: Flag for bool to trigger plot mode, 1 optional argument<br><br>
# <i>--expo</i>: Flag for bool to trigger save to excel mode, 1 optional argument<br><br>
# <i>--clus</i>: Flag for int number of clusters to find, 1 optional argument, if changed, --strt 'rand' must be active<br><br>
# <i>--strt</i>: Flag for str name of starting point mode, 1 optional argument, choose 'rand' or 'def'<br><br>
# 
# #### Examples of usage:
# 
# Run default:
# 
#         >BCCluster.py
#         
# Get help:
# 
#         >BCCluster.py -h
#         
# Run with batch mode off:
# 
#         >BCCluster.py --mini False
#         
# Run with 4 clusters:
# 
#         >BCCluster.py --clus 4 --strt 'rand'
#         
# Run with 5 clusters, in plot mode, with only a few selected features:
# 
#         >BCCluster.py --clus 5 --strt 'rand' --plot True --feat "liquor_percentage" 
#             "wine_percentage" "total_price"
#             
# As of right now, specific starting points other than the default cannot be defined from the command line. This will be changed in the future.

# ## Import Statements:

# In[1]:


import scipy.special as sp
import numpy as np
import argparse as ag
import pandas as pd
import networkx as nx
import time as tm
import sklearn.cluster as skl
import sklearn.feature_selection as fs
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D


# ## Draw Functions:

# In[2]:


def Draw3D(predictions, data, f1_name, f2_name, 
           f3_name, fit=True):
    colors = ["k", "k", "k", "k", "k"]
    data = data.iloc[range(2, 502),:]
    if(fit):
        colors = ["b", "c", "k", "m", "g"]
    fig = plt.figure()
    ax = Axes3D(fig)
    for ii in range(0, int(data.shape[0])):
        col = colors[int(predictions[ii])]
        ax.scatter(data.iloc[ii][f1_name], data.iloc[ii][f2_name], 
                   data.iloc[ii][f3_name], color = col)
    
    ax.set_xlabel(f1_name)
    ax.set_ylabel(f2_name)
    ax.set_zlabel(f3_name)
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_zlim(0,1)
    plt.show()


# In[3]:


def Draw2D(predictions, data, f1_name, f2_name, fit=True):
    colors = ["k", "k", "k", "k", "k"]
    data = data.iloc[range(2, 502),:]
    if(test):
        colors = ["b", "c", "k", "m", "g"]
    fig = plt.figure()
    for ii in range(0, int(data.shape[0])):
        col = colors[int(predictions[ii])]
        plt.scatter(data.iloc[ii][f1_name], data.iloc[ii][f2_name], color = col)
    
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.axis([0,1,0,1])
    plt.show()


# In[4]:


def DrawCombos(predictions, data, feature_names, fit=True):
    for x in range(0, len(feature_names)):
        for y in range(x + 1, len(feature_names)):
            for z in range(y + 1, len(feature_names)):
                Draw3D(predictions, data, feature_names[x], feature_names[y],
                       feature_names[z], fit = fit)
                
    for x in range(0, len(feature_names)):
        for y in range(x + 1, len(feature_names)):
            Draw2D(predictions, data, feature_names[x], feature_names[y], 
                   fit = fit)    


# ## Cleaning Function:

# In[5]:


def ReadAndCleanDocument(excel_path, excel_sheet, feature_names):
    df = pd.read_excel(excel_path, excel_sheet)
    
    df = df.fillna(0)
    data = df.drop([0,1, df.shape[0] - 1], axis=0)
    
    for x in range(0, len(feature_names)):
        if(feature_names[x] == "liquor_percentage"):
            data[feature_names[x]] = GetLiquorPercentage(data)
        elif(feature_names[x] == "beer_percentage"):
            data[feature_names[x]] = GetBeerPercentage(data)
        elif(feature_names[x] == "wine_percentage"):
            data[feature_names[x]] = GetWinePercentage(data)
        elif(feature_names[x] == "supplies_percentage"):
            data[feature_names[x]] = GetSupplyPercentage(data)
        elif(feature_names[x] == "total_price"):
            data[feature_names[x]] = GetTotalPrice(data)
            
            
    data = data.reindex(np.random.permutation(data.index))
    df2 = pd.DataFrame()
    df2["IDCardNumber"] = data.iloc[:,0]
    
    for x in range(0, len(feature_names)):
        df2[feature_names[x]] = data[feature_names[x]].values.astype(np.float64)
    
    for x in range(0, len(feature_names)):
        df2 = df2.drop(df2[df2[feature_names[x]] < 0].index)
        
        
    return df2


# In[6]:


def ReadAndCleanDocument_Type(excel_path, excel_sheet):
    print("Reading excel sheet...")
    df = pd.read_excel(excel_path, excel_sheet)
    
    print("Cleaning data: ", end="")
    df2 = df.copy()
    df2 = df2.fillna(0)
    df2 = df2.drop([0,1, df.shape[0] - 1], axis=0)
    df2 = df2.drop(df2[df2["Unnamed: 148"] <= 0].index)
    print("*", end="")
    
    df3 = pd.DataFrame()
    df3["IDCardNumber"] = df2["Unnamed: 0"]
    for x in range(1, 147):
        df3[df["Unnamed: " + str(x)].iloc[1]] = df2["Unnamed: " + str(x)]
    df3["Other"] = df2["Unnamed: 147"]
    df3["Total"] = df2["Unnamed: 148"]
    df3 = df3.drop(df3[df3["Total"] <= 0].index)
    print("*", end="")
    
    df4 = pd.DataFrame()
    df4["IDCardNumber"] = df3["IDCardNumber"]
    for x in range(1, 147):
        df4[df["Unnamed: " + str(x)].iloc[1] + " %"] =                 df3[df["Unnamed: " + str(x)].iloc[1]].values.astype(np.float64) / df3["Total"].values.astype(np.float64)
    df4["Other %"] = df3["Other"] / df3["Total"]
    print("*", end="")
    
    data = pd.DataFrame()
    #data["IDCardNumber"] = df4["IDCardNumber"]
    for x in range(1, 147):
        data[df["Unnamed: " + str(x)].iloc[1] + " %"] = df4[df["Unnamed: " + str(x)].iloc[1] + " %"].values.astype(np.float64)
    data["Other %"] = df4["Other %"].values.astype(np.float64)
    print("*", end="")
    
    for x in range(0, data.shape[1]):
        data = data.drop(data[data.iloc[:,x] < 0].index)
    print("*", end="")
    
    data = data.reindex(np.random.permutation(data.index))
    print("*")
    
    return data


# ## Feature Functions:

# In[7]:


def GetLiquorPercentage(data):
    return data.iloc[:,1] / data.iloc[:,12]


# In[8]:


def GetBeerPercentage(data):
    return data.iloc[:,2] / data.iloc[:,12]


# In[9]:


def GetWinePercentage(data):
    return data.iloc[:,3] / data.iloc[:,12]


# In[10]:


def GetSupplyPercentage(data):
    return (data.iloc[:,12] - data.iloc[:,1] - data.iloc[:,2] - data.iloc[:,3])                                 / data.iloc[:,12]


# In[11]:


def GetTotalPrice(data):
    return 2 * sp.expit(12 * (data.iloc[:,12] /                               data.iloc[:,12].max()).values.astype(np.float64))


# ## Fitting Function:

# In[12]:


def FitData(data, make_plot, mini_batch, num_clusters, 
            feature_names, starting, name):
    testData = data.iloc[:,1:len(feature_names)+1]
    color = np.zeros(testData.shape[0])
    pred = []
    if(make_plot):
        DrawCombos(color, testData, feature_names, fit=False)
    
    if(starting == "def"):
        strtpoints = np.array([[1,0,0,0,0], [0,0,1,0,0], [.5,.5,.5,.5,.5]])
        if(mini_batch):
            kmn = skl.MiniBatchKMeans(n_clusters = num_clusters, 
                                  batch_size = 1000,
                                 init=strtpoints, n_init=1)
            pred = kmn.fit_predict(testData)
        else:
            kmn = skl.KMeans(n_clusters = num_clusters, init=strtpoints, n_init=1)
            kmn.fit(testData)
            pred = kmn.predict(testData)
    else:
        if(mini_batch):
            kmn = skl.MiniBatchKMeans(n_clusters = num_clusters, 
                                  batch_size = 1000,
                                  n_init=100)
            pred = kmn.fit_predict(testData)
        else:
            kmn = skl.KMeans(n_clusters = num_clusters, n_init=10)
            kmn.fit(testData)
            pred = kmn.predict(testData)
            
    if(make_plot):
        DrawCombos(pred, testData, feature_names)
    data[name] = pred
    return data


# ## Exporting Function:

# In[13]:


def ExportToExcel(data, excel_output, sheet_name):
    writer = pd.ExcelWriter(excel_output, engine='xlsxwriter')
    data.to_excel(writer, sheet_name=sheet_name)
    writer.save()


# ## Linkage Graphing Function:

# In[14]:


def GraphLinkages(links, path="BCLinkages"):
    G = nx.DiGraph()
    try:
        G.add_edges_from(links)
    except TypeError:
        G.add_weighted_edges_from(links)
    nx.write_graphml(G, path + ".graphml")
    nx.write_gexf(G, path + ".gexf")
    nx.write_gml(G, path + ".gml")


# ## Create Linkages Function:

# In[15]:


def GetLinkages(data, num_clusters_start=5, num_clusters_end=20, significance_value=.1):
    links = []
    print("Prepping data for " + str(significance_value * 100) + "% linkages")
    sample_data = data.head(n=10000)
    sample_data = sample_data.copy()
    selector = fs.VarianceThreshold(threshold = .99 * (1 - .99))
    selector.fit_transform(sample_data)
    
    print("Beginning Linkages")
    print("Fitting:", end=" ")
    for x in range(num_clusters_start, num_clusters_end + 1):
        
        kmn = skl.MiniBatchKMeans(n_clusters=x, 
                                  batch_size=1000,
                                  n_init=100)
        pred = kmn.fit_predict(data)
        
        temp_data = data.copy()
        temp_data["Sub-Cluster"] = pred
        print("*", end="")
        for y in range(0, x):
            vec = temp_data[temp_data["Sub-Cluster"] == y].mean()
            type_max = ""
            type_max2 = ""
            max_mean = 0.0
            max_mean2 = 0.0
            
            for key, z in vec.iteritems():
                if(key != "IDCardNumber" and key != "Sub-Cluster"):
                    if(z > max_mean):
                        type_max2 = type_max
                        max_mean2 = max_mean
                        type_max = key
                        max_mean = z
                    elif(z > max_mean2):
                        type_max2 = key
                        max_mean2 = z
                        
            if(max_mean2 >= significance_value 
               and not (type_max, type_max2) in links):
                links.append((type_max, type_max2))
                
    return links


# ## Main Cluster Function:

# In[16]:


def RunCluster(excel_path="AnonFullBottleClubInfo.xlsx", excel_sheet="Categories",
              excel_output="BottleClubClusters.xlsx", mini_batch=True, 
              num_clusters=3, make_plot=False,
              feature_names=["liquor_percentage", "beer_percentage", "wine_percentage", 
                 "total_price", "supplies_percentage"],
              export_to_excel=True,
              starting = "def"):
    
    data = ReadAndCleanDocument(excel_path, excel_sheet, feature_names)
    data = FitData(data, make_plot, mini_batch, 
                   num_clusters, feature_names, starting,
                  "Cluster")
    if(export_to_excel):
        ExportToExcel(data, excel_output, "Clusters")
    return data


# ## Sub Cluster Function:

# In[17]:


def RunSubCluster(data, excel_output, mini_batch, num_sub_clusters, make_plot, 
              sub_feature_names, export_to_excel, starting, cluster_num):
    sub_data = data[data["Cluster"] == cluster_num]
    sub_data = FitData(sub_data, make_plot, mini_batch, num_sub_clusters,
                      sub_feature_names, starting, "Sub-Cluster")
    if(export_to_excel):
        ExportToExcel(data, excel_output, "Sub-Clusters")
    return sub_data


# ## Type Linkage Graph Function:

# In[32]:


def GenerateLinkages(excel_path="BottleClubCategorical.xlsx", excel_sheet="Sheet2",
                     num_starting_clusters=5, num_ending_clusters=20,
                     sig_values=[.05, .1, .2]):

    data = ReadAndCleanDocument_Type(excel_path, excel_sheet)
    netLinks = []
    for x in sig_values:
        links = GetLinkages(data, num_starting_clusters, num_ending_clusters, x)
        for each in links:
            netLinks.append((each[0], each[1], x))
        GraphLinkages(links, str(100 * x) + "%BCLinkages")
        print(str(100 * x) + "% Linkage Graphed")
    print()
    
    
    trueNetLinks = []
    sigs = np.array(sig_values)
    sig_values.reverse()
    for x in sig_values:
        tempLinks = [item for item in netLinks if (item[2] == x)]
        if(x == max(sig_values)):
            for element in tempLinks:
                trueNetLinks.append(element)
            continue
        else:
            ridLinks = []
            for y in sigs[np.where(sigs > x)]:
                for el1, el2, el3 in tempLinks:
                    for ell1, ell2, ell3 in trueNetLinks:
                        if ell1 == el1 and ell2 == el2:
                            ridLinks.append((el1, el2, el3))
        
            for element in tempLinks:
                if tuple(element) not in ridLinks:
                    trueNetLinks.append(element)  
                    
    finalNetLinks = []
    for item in trueNetLinks:
        finalNetLinks.append(tuple(item))
    
    df = pd.read_excel("Types.xls")
    
    finalFinalNetLinks = []
    for name1, name2, weight in finalNetLinks:
        edit1 = name1[:4]
        edit2 = name2[:4]
        easyName1 = ""
        easyName2 = ""
        try:
            easyName1 = df["name"].iloc[df[df["type"] == edit1].index].values[0].strip()
        except IndexError:
            easyName1 = "Other"
            
        try:
            easyName2 = df["name"].iloc[df[df["type"] == edit2].index].values[0].strip()
        except IndexError:
            easyName2 = "Other"
            
        finalFinalNetLinks.append(tuple((easyName1, easyName2, weight)))
    
    GraphLinkages(finalFinalNetLinks, "FullBCLinkages")
    print("Complete")


# ## Full Cluster Function:

# In[19]:


def RunFullCluster(excel_path="AnonFullBottleClubInfo.xlsx", excel_sheet="Categories",
              excel_output="BottleClubClusters.xlsx", mini_batch=True, 
              num_clusters=3, make_plot=False,
              feature_names=["liquor_percentage", "beer_percentage", "wine_percentage", 
                 "total_price", "supplies_percentage"],
              export_to_excel=True,
              starting = "def", make_sub=-1, num_sub_clusters=5,
              sub_features = ["brand_selection", "category_selection",
                             "store_number"], sub_starting="def"):
    
    data = RunCluster(excel_path, excel_sheet, excel_output, mini_batch,
                     num_clusters, make_plot, feature_names, export_to_excel,
                     starting)
    sub_data = []
    if(make_sub == -1):
        max_count = 0
        largest_clus = 0
        for x in range(0, num_clusters):
            if(data[data["Cluster"] == x].count > max_count):
                max_count = data[data["Cluster"] == x].count
                largest_clus = x
        sub_data = SubCluster(data, excel_output, mini_batch, 
                             num_sub_clusters, make_plot, sub_features,
                             export_to_excel, sub_starting, largest_clus)
    else:
        sub_data = SubCluster(data, excel_output, mini_batch, 
                             num_sub_clusters, make_plot, sub_features,
                             export_to_excel, sub_starting, make_sub)
    return data, sub_data


# ## Testing Linkages:

# In[33]:


function_time = tm.time()
GenerateLinkages(num_ending_clusters=30, sig_values=[.05, .075, .1, .125, .15, .175, .2, .225, .25])
print("Function took: " + str(tm.time() - function_time) + " seconds to complete")


# ## Command-line Mode:

# In[ ]:


parser = ag.ArgumentParser(description='Cluster Bottle Club User Data.')
parser.add_argument('--einp', nargs=1, type=str, required=False, 
                    help='Excel File Path for Input')
parser.add_argument('--esht', nargs=1, type=str, required=False, 
                    help='Excel Sheet for Input')
parser.add_argument('--eotp', nargs=1, type=str, required=False, 
                    help='Excel File Path for Output')
parser.add_argument('--feat', nargs='+', type=str, required=False, 
                    help='Feature names')
parser.add_argument('--mini', nargs=1, type=bool, required=False, 
                    help='Run program in batches?')
parser.add_argument('--plot', nargs=1, type=bool, required=False, 
                    help='Plot the possible graphs?')
parser.add_argument('--expo', nargs=1, type=bool, required=False, 
                    help='Export Results to Excel?')                   
parser.add_argument('--clus', nargs=1, type=int, required=False, 
                    help='Number of clusters to find') 
parser.add_argument('--strt', nargs=1, type=str, required=False, 
                    help='Choose \'rand\' for random start, \'def\' for default',
                   choices=['rand', 'def']) 
args = parser.parse_args()

excel_path="AnonFullBottleClubInfo.xlsx"
excel_sheet="Categories"
excel_output="BottleClubClusters.xlsx"
mini_batch=True
num_clusters=3
make_plot=False
feature_names=["liquor_percentage", "beer_percentage", "wine_percentage",
                           "total_price", "supplies_percentage"]
export_to_excel=True
starting = "def"
if(args.clus != None):
    num_clusters = args.clus[0]
if(args.einp != None):
    excel_path = args.einp[0]
if(args.esht != None):
    excel_sheet = args.esht[0]
if(args.eotp != None):
    excel_output = args.eotp[0]
if(args.feat != None):
    feature_names = args.feat
if(args.mini != None):
    mini_batch = args.mini[0]
if(args.plot != None):
    make_plot = args.plot[0]
if(args.expo != None):
    export_to_excel = args.expo[0]
if(args.strt != None):
    starting = args.strt[0]

data = RunCluster(excel_path, excel_sheet, excel_output, mini_batch,
           num_clusters, make_plot, feature_names,
           export_to_excel, starting)
print(data.describe())

