import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
import random
import pylab as pl
import matplotlib.pyplot as plt
from scipy import optimize
import time
import seaborn as sns




def read_csv(file_path, usecols=None, index_col=None):
    """
    general description: A CSV read data function that loads a panda dataframe and gives general metrics of database description
    input: path file, column to set the index, selection of columns to import
    print-output: successfull legend and dataframe dimensions
    output: pandas dataframe 
    """
    import warnings
    warnings.filterwarnings("ignore")
    import pandas as pd
    import ntpath
    df = pd.read_csv(file_path, usecols, index_col)
    
    head, tail = ntpath.split(file_path)
    print("**",tail,"**","has been loaded succesfully into pandas!")
    print("_____________________________")
    print("")
    print("# of Rows:", df.shape[0])
    print("# of Columns:", df.shape[1])
    print("_____________________________")
    
    return df



def explore_df(df):
    """
    general description: A function that loads a dataframe and gives general description of data variables
    input: dataframe
    output: 
            1. Column types
            2. First 2 rows of the dataframe
    """
    import pylab as pl
    import pandas as pd
    pl.rcParams['figure.figsize'] = (20, 20)

    print("Column types:")
    print("_____________________________")
    print (df.info())
    print("_____________________________")
    print("")
    
    print("Summary statistics from columns:")
    print("_____________________________")
    pl.rcParams['figure.figsize'] = (20, 20)
    #print(df.describe().unstack())
    
    
    return df.hist()
    
def plot_corr_matrix(df):
    '''
    Heatmap of correlation matrix
    Inputs: dataframe
    Returns: Heatmap
            (Green + corr. Red - corr.)
    '''
    import seaborn as sns
    ax = plt.axes()
    with sns.axes_style("white"):
        sns.heatmap(df.corr(), square=True, annot=True, annot_kws={"size": 6})
        ax.set_title('Correlation Matrix')


def count_plot(df,column):
    """
    general description: A function that loads a dataframe and make a distribution plot for categorical values
    input: dataframe,column
    output: plot
    """
    
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    get_ipython().magic('matplotlib inline')

    import seaborn as sns
    sns.set(color_codes=True)

    import plotly
    import plotly.graph_objs as py
    plotly.offline.init_notebook_mode()

    col=str(column)
    histo = df[col].value_counts()
      
    plt.figure(figsize=(20, 14))
    g = sns.countplot(y=col, saturation= 1, data=df)
    plt.title(col + ' distribution')
    plt.xlabel('Frequency')
    plt.ylabel(col)



def count_plot_order(df,column):
    """
    general description: A function that loads a dataframe and make a distribution plot for categorical values in order
    input: dataframe, column
    output: 
            1. plot
    """    
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    get_ipython().magic('matplotlib inline')

    import seaborn as sns
    sns.set(color_codes=True)

    import plotly
    import plotly.graph_objs as py
    plotly.offline.init_notebook_mode()

    col=str(column)
    histo = df[col].value_counts()
      
    plt.figure(figsize=(20, 14))
    g = sns.countplot(y=col, saturation= 1, data=df, order=histo.index)
    plt.title(col + ' distribution')
    plt.xlabel('Frequency')
    plt.ylabel(col)



def finding_na(df):
    """
    general description: Check the proportion of missing values from columns
    input: dataframe
    output: description of the percentage of missing values and its columns
    """
    for c in df.columns:
        if df[c].count() < len(df):
            missing_perc = ((len(df) - df[c].count()) / float(len(df))) * 100.0
            print("%.1f%% NaN values from the colum: %s" %(missing_perc, c))




def nan_zero(df,column):
    """
    general description:   Fill NaN values with 0
    input:                 Dataframe, column
    output:                Dataframe with zeros in the NaN data
    ________________________________________________________________________________
    Other useful information:
    Why NaN? You should be aware that NaN is a bit like a data virus which infects any other object it touches.
    We cannot drop single values from a DataFrame; we can only drop full rows or full columns. 
    By default, dropna() will drop all rows in which any null value is present.
    you can drop NA values along a different axis: axis=1 drops all columns containing a null value: df.dropna(axis=1)
    """
    import pandas as pd
    import sys
    col= str(column)
    
    df[col] = df[col].fillna(0)
    return df    


def nan_zero_ask(df):
    """
    general description:   Fill NaN values with 0
    input:                 Dataframe, column
    output:                Dataframe with zeros in the NaN data
    ________________________________________________________________________________
    Other useful information:
    Why NaN? You should be aware that NaN is a bit like a data virus which infects any other object it touches.
    We cannot drop single values from a DataFrame; we can only drop full rows or full columns. 
    By default, dropna() will drop all rows in which any null value is present.
    you can drop NA values along a different axis: axis=1 drops all columns containing a null value: df.dropna(axis=1)
    """
    import pandas as pd
    import sys
    
    if sys.version_info[0] >= 3:
        raw_input = input
    x = df.columns[pd.isnull(df).sum() > 0].tolist()
    print(x)
    name = raw_input("What columns you want to replace the NaN values for zeros from the columns above")
    
    for i in x:
        if i == name:
            df[col] = df[col].fillna(0)
            return df    
    
    return df



def nan_mean(df,column):
    """
    general description:   Fill NaN values with 0
    input:                 Dataframe, column
    output:                Dataframe with zeros in the NaN data
    ________________________________________________________________________________
    Other useful information:
    Why NaN? You should be aware that NaN is a bit like a data virus which infects any other object it touches.
    We cannot drop single values from a DataFrame; we can only drop full rows or full columns. 
    By default, dropna() will drop all rows in which any null value is present.
    you can drop NA values along a different axis: axis=1 drops all columns containing a null value: df.dropna(axis=1)
    """
    import pandas as pd
    import sys
    
    df[column].fillna(df[column].mean(), inplace=True)
    df[column]= df[column]
    return df


def nan_mean_ask(df):
    """
    general description:   Fill NaN values with mean by detecting NaN columns and confirming columns
    input:                 Dataframe
    raw_input:             Confirmation of column
    output:                Dateframe with zeros in the NaN data
    ________________________________________________________________________________
    Other useful information:
    Why NaN? You should be aware that NaN is a bit like a data virus which infects any other object it touches.
    We cannot drop single values from a DataFrame; we can only drop full rows or full columns. 
    By default, dropna() will drop all rows in which any null value is present.
    you can drop NA values along a different axis: axis=1 drops all columns containing a null value: df.dropna(axis=1)
    """
    
    import pandas as pd
    import sys
    
    if sys.version_info[0] >= 3:
        raw_input = input
    x = df.columns[pd.isnull(df).sum() > 0].tolist()
    print(x)
    
    if len(x)>0:
        name = raw_input("What columns you want to replace the NaN values for mean from the columns above")
        for i in x:
            if i == name:
                df[name].fillna(df[name].mean(), inplace=True)
                df[name]= df[name]
                return df
    else:
        print("No columns with NaN")




def nan_mean_int(df, column):
    """
    general description:   Fill NaN values with mean and converts to integer
    input:                 Dataframe, column
    output:                Dateframe with mean in the NaN data and converted in integer

    """
    import pandas as pd
    import sys
    
    df[column].fillna(df[column].mean().round().astype(int), inplace=True)
    df[column]= df[column].astype(int)
    return df




def nan_mean_int_ask(df):
    """
    general description:   Fill NaN values with mea by detecting NaN columns and confirming columns
    input:                 Dataframe
    raw_input:             Confirmation of column
    output:                Dateframe with mean in the NaN data and converted to integer
    ________________________________________________________________________________
    Other useful information:
    Why NaN? You should be aware that NaN is a bit like a data virus which infects any other object it touches.
    We cannot drop single values from a DataFrame; we can only drop full rows or full columns. 
    By default, dropna() will drop all rows in which any null value is present.
    you can drop NA values along a different axis: axis=1 drops all columns containing a null value: df.dropna(axis=1)
    """
    
    import pandas as pd
    import sys
    
    if sys.version_info[0] >= 3:
        raw_input = input
    x = df.columns[pd.isnull(df).sum() > 0].tolist()
    print(x)
    if len(x)>0:
        name = raw_input("What columns you want to replace the NaN values for mean from the columns above")
        for i in x:
            if i == name:
                df[name].fillna(df[name].mean().round().astype(int), inplace=True)
                df[name]= df[name].astype(int)
                return df
    else:
        print("No columns with NaN")




def scaling(df,column):
    """
    general description:    Scales a specific column from a dataframe
    input:                  Dataframe, column
    output:                 Datframe with the scaled column 
    """
    col=str(column)
    df[column] = StandardScaler().fit_transform(df['column'])

    return df




def nan_category_mode(df, column):
    """
    general description:    Inputs NaN of categories variables with the mode
    input:                  Dataframe, column
    output:                 Datframe with inputed column with mode 
    """
    data[col] = data[col].fillna(df[column].value_counts().index[0])
    return df


def categorize(df,column):
    """
    general description:    Transforms to a category type
    input:                  Dataframe, column
    output:                 Datframe with the transformed column 
    """
    col=str(column)
    df[col]= df[col].astype("category")
    
    object_type= df.dtypes[df.dtypes == "category"].index
    print(object_type)
    return df




def discretize(df,column,bins):
    """
    general description:    Discretize a continuous column from float64 to bins
    input:                  Dataframe, Dataframe-column
    output:                 Datframe with the transform column 
    """
    new_col = 'bins_' + str(column)
    data[new_col] = pd.cut(data[column], bins=bins)
    return data[new_col]



def indexize(df,column):
    """
    general description:    Discretize a continuous column
    input:                  Dataframe, Dataframe-column
    output:                 Datframe with the transformed column 
    """
    col= str(column)
    dataframe = df.set_index(df[col])
    del dataframe[col]
    return dataframe




def to_dummy(df, column):
    """
    general description:    Take a categorical variable and create binary/dummy variables from it
    input:                  Dataframe, Dataframe-column
    output:                 Datframe with the transformed column 
    """
    col= str(column)
    dummy_df = pd.get_dummies(df[col])
    df_new = df.join(dummy_df)
    return df_new



def to_int(df,column):
    """
    general description:    Discretize a continuous column from float64 to int
    input:                  Dataframe, Dataframe-column
    output:                 Datframe with the transform column 
    """
    col=str(column)
    df[col]= df[col].astype(int)
    
    object_type= df.dtypes[df.dtypes == int].index
    print(object_type)
    return df


def randomforest_model(y_column, df):
    """
    general description:    Random forest model evaluation. It creates an out-sample of .75-.25 %. It then runs a model and returns
                            the probabilities for the fitted model for the new dataset
    input:                  Dataframe, column
    output:                 modelo= model name,
                            fit_prob= probabilities of the predicted outcome
                            fit_class= predicted class outcome
                            X_train, X_test, 
                            y_train, y_test 
    """
    target = str(y_column)
    from sklearn.ensemble import RandomForestClassifier
    modelo = RandomForestClassifier()
    
    
    X = df.drop(target, axis=1)
    y = df[target]
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    fit_class = modelo.fit(X_train, y_train)
    fit_prob = modelo.fit(X_train, y_train).predict_proba(X_test)
    
    return modelo, fit_prob, fit_class, X_train, X_test, y_train, y_test




def linearlog_model(y_column, df):
    """
    general description:    Linear Regression model evaluation. It creates an out-sample of .75-.25 %. It then runs a model and returns
                            the probabilities for the fitted model for the new dataset
    input:                  Dataframe, column
    output:                 modelo= model name,
                            fit_prob= probabilities of the predicted outcome
                            fit_class= predicted class outcome
                            X_train, X_test, 
                            y_train, y_test 
    """
    
    target = str(y_column)
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    modelo = LogisticRegression()
    
    
    X = df.drop(target, axis=1)
    y = df[target]
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    fit_class = modelo.fit(X_train, y_train)
    fit_prob = modelo.fit(X_train, y_train).predict_proba(X_test)
    
    return modelo, fit_prob, fit_class, X_train, X_test, y_train, y_test




def roc(y_test, fit_prob):
    """
    general description:    PLot of the ROC curve. 
    input:                  y_test = labeled succesful outcomes, fit_prob=probability of the succesfull outcome
    output:                 plot
    """
    from sklearn.metrics import roc_curve, auc
    import pylab as pl
    
    fpr, tpr, thresholds = roc_curve(y_test, fit_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.05])
    pl.ylim([0.0, 1.05])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title("")
    pl.legend(loc="lower right")
    pl.show()



def accuracy_evaluation(X_test,fit,y_test):
    """
    general description:    Gives you back the accuracy of the model 
    input:                  X_test = feature values from test data set 
                            fit = fitted model
                            y_test = labeled succesful outcomes
    output:                 plot
    """
    y_hat=fit.predict(X_test)
    from sklearn.metrics import accuracy_score
    ac = accuracy_score(y_test, y_hat)
    return ac



### cross validation
#from sklearn.model_selection import cross_val_score
#import numpy as np
#scores = cross_val_score(modelo, X, y, cv=10)
#print(np.mean(scores))



def define_clfs_params(grid_size):
    """
    general description:    Gives you back a specific paramaters set models: Random Forest, Ada BoostClassifier, Logistic Regression, Support Vector Machines, Gradient Boosting Classifier, and K-NN
    
    input:                  Level of the grid parameter permutations. 
    output:                 1. A dictionary with models and default parameters
                            2. A test grid with specific parameters
    """
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
    from sklearn import svm
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier

    clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'SVM': svm.LinearSVC(penalty='l1',random_state=0, dual=False),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        'KNN': KNeighborsClassifier(n_neighbors=3) 
            }
    
    small_grid = { 
    'RF':{'n_estimators': [10,50], 'max_depth': [5,20], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [10,100,]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.1,1]},
    'SVM' :{'C' :[0.01, .1],'penalty':['l1','l2']},
    'GB': {'n_estimators': [10,20], 'learning_rate' : [0.1,0.5],'subsample' : [0.5,1.0], 'max_depth': [5,20]},
    'KNN' :{'n_neighbors': [10,50],'weights': ['uniform','distance']}
           }
   
    
    if (grid_size == 'small'):
        return clfs, small_grid
    else:
        return 0, 0



def plot_precision_recall_n(y_true, y_prob, model_name):
    from sklearn.metrics import precision_recall_curve
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    ax1.set_ylim([0,1])
    ax1.set_ylim([0,1])
    ax2.set_xlim([0,1])
    
    name = model_name
    plt.title(name)
    #plt.savefig(name)
    plt.show()



def generate_binary_at_k(y_scores, k):
    cutoff_index = int(len(y_scores) * (k / 100.0))
    test_predictions_binary = [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return test_predictions_binary


def precision_at_k(y_true, y_scores, k):
    preds_at_k = generate_binary_at_k(y_scores, k)
    precision = precision_score(y_true, preds_at_k)
    return precision



def precision_at_k(y_true, y_scores, k):
    preds_at_k = generate_binary_at_k(y_scores, k)
    precision = precision_score(y_true, preds_at_k)
    return precision



def roc2(y_test, fit_prob):
    """
    general description:    PLot of the ROC curve. 
    input:                  y_test = labeled succesful outcomes, fit_prob=probability of the succesfull outcome
    output:                 plot
    """
    from sklearn.metrics import roc_curve, auc
    import pylab as pl
    
    fpr, tpr, thresholds = roc_curve(y_test, fit_prob)
    roc_auc = auc(fpr, tpr)
    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.05])
    pl.ylim([0.0, 1.05])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title("")
    pl.legend(loc="lower right")
    pl.show()



def clf_loop(models_to_run, clfs, grid, X, y):
    """
    general description:    Loops from  
    input:                  y_test = labeled succesful outcomes, fit_prob=probability of the succesfull outcome
    output:                 plot
    """
    
    from sklearn.cross_validation import train_test_split
    from sklearn.grid_search import ParameterGrid
    from sklearn.metrics import roc_auc_score
    
    
    general_start = time.time()
    results_df =  pd.DataFrame(columns=('model_type','clf', 'parameters', 'auc-roc','p_at_5', 'p_at_10', 'p_at_20'))
    for n in range(1, 2):
        # create training and valdation sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        for index,clf in enumerate([clfs[x] for x in models_to_run]):

            
            parameter_values = grid[models_to_run[index]]
            
            for p in ParameterGrid(parameter_values):
                try:
                   
                    
                    clf.set_params(**p)
                    
                    
                    if models_to_run[index] == 'SVM':
                        start = time.time()
                        y_pred_probs = clf.fit(X_train, y_train).decision_function(X_test)
                        elapsed = time.time() - start
                        print('Time to complete fit and prediction was ', elapsed)
                    
                    else:
                        start = time.time()
                        y_pred_probs = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]
                        elapsed = time.time() - start
                        print('Time to complete fit and prediction was ', elapsed)
                        
                    
                    
                    roc2(y_test, y_pred_probs)
                    
                    

                    
                    y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))
                    results_df.loc[len(results_df)] = [models_to_run[index],clf, p,
                                                       roc_auc_score(y_test, y_pred_probs),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,20.0)]
                    
                    plot_precision_recall_n(y_test,y_pred_probs,clf)
                
                except IndexError as e:
                    print('Error:', e)
                    continue
    general_elapsed = time.time() - general_start
    print(general_elapsed)
    
    return results_df