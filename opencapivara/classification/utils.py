import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import Type, List, Tuple
import seaborn as sns
from sklearn import metrics
import requests
#######
## Data Processing
class_hierarchies = {
    # TODO
}

def preprocess_classes(df:pd.DataFrame, 
                       merges: List[Tuple[str, str]], 
                       class_col: str = 'category_truth',
                       remove_list: List[str] = ['--- Please choose ---', 'Unknown'], 
                       limit_classes: List[str] = None,
                       default_threshold: int = 20, #TODO: sugiro ser maior
                       default_class: str = None,
                       verbose: bool = True) -> pd.DataFrame:
    '''
    merges: list of tuples in the format (from, to), with the name of the classes
    If default_class is not provided, classes bellow default_threshold are removed
    '''
    for _from, _to in merges:
        df.loc[df[class_col]==_from, 'category_truth'] = _to
        
    for _class in remove_list:
        df = df[df[class_col]!=_class]
        
    counts = df[class_col].value_counts()
    if default_class:
        df.loc[df[class_col].isin(counts[counts<default_threshold].index), class_col] = default_class
    else:
        df = df[df[class_col].isin(counts[counts>default_threshold].index)]

    if limit_classes!=None:
        df = df[df[class_col].isin(limit_classes)]

    if verbose:
        counts = df[class_col].value_counts()
        print('Number of classes: %d'%len(counts))
        print(counts)
    return df
    
def x_y_separation(df:pd.DataFrame):
    X = df['title'] + df['description']
    y = df['category_truth']

    X = X.astype(str)
    X.fillna('', inplace=True)
    return X, y

#######
## Evaluation
def evaluate_classifier(y_true, y_pred, figsize=(12,12), names:list=None, verbose:int = 3) -> list:  
    if verbose>=2 and type(names)!=type(None):
        print('----- Classes -----')
        _n=0
        for i in names:
            print('Class ' + str(_n) + ': ' + i)
            _n+=1
    if verbose>=1:
        print('')
        print('----- Metrics Report -----')
        m = metrics.classification_report(y_true, y_pred)
        print(m)
        #with open("output_evaluation_metrics.txt", "w") as text_file: print(m, file=text_file)
    if verbose>=3:
        #print('----- Confusion Matrix -----')
        cm = metrics.confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_perc = cm / cm_sum.astype(float) * 100
        annot = np.empty_like(cm).astype(str)
        nrows, ncols = cm.shape
        for i in range(nrows):
            for j in range(ncols):
                c = cm[i, j]
                p = cm_perc[i, j]
                if i == j:
                    s = cm_sum[i]
                    annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
                elif c == 0:
                    annot[i, j] = ''
                else:
                    annot[i, j] = '%.1f%%\n%d' % (p, c)
        cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
        cm.index.name = 'Actual'
        cm.columns.name = 'Predicted'
        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax, linewidths=.5)
    
    f1 = metrics.f1_score(y_true, y_pred, average='macro')
    return [{'name': 'f1-score-macro', 'value': f1}]
    
def regression_report(y_hat, y_test, alpha=0.05, title="Model Evaluation"):

    print ("MAE:                ", metrics.mean_absolute_error(y_test, y_hat))
    print ("RMSE:               ", np.sqrt(metrics.mean_squared_error(y_test, y_hat)))
    print ("Percentual:         ", metrics.mean_absolute_error(y_test,y_hat)/y_test.mean()*100, "%")

    # Previsto vs real
    line = np.arange(np.min([y_test, y_hat]),
                     np.max([y_test, y_hat]),
                     1)

    plt.scatter(y_test,y_hat, Alpha=alpha)
    plt.scatter(line,line, marker='.')
    plt.grid(True)
    plt.title(title)
    plt.xlabel("Real values")
    plt.ylabel("Predicted values")
    
def plot_history(history, figsize=(12, 5), metric='mae'):
    acc = history.history[metric]
    val_acc = history.history['val_'+metric]
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training '+metric)
    plt.plot(x, val_acc, 'r', label='Validation '+metric)
    plt.title('Training and validation '+metric)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
def perturbation_rank(model, x, y, names, regression):
    '''
    Baseado em https://www.researchgate.net/publication/220405223_Factor_Sensitivity_Analysis_with_Neural_Network_Simulation_based_on_Perturbation_System
    '''
    errors = []

    for i in range(x.shape[1]):
        hold = np.array(x[:, i])
        np.random.shuffle(x[:, i])
        
        if regression:
            pred = model.predict(x)
            error = metrics.mean_absolute_error(y, pred)
        else:
            pred = model.predict_proba(x)
            error = metrics.log_loss(y, pred)
            
        errors.append(error)
        x[:, i] = hold
        
    max_error = np.max(errors)
    importance = [e/max_error for e in errors]

    data = {'name':names,'error':errors,'importance':importance}
    result = pd.DataFrame(data, columns = ['name','error','importance'])
    result.sort_values(by=['importance'], ascending=[0], inplace=True)
    result.reset_index(inplace=True, drop=True)
    return result