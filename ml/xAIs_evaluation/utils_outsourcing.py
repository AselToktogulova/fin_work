import ipywidgets as widgets
import pandas as pd
import numpy as np
import json
from collections import OrderedDict
import tabulate
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from sklearn.tree import plot_tree
import xgboost as xgb
import joblib

from IPython.display import display, HTML
import ipywidgets as widgets
import shap
import names
from ipywidgets import interact, interact_manual, fixed
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
#pd.set_option('display.max_colwidth', -1)

df = pd.read_csv('./../data/german_onehotencoded.csv')
ds = df.to_numpy()
ALL = 'ALL'
class Kunde:
    def __init__(self):
        selected_id = -1
        selected_name = ""

tf = open("./../random_names.json", "r")
names_dict = json.load(tf)

tf = open("./../random_names_reversed.json", "r")
names_dict_reversed = json.load(tf)

kunde = Kunde()

dt = pickle.load(open('./../models/dt_eval.model', 'rb'))
model_xgb = xgb.Booster()
model_xgb.load_model('./../models/xgboost_eval.model')

Xd = xgb.DMatrix(ds[:, :-2], label=ds[:, -2: -1])
ex_filename = './../models/explainer.bz2'

explainer = joblib.load(filename=ex_filename)
shap_values = explainer.shap_values(Xd)
shap_values2 = explainer(Xd)

def unique_sorted_values_plus_ALL(array):
    unique = [x for x in array.unique().tolist()]
    unique.sort()
    unique.insert(0, ALL)
    return unique

dropdown_v1 = widgets.Dropdown(options = names_dict.values())

def dropdown_v1_eventhandler(change):
    if (change.new == ALL):
        display(names_dict.values())
    else:
        kunde.selected_name = change.new
        kunde.selected_id = names_dict_reversed[str(kunde.selected_name)]
        display(df.iloc[int(kunde.selected_id):int(kunde.selected_id)+1])

def plot_decision_tree(model, feature_names, class_names):
    # plot_tree function contains a list of all nodes and leaves of the Decision tree
    tree = plot_tree(model, feature_names = feature_names, class_names = class_names,
                     rounded = True, proportion = True, precision = 2, filled = True, fontsize=10)
    # I return the tree for the next part
    return tree

def plot_decision_path_tree(model, X, class_names=None, size=(15, 12)):
    fig = plt.figure(figsize=size)
    class_names = model.classes_.astype(str) if type(class_names) == type(None) else class_names
    feature_names = X.index if type(X) == type(pd.Series()) else X.columns

    # Getting the tree from the function programmed above
    tree = plot_decision_tree(model, feature_names, class_names)

    # Get the decision path of the wanted prediction
    decision_path = model.decision_path([X])

    # Now remember the tree object contains all nodes and leaves so the logic here
    # is to loop into the tree and change visible attribute for components that
    # are not in the decision path
    for i in range(0,len(tree)):
        if i not in decision_path.indices:
            plt.setp(tree[i],visible=False)

def lang():
  langSelect = ["English","Deustche","Espanol","Italiano","한국어","日本人"]
  print(langSelect[0])

def show_dt():
    fig = plt.figure(figsize=(50, 50))
    plot_decision_tree(dt, df.columns[:-2], ['0','1'])
    plt.show()

def show_shap_global2():
    shap.summary_plot(shap_values, feature_names=df.columns, plot_type="bar")

def show_shap_global():
    shap.summary_plot(shap_values, ds[:, :-2], feature_names=df.columns)

def classify_dt():
    #print(dt.predict(df.columns[:-2][kunde.selected_id]))
    if dt.predict(df.iloc[kunde.selected_id,:-2].to_numpy().reshape(1,-1))[0] == 1:
        print('es besteht Kreditrisiko')
    else:
        print('Ein Kredit kann genehmigt werden')

def plot_decision_path_tree(model, class_names=None, size=(15, 30)):
    X = df.iloc[kunde.selected_id,:-2]
    fig = plt.figure(figsize=size)
    class_names = model.classes_.astype(str) if type(class_names) == type(None) else class_names
    feature_names = X.index if type(X) == type(pd.Series()) else X.columns

    # Getting the tree from the function programmed above
    tree = plot_decision_tree(model, feature_names, class_names)

    # Get the decision path of the wanted prediction
    decision_path = model.decision_path([X])

    # Now remember the tree object contains all nodes and leaves so the logic here
    # is to loop into the tree and change visible attribute for components that
    # are not in the decision path
    for i in range(0,len(tree)):
        if i not in decision_path.indices:
            plt.setp(tree[i],visible=False)

def explain_dt():
    plot_decision_path_tree(dt, class_names=['0','1'])

def classify_xgb():
    dtest = xgb.DMatrix(np.float32(df.iloc[kunde.selected_id, :-2].to_numpy().reshape(1, -1)))
    pred = round(model_xgb.predict(dtest)[0])
    if pred == 1:
        print('es besteht Kreditrisiko')
    else:
        print('Ein Kredit kann genehmigt werden')

def show_shap_local():
    dtest = xgb.DMatrix(np.float32(df.iloc[kunde.selected_id, :-2].to_numpy().reshape(1, -1)))
    shap_values = explainer.shap_values(dtest)
    shap.summary_plot(shap_values, ds[kunde.selected_id, :-2])

def show_shap_local2():
    dtest = xgb.DMatrix(np.float32(df.iloc[kunde.selected_id, :-2].to_numpy().reshape(1, -1)))
    shap_values = explainer.shap_values(dtest)
    shap.plots.bar(shap_values[0])


#shap.summary_plot(shap_values)
#plt.show()
def create_image_shap():
    shap.force_plot(explainer.expected_value, shap_values[kunde.selected_id,:], df.iloc[kunde.selected_id, :-2], matplotlib=True, show=False, text_rotation=15, )
    plt.savefig('force_plot.png', dpi=300,
                            transparent=True,
                            bbox_inches="tight",)
    plt.clf()

def create_image_shap2():
    shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[kunde.selected_id], feature_names=df.columns, )
    plt.savefig('force_plot2.png', dpi=300,
                transparent=True,
                bbox_inches="tight", )
    plt.clf()

