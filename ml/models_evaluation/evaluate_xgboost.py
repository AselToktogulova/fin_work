from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import xgboost as xgb
from sklearn.metrics import RocCurveDisplay
import sklearn.feature_selection
import sklearn.pipeline
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, auc, precision_recall_curve

import warnings
warnings.filterwarnings('ignore')

# play with these params
params = {
    'max_depth': [3, 4, 5, 6],
    'n_estimators': [150, 200, 250],
    #'subsample': np.arange(0.05, 1.05, 0.1),
    'learning_rate': [0.1, 0.05],
    #'colsample_bytree': np.arange(0.05, 1.05, 0.1),
    #'gamma': [0.05, 0.1, 0.5, 1]

}

if False:
    df = pd.read_csv('./../data/german_onehotencoded.csv')
    ds = df.to_numpy()
    y = ds[:, -2: -1]  # for last column
    x = ds[:, :-2]  # for all but last column

    cv = StratifiedKFold(n_splits=5)

    tprs = []
    aucs = []
    precisionrecalllist = []
    y_pred_scores_list = []
    mean_fpr = np.linspace(0, 1, 100)

    y_real = []
    y_proba = []
    axes_steps = []

    fig, ax = plt.subplots()
    for i, (train, test) in enumerate(cv.split(x, y)):
        xgb_clf = xgb.XGBClassifier(seed=42)

        grid_search_cv = GridSearchCV(xgb_clf,
                                      params,
                                      scoring="roc_auc",
                                      n_jobs=1,
                                      verbose=0)

        grid_search_cv.fit(x[train], y[train])

        model = grid_search_cv.best_estimator_

        pred = model.predict(x[test])

        pred_proba = model.predict_proba(x[test])
        pred_scores = pred_proba.max(axis=1)

        res = ""
        res = res + 'F1: ' + str(f1_score(pred, y[test]))
        res = res + '\nPrecision: ' + str(precision_score(pred, y[test]))
        res = res + '\nRecall: '+ str(recall_score(pred, y[test]))
        tn, fp, fn, tp = confusion_matrix(y[test], pred).ravel()
        false_positive_rate = fp / (fp + tn)
        true_positive_rate = tp / (tp + fp)
        res = res + '\nfalse positive rate: ' + str(false_positive_rate)
        res = res + '\ntrue positive rate:'+ str(true_positive_rate)

        file1 = open(str(i) + "fold_german_onehotencoded.txt", "w")
        file1.write(res)
        file1.close()  # to change file access modes

        precision, recall, _ = precision_recall_curve(y[test], pred_proba[:, 1])

        y_real.append(y[test])
        y_proba.append(pred_proba[:, 1])
        y_pred_scores_list.append(pred_scores)

        lab = 'Fold %d' % (i + 1)
        axes_steps.append((recall, precision, lab))

        viz = RocCurveDisplay.from_estimator(
            model,
            x[test],
            y[test],
            name="ROC fold {}".format(i),
            alpha=0.3,
            lw=1,
            ax=ax,
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

        prec, recall, _ = precision_recall_curve(pred, pred_scores, pos_label=np.array([x[0] for x in y[test].tolist()]))
        precisionrecalllist.append([prec, recall])

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
    )
    ax.legend(loc="lower right")
    plt.savefig('xg_german_onehotencoded_roc.pdf')
    plt.show()

    # print(precisionrecalllist)

    print(np.array(y_proba).shape)
    print(np.array(y_real).shape)
    print(np.array(y_pred_scores_list).shape)

    fig, ax = plt.subplots()

    for i in range(len(axes_steps)):
        ax.step(axes_steps[i][0], axes_steps[i][1], label = axes_steps[i][2])

    y_real_test = np.concatenate(np.array(y_real))
    y_proba_test = np.concatenate(np.array(y_proba))

    precision, recall, _ = precision_recall_curve(y_real_test, y_proba_test)
    lab = 'Mean Pr-Rc curve'
    ax.step(recall, precision, label=lab, lw=2, color='b')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(loc='lower left', fontsize='small')
    plt.savefig('xg_german_onehotencoded_prre.pdf')

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
    )

    plt.show()

    df = pd.read_csv('./../data/creditcard.csv')
    ds = df.to_numpy()
    y = ds[:, -1]  # for last column
    x = ds[:, :-1]  # for all but last column

    cv = StratifiedKFold(n_splits=5)

    tprs = []
    aucs = []
    precisionrecalllist = []
    y_pred_scores_list = []
    mean_fpr = np.linspace(0, 1, 100)

    y_real = []
    y_proba = []
    axes_steps = []

    fig, ax = plt.subplots()
    for i, (train, test) in enumerate(cv.split(x, y)):
        xgb_clf = xgb.XGBClassifier(seed=42)

        grid_search_cv = GridSearchCV(xgb_clf,
                                      params,
                                      scoring="roc_auc",
                                      n_jobs=1,
                                      verbose=0)

        grid_search_cv.fit(x[train], y[train])

        model = grid_search_cv.best_estimator_

        pred = model.predict(x[test])

        pred_proba = model.predict_proba(x[test])
        pred_scores = pred_proba.max(axis=1)
        res = ""
        res = res + 'F1: ' + str(f1_score(pred, y[test]))
        res = res + '\nPrecision: ' + str(precision_score(pred, y[test]))
        res = res + '\nRecall: '+ str(recall_score(pred, y[test]))
        tn, fp, fn, tp = confusion_matrix(y[test], pred).ravel()
        false_positive_rate = fp / (fp + tn)
        true_positive_rate = tp / (tp + fp)
        res = res + '\nfalse positive rate: ' + str(false_positive_rate)
        res = res + '\ntrue positive rate:'+ str(true_positive_rate)

        file1 = open(str(i) + "fold_creditcard.txt", "w")
        file1.write(res)
        file1.close()  # to change file access modes

        precision, recall, _ = precision_recall_curve(y[test], pred_proba[:, 1])

        y_real.append(y[test])
        y_proba.append(pred_proba[:, 1])
        y_pred_scores_list.append(pred_scores)

        lab = 'Fold %d' % (i + 1)
        axes_steps.append((recall, precision, lab))

        viz = RocCurveDisplay.from_estimator(
            model,
            x[test],
            y[test],
            name="ROC fold {}".format(i),
            alpha=0.3,
            lw=1,
            ax=ax,
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

        prec, recall, _ = precision_recall_curve(pred, pred_scores, pos_label=y[test])
        precisionrecalllist.append([prec, recall])

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
    )
    ax.legend(loc="lower right")
    plt.savefig('xg_creditcard_roc.pdf')
    plt.show()


    print(np.array(y_proba).shape)
    print(np.array(y_real).shape)
    print(np.array(y_pred_scores_list).shape)

    fig, ax = plt.subplots()

    for i in range(len(axes_steps)):
        ax.step(axes_steps[i][0], axes_steps[i][1], label = axes_steps[i][2])

    y_real_test = np.concatenate(np.array(y_real))
    y_proba_test = np.concatenate(np.array(y_proba))

    precision, recall, _ = precision_recall_curve(y_real_test, y_proba_test)
    lab = 'Mean Pr-Rc curve'
    ax.step(recall, precision, label=lab, lw=2, color='b')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(loc='lower left', fontsize='small')
    plt.savefig('xg_creditcard_prre.pdf')

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
    )

    plt.show()

df = pd.read_csv('./../data/application_data_onehotencoded.csv')
df = df.fillna(0)
y = df['TARGET'].to_numpy().reshape(-1,1) # for last column

a = [df.columns[0]]
b = df.columns[2:]
res = [*a, *b]
x = df[res].to_numpy()

print(x.shape)
print(y.shape)

cv = StratifiedKFold(n_splits=5)

tprs = []
aucs = []
precisionrecalllist = []
y_pred_scores_list = []
mean_fpr = np.linspace(0, 1, 100)

y_real = []
y_proba = []
axes_steps = []

fig, ax = plt.subplots()
for i, (train, test) in enumerate(cv.split(x, y)):
    xgb_clf = xgb.XGBClassifier(seed=42)

    grid_search_cv = GridSearchCV(xgb_clf,
                                  params,
                                  scoring="roc_auc",
                                  n_jobs=10,
                                  verbose=0)

    grid_search_cv.fit(x[train], y[train])

    model = grid_search_cv.best_estimator_

    pred = model.predict(x[test])

    pred_proba = model.predict_proba(x[test])
    pred_scores = pred_proba.max(axis=1)


    res = ""
    res = res + 'F1: ' + str(f1_score(pred, y[test]))
    res = res + '\nPrecision: ' + str(precision_score(pred, y[test]))
    res = res + '\nRecall: '+ str(recall_score(pred, y[test]))
    tn, fp, fn, tp = confusion_matrix(y[test], pred).ravel()
    false_positive_rate = fp / (fp + tn)
    true_positive_rate = tp / (tp + fp)
    res = res + '\nfalse positive rate: ' + str(false_positive_rate)
    res = res + '\ntrue positive rate:'+ str(true_positive_rate)

    file1 = open(str(i) + "fold_application_data.txt", "w")
    file1.write(res)
    file1.close()

    precision, recall, _ = precision_recall_curve(y[test], pred_proba[:, 1])
    precisionrecalllist.append([precision, recall])

    y_real.append(y[test])
    y_proba.append(pred_proba[:, 1])
    y_pred_scores_list.append(pred_scores)

    lab = 'Fold %d' % (i + 1)
    axes_steps.append((recall, precision, lab))

    viz = RocCurveDisplay.from_estimator(
        model,
        x[test],
        y[test],
        name="ROC fold {}".format(i),
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

    # prec, recall, _ = precision_recall_curve(pred, pred_scores, pos_label=y[test])
    # precisionrecalllist.append([prec, recall])

ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
)
ax.legend(loc="lower right")
plt.savefig('xg_application_data_onehotencoded_roc.pdf')
#plt.show()
print(np.array(y_proba).shape)
print(np.array(y_real).shape)
print(np.array(y_pred_scores_list).shape)

fig, ax = plt.subplots()

for i in range(len(axes_steps)):
    ax.step(axes_steps[i][0], axes_steps[i][1], label= axes_steps[i][2])

y_real_test = np.concatenate(np.array(y_real))
y_proba_test = np.concatenate(np.array(y_proba))

precision, recall, _ = precision_recall_curve(y_real_test, y_proba_test)
lab = 'Mean Pr-Rc curve'
ax.step(recall, precision, label=lab, lw=2, color='b')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.legend(loc='lower left', fontsize='small')
plt.savefig('xg_application_data_onehotencoded_prre.pdf')

ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
)

#plt.show()
