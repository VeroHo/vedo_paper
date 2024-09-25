import pandas as pd
import numpy as np
# from numpy import mean, std
# import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, RocCurveDisplay, auc
# from sklearn.model_selection import StratifiedKFold, permutation_test_score
from numpy import mean
from scipy import stats
from sklearn.model_selection import permutation_test_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
# from sklearn import linear_model
import warnings

warnings.filterwarnings('ignore')
scaler = StandardScaler()

rc_ticks = {
    "text.usetex": False,
    "font.size": 8,
    "legend.fontsize": 4,
    "legend.title_fontsize": 4,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "lines.linewidth": 0.5,
    "axes.linewidth": 0.4,
    "lines.markersize": 3,
    "xtick.major.size": 2.5,
    "xtick.major.width": 0.5,
    "xtick.major.pad": 1,
    "xtick.minor.size": 1.5,
    "xtick.minor.width": 0.5,
    "xtick.minor.pad": 1,
    "ytick.major.size": 2.5,
    "ytick.major.width": 0.5,
    "ytick.major.pad": 1,
    "ytick.minor.size": 1.5,
    "ytick.minor.width": 0.5,
    "ytick.minor.pad": 1,
    "axes.labelpad": 1,
    "axes.titlepad": 1
}

def run_k_fold_CV(X, X_scaler, y, classifier, coeff_or_not, run_perm, data_type):
    data_set_size = len(y)
    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=36851234)
    # initiate all the empty variables 2 be filled
    tprs = []
    aucs = []
    data = pd.DataFrame([])
    y_true, y_pred_train, y_auc_test = list(), list(), list()
    mean_fpr = np.linspace(0, 1, 100)
    # ----------------------------------------------------------------------------

    fig, ax = plt.subplots()
    for i, (train, test) in enumerate(cv.split(X, y)):
        # fit the classifier
        classifier.fit(X_scaler[train, :], y.iloc[train])
        # get training prediction
        yhat_train = classifier.predict(X_scaler[train, :])
        # get recall and store
        y_pred_train.append(roc_auc_score(y.iloc[train], yhat_train))
        viz = RocCurveDisplay.from_estimator(
            classifier,
            X_scaler[test, :],
            y.iloc[test],
            name="ROC fold {}".format(i),
            alpha=0.3,
            lw=1,
            ax=ax,
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

        if coeff_or_not == True:
            new_coef = pd.DataFrame(classifier.coef_, columns=X.columns)
            data = data.append(new_coef, verify_integrity=False, ignore_index=False, )
        if coeff_or_not == 'RF':
            data = data.append(pd.DataFrame(classifier.feature_importances_, index=X.columns).T)
        if coeff_or_not == 'perm':
            perm_importance = permutation_importance(classifier, X.iloc[test], y.iloc[test],
                                                     n_repeats=1,
                                                     random_state=0)
            data = data.append(pd.DataFrame(perm_importance.importances_mean,
                                            index=X.columns).T)
    # calculate accuracy

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = stats.sem(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )
    print('Mean ROC-AUC Training: %.3f' % mean(y_pred_train))
    print('Mean ROC Testing: %.3f' % mean_auc)

    y_auc_test = [mean(y_pred_train), np.std(y_pred_train), mean_auc, std_auc]

    if run_perm == True:
        score_data, perm_scores_data, pvalue_data = permutation_test_score(
            classifier, X, y, scoring="roc_auc", cv=cv, n_permutations=10
        )
        print('Permutation p-value: %.3f' % pvalue_data)
        # np.savetxt(
        #    str('/home/lisbet/Data/2022_HegazyBerlin/PredictiveModels/Data/R_metav8_RP_LogReg_' + data_type + '_pval.csv'),np.arange(pvalue_data),delimiter=',' )
        print('Mean ROC-AUC Permutation: %.3f' % mean(perm_scores_data))
        print('Mean ROC-AUC data: %.3f' % mean(score_data))

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
        title="Receiver operating characteristic \n" + str(data_type),
    )
    ax.legend(loc="lower right", prop={'size': 6})

    return (data, classifier, y_auc_test)


def run_over_response_and_treatment(which_loc, which_dataset, printout_console):
    print("treatment:\n================================================")
    data_treatment = only_treated.iloc[:, which_loc]
    data_treatment['Response'] = only_treated['timepoint']
    X, y = Run_analysis(data_frame=data_treatment, treatment_or_response="treatment")

    print("Only" + which_dataset + ":\n================================================")
    scan_over_models(X=X, y=y, data_type=str(printout_console + "_treatment"))

    print("response:\n================================================")

    data_response = only_timepoint_one.iloc[:, which_loc]
    data_response['Response'] = only_timepoint_one['response']
    X, y = Run_analysis(data_frame=data_response, treatment_or_response="response")

    print("Only" + which_dataset + ":\n================================================")
    scan_over_models(X=X, y=y, data_type=str(printout_console + "_response"))


def scan_over_models(X, y, data_type):
    X_scaler = scaler.fit_transform(X)
    X_scaler = scaler.transform(X)

    print("=======================Linear Regression==========================")
    svm_type = 'linear_regression'
    clf = LogisticRegression()
    data, model, y_pred_test = run_k_fold_CV(X, X_scaler, y, classifier=clf, coeff_or_not=True, run_perm=True,
                                             data_type=str(data_type + "_" + svm_type))
    data = data.reindex(data.abs().mean().sort_values(ascending=False).index, axis=1)
    df_ytest = pd.DataFrame(y_pred_test)
    df_ytest.to_csv(
        str('R_LogisticRegression_' + data_type + '_pred_test.csv'))
    data.to_csv(
        str('R_LogisticRegression_' + data_type + '_coefficients.csv'))
    features = data.abs().mean().sort_values(ascending=False)

    print("=======================LASSO==========================")
    svm_type = 'LASSO'
    clf  = LogisticRegression(penalty='l1', solver='liblinear', C = 1)
    data, model, y_pred_test = run_k_fold_CV(X, X_scaler, y, classifier=clf, coeff_or_not=True, run_perm=True,
                                             data_type=str(data_type + "_" + svm_type))
    data = data.reindex(data.abs().mean().sort_values(ascending=False).index, axis=1)
    df_ytest = pd.DataFrame(y_pred_test)
    df_ytest.to_csv(
        str('R_LASSO_' + data_type + '_pred_test.csv'))
    data.to_csv(
        str('R_LASSO_' + data_type + '_coefficients.csv'))
    features = data.abs().mean().sort_values(ascending=False)

    return features

def Run_analysis(data_frame, treatment_or_response="response"):
    # remove the NaN entry rows = these proteins were not detected

    data_frame.dropna(axis=0, how='any', inplace=True)

    # make life a bit easier
    if treatment_or_response == "response":
        data_frame_subset = data_frame[(data_frame.Response == 'R') | (data_frame.Response == 'NR')]
        data_frame_subset['Response_bin'] = np.where(data_frame_subset['Response'] == "R", 1, 0)
    elif treatment_or_response == "treatment":
        # data_frame_subset = data_frame[(data_frame.Response == '1') | (data_frame.Response == '3')]
        data_frame_subset = data_frame
        data_frame_subset['Response_bin'] = np.where(data_frame_subset['Response'] == 3, 1, 0)
    data_frame_subset = data_frame_subset.drop(columns='Response')
    # olink_subset.dropna(axis = 1, how = 'any', inplace = True)

    # prepare for the svm training process
    # -----------------------------------------------------------------------

    X = data_frame_subset.drop('Response_bin', axis=1)
    y = data_frame_subset.Response_bin
    print("The number of patients:", len(y), "\n================================================")
    return (X, y)


def extract_features(features, data_set_name):
    print("Whole feature space is:" + str(features.index) + " \n")
    #for i in [0.5, 0.25, 0.2, 0.1, 0.05]:
    for i in [0.5]:
        selected_features = features.index[0:int(round(len(features) * i, 0))]
        if len(selected_features) > 0:
            print("Selected features are:" + str(selected_features) + " \n")
            contrast_data = only_timepoint_one[selected_features]
            contrast_data['Response'] = only_timepoint_one['response']
            # remove the NaN entry rows = these proteins were not detected
            X, y = Run_analysis(data_frame=contrast_data)

            print("---- " + str(data_set_name) + " selected " + str(i) + " features:=====================\n")
            scan_over_models(X=X, y=y, data_type='selected' + str(data_set_name) + '__' + str(i))
        else:
            print("No features left!")

def run_modality(input_features, data_modality_name):
    contrast_data = only_timepoint_one.iloc[:, np.r_[input_features]]
    contrast_data['Response'] = only_timepoint_one['response']
    # remove the NaN entry rows = these proteins were not detected
    X, y = Run_analysis(data_frame=contrast_data)

    print(str(data_modality_name) + ":\n================================================")
    features = scan_over_models(X=X, y=y, data_type=str(data_modality_name))
    extract_features(features=features, data_set_name=data_modality_name)