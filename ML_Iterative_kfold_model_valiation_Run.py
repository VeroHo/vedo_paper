exec(open("/home/lisbet/git-lisbet/code/2021_Hegazy/2022_MachineLearning/20240308_HelperFunctionsUpdated.py").read())
import sys
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import accuracy_score, f1_score

# Open the file in write mode.
sys.stdout = open('../ML_outputLog_Feature_StabilityAnalysis.txt', 'w')

# clinicaliloc = [13,14,15,16,17,21]
clinicaliloc = [12, 13, 14, 15, 16, 19]
clinicalsubiloc = [13,  15]
cytof_freiloc = range(22, 38)
cytof_totaliloc = range(38, 54)
cytof_totalAbiloc = range(54, 70)
cytof_Abiloc = range(71, 87)
facsiloc = range(163, 193)
olinikiloc = range(87, 161)


# Define a dataframe that combines each non-responder with 2 different responders in 4 runs
# ------------------------------------------------------------------------------------------


metadata = pd.read_csv("../Metadata_file.csv", sep=';', index_col=0, decimal=',')

# Filter for timepoint 1 and VDZ treatment
only_timepoint_one = metadata[metadata["timepoint"] == 1]
only_timepoint_one_filtered = only_timepoint_one[
    (only_timepoint_one["response"] == "R") | (only_timepoint_one["response"] == "NR")]


def Run_Stability_Check(data_type_name, feature_indexes, number_iterations):
    data_type = data_type_name
    contrast_data = only_timepoint_one_filtered.iloc[:, np.r_[feature_indexes]]
    contrast_data['Response'] = only_timepoint_one_filtered['response']
    contrast_data.dropna(axis=0, how='any', inplace=True)

    for i in range(0, number_iterations):
        # random.seed(i)
        print("=======================Iteration" + str(i) + "==========================")
        train, test = train_test_split(contrast_data, test_size=0.2, stratify=contrast_data['Response'], random_state=i)
        # Documentation which patients were used in held out data
        print(test["Response"])
        # remove the NaN entry rows = these proteins were not detected
        X, y = Run_analysis(data_frame=train)
        X_scaler = scaler.fit_transform(X)
        X_scaler = scaler.transform(X)
        print("=======================" + str(data_type) + "==========================")
        print("=======================Linear Regression==========================")
        print("=======================Model training")
        # Traing the classifer -------------------------------------------------------------------------------------
        svm_type = 'linear_regression'
        clf = LogisticRegression()
        data, model, y_pred_test = run_k_fold_CV(X, X_scaler, y, classifier=clf, coeff_or_not=True, run_perm=True,
                                                 data_type=str(data_type + "_" + svm_type))
        data = data.reindex(data.abs().mean().sort_values(ascending=False).index, axis=1)
        df_ytest = pd.DataFrame(y_pred_test)
        df_ytest.to_csv(
            str('LogisticRegression_' + data_type + "_Run" + str(
                i) + '_pred_test.csv'))
        data.to_csv(
            str('LogisticRegression_' + data_type + "_Run" + str(
                i) + '_coefficients.csv'))
        mean_per_feature = data.abs().mean()
        sem_per_feature = data.abs().sem()
        features = mean_per_feature - sem_per_feature
        features = features.sort_values(ascending=False)

        # Extract top features -------------------------------------------------------------------------------------
        top_10_percent_count = int(len(features) * 0.1)
        top_10_percent_row_names = features.index[:top_10_percent_count]
        results_df = pd.DataFrame(columns=['Run', 'Accuracy', 'F1'])
        print("=======================Testing of top 10% features")
        contrast_data_subset = contrast_data[top_10_percent_row_names.tolist() + ['Response']]
        contrast_data_subset['Response_bin'] = np.where(contrast_data_subset['Response'] == "R", 1, 0)
        contrast_data_subset = contrast_data_subset.drop(columns='Response')
        X = contrast_data_subset.drop('Response_bin', axis=1)
        X_test = X[X.index.isin(test.index)]
        X_train = X[X.index.isin(train.index)]
        X_train_scaler = scaler.fit_transform(X_train)
        X_train_scaler = scaler.transform(X_train)
        X_test_scaler = scaler.fit_transform(X_test)
        # X_test_scaler = scaler.transform(X_test)
        y = contrast_data_subset.Response_bin
        y_train = y[y.index.isin(train.index)]
        y_test = y[y.index.isin(test.index)]
        print(y_test)
        # Retrain training data on top 10% -------------------------------------------------------------------------------------
        logreg = LogisticRegression()
        logreg.fit(X_train_scaler, y_train)
        # Predict on test set -------------------------------------------------------------------------------------
        y_pred = logreg.predict(X_test_scaler)
        # print(y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:" + str(accuracy))

        f1 = f1_score(y_test, y_pred)
        results_df = results_df.append({'Run': i, 'Accuracy': accuracy, 'F1': f1}, ignore_index=True)
        results_df.to_csv(
            str('LogisticRegression_' + data_type + "_Run" + str(
                i) + '_testing.csv'))

        print("=======================LASSO==========================")
        print("=======================Model training")
        X, y = Run_analysis(data_frame=train)
        X_scaler = scaler.fit_transform(X)
        X_scaler = scaler.transform(X)
        # Traing the classifer -------------------------------------------------------------------------------------
        svm_type = 'LASSO'
        clf = LogisticRegression(penalty='l1', solver='liblinear', C=1)
        data, model, y_pred_test = run_k_fold_CV(X, X_scaler, y, classifier=clf, coeff_or_not=True, run_perm=True,
                                                 data_type=str(data_type + "_" + svm_type))
        data = data.reindex(data.abs().mean().sort_values(ascending=False).index, axis=1)
        df_ytest = pd.DataFrame(y_pred_test)
        df_ytest.to_csv(
            str('LASSO_' + data_type + "_Run" + str(
                i) + '_pred_test.csv'))
        data.to_csv(
            str('LASSO_' + data_type + "_Run" + str(
                i) + '_coefficients.csv'))
        mean_per_feature = data.abs().mean()
        sem_per_feature = data.abs().sem()
        features = mean_per_feature - sem_per_feature
        features = features.sort_values(ascending=False)
        # Extract top features -------------------------------------------------------------------------------------
        top_10_percent_count = int(len(features) * 0.1)
        top_10_percent_row_names = features.index[:top_10_percent_count]
        results_df = pd.DataFrame(columns=['Run', 'Accuracy', 'F1'])
        print("=======================Testing of top 10% features")
        contrast_data_subset = contrast_data[top_10_percent_row_names.tolist() + ['Response']]
        contrast_data_subset['Response_bin'] = np.where(contrast_data_subset['Response'] == "R", 1, 0)
        contrast_data_subset = contrast_data_subset.drop(columns='Response')
        X = contrast_data_subset.drop('Response_bin', axis=1)
        X_test = X[X.index.isin(test.index)]
        X_train = X[X.index.isin(train.index)]
        X_train_scaler = scaler.fit_transform(X_train)
        X_train_scaler = scaler.transform(X_train)
        X_test_scaler = scaler.fit_transform(X_test)
        X_test_scaler = scaler.transform(X_test)
        y = contrast_data_subset.Response_bin
        y_train = y[y.index.isin(train.index)]
        y_test = y[y.index.isin(test.index)]
        print(y_test)
        # Retrain training data on top 10% -------------------------------------------------------------------------------------
        logreg = LogisticRegression(penalty='l1', solver='liblinear', C=0.75)
        logreg.fit(X_train_scaler, y_train)
        # Predict on test set -------------------------------------------------------------------------------------
        y_pred = logreg.predict(X_test_scaler)
        # print(y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:" + str(accuracy))

        f1 = f1_score(y_test, y_pred)
        results_df = results_df.append({'Run': i, 'Accuracy': accuracy, 'F1': f1}, ignore_index=True)
        results_df.to_csv(
            str('LASSO_' + data_type + "_Run" + str(
                i) + '_testing.csv'))

        print("=======================ELASTICNET==========================")
        X, y = Run_analysis(data_frame=train)
        X_scaler = scaler.fit_transform(X)
        X_scaler = scaler.transform(X)
        print("=======================Model training")
        # Traing the classifer -------------------------------------------------------------------------------------
        svm_type = 'ELASTICNET'
        clf = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=0.5)
        data, model, y_pred_test = run_k_fold_CV(X, X_scaler, y, classifier=clf, coeff_or_not=True, run_perm=True,
                                                 data_type=str(data_type + "_" + svm_type))
        data = data.reindex(data.abs().mean().sort_values(ascending=False).index, axis=1)
        df_ytest = pd.DataFrame(y_pred_test)
        df_ytest.to_csv(
            str('ELASTICNET_' + data_type + "_Run" + str(
                i) + '_pred_test.csv'))
        data.to_csv(
            str('ELASTICNET_' + data_type + "_Run" + str(
                i) + '_coefficients.csv'))
        mean_per_feature = data.abs().mean()
        sem_per_feature = data.abs().sem()
        features = mean_per_feature - sem_per_feature
        features = features.sort_values(ascending=False)
        # Extract top features -------------------------------------------------------------------------------------
        top_10_percent_count = int(len(features) * 0.1)
        top_10_percent_row_names = features.index[:top_10_percent_count]
        results_df = pd.DataFrame(columns=['Run', 'Accuracy', 'F1'])
        print("=======================Testing of top 10% features")
        contrast_data_subset = contrast_data[top_10_percent_row_names.tolist() + ['Response']]
        contrast_data_subset['Response_bin'] = np.where(contrast_data_subset['Response'] == "R", 1, 0)
        contrast_data_subset = contrast_data_subset.drop(columns='Response')
        X = contrast_data_subset.drop('Response_bin', axis=1)
        X_test = X[X.index.isin(test.index)]
        X_train = X[X.index.isin(train.index)]
        X_train_scaler = scaler.fit_transform(X_train)
        X_train_scaler = scaler.transform(X_train)
        X_test_scaler = scaler.fit_transform(X_test)
        X_test_scaler = scaler.transform(X_test)
        y = contrast_data_subset.Response_bin
        y_train = y[y.index.isin(train.index)]
        y_test = y[y.index.isin(test.index)]
        print(y_test)
        # Retrain training data on top 10% -------------------------------------------------------------------------------------
        logreg = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=0.5)
        logreg.fit(X_train_scaler, y_train)
        # Predict on test set -------------------------------------------------------------------------------------
        y_pred = logreg.predict(X_test_scaler)
        # print(y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:" + str(accuracy))

        f1 = f1_score(y_test, y_pred)
        results_df = results_df.append({'Run': i, 'Accuracy': accuracy, 'F1': f1}, ignore_index=True)
        results_df.to_csv(
            str('ELASTICNET_' + data_type + "_Run" + str(
                i) + '_testing.csv'))


Run_Stability_Check(data_type_name= "OLINK_FACS", feature_indexes =(olinikiloc, facsiloc), number_iterations= 50)
Run_Stability_Check(data_type_name= "FACS_CyTOFa4b7", feature_indexes =(cytof_Abiloc, facsiloc), number_iterations= 50)
Run_Stability_Check(data_type_name= "FACS_CyTOF_Clinical", feature_indexes =(cytof_freiloc, facsiloc, clinicaliloc), number_iterations= 50)
Run_Stability_Check(data_type_name= "OLINK_FACS_CyTOF", feature_indexes =(olinikiloc, facsiloc, cytof_freiloc), number_iterations= 50)


sys.stdout.close()
