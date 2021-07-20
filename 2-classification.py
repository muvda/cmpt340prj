import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


class FigNum:
    figNum = 1


def result_extraction(data, selector):
    mask = selector.get_support()
    selected_features = data.loc[:, mask].columns.tolist()
    return selected_features


def features_selection_rfe(data, outcomes, noFeaturesDesired):
    print('Begin Recursive feature elimination')
    train_data, test_data, train_labels, test_labels = train_test_split(data, outcomes, test_size=0.1, random_state=1)
    rfe_estimator = RandomForestClassifier(random_state=1)
    rfe_selector = RFE(estimator=rfe_estimator, step=20, n_features_to_select=noFeaturesDesired)
    rfe_selector = rfe_selector.fit(train_data, train_labels)
    return result_extraction(train_data, rfe_selector)


def features_selection_rfecv(data, outcomes):
    print('Begin Recursive feature elimination with cross-validation')
    train_data, test_data, train_labels, test_labels = train_test_split(data, outcomes, test_size=0.1, random_state=1)
    rfecv_estimator = RandomForestClassifier(random_state=1)
    rfecv_selector = RFECV(estimator=rfecv_estimator, step=20, cv=StratifiedKFold(2), scoring='accuracy',
                           min_features_to_select=1)
    rfecv_selector = rfecv_selector.fit(train_data, train_labels)
    return result_extraction(train_data, rfecv_selector)


def features_selection_lasso(data, outcomes, noFeaturesDesired):
    print('Begin L1-based feature selection')
    train_data, test_data, train_labels, test_labels = train_test_split(data, outcomes, test_size=0.1, random_state=1)
    lr_estimator = LogisticRegression(penalty='l1', solver='saga', max_iter=100000)
    lr_selector = SelectFromModel(lr_estimator, max_features=noFeaturesDesired)
    lr_selector = lr_selector.fit(train_data, train_labels)
    return result_extraction(train_data, lr_selector)


def feature_selection_tree(data, outcomes, noFeaturesDesired):
    print('Begin tree-based feature selection')
    train_data, test_data, train_labels, test_labels = train_test_split(data, outcomes, test_size=0.1, random_state=1)
    rf_estimator = RandomForestClassifier(random_state=1)
    rf_selectors = SelectFromModel(rf_estimator, max_features=noFeaturesDesired)
    rf_selectors = rf_selectors.fit(train_data, train_labels)
    return result_extraction(train_data, rf_selectors)


def examine_different_methods(features, target, noFeaturesDesired):
    print('Examining different Feature Selection Methods (this may take approx 3 minutes)')
    rfe = features_selection_rfe(features, target, noFeaturesDesired)
    rfecv = features_selection_rfecv(features, target)
    lasso = features_selection_lasso(features, target, noFeaturesDesired)
    tree = feature_selection_tree(features, target, noFeaturesDesired)
    print('Result:')
    train_classifier(features, target, rfe, 'Recursive_feature_elimination')
    train_classifier(features, target, rfecv, 'Recursive_feature_elimination_cross-validation')
    print(f'Number of features for RFECV: {len(rfecv)} ')
    train_classifier(features, target, lasso, 'L1-based_feature_selection')
    train_classifier(features, target, tree, 'Tree-based_feature_selection')


def validation(classifier, data, label, testType, classifierString):
    predictions = classifier.predict(data)
    success_margin = accuracy_score(label, predictions)
    conf_matrix = confusion_matrix(label, predictions)
    display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    display.plot()
    plt.figure(FigNum.figNum)
    plt.title(f'{classifierString} {testType} Confusion Matrix ')
    plt.savefig(f'./figures/{classifierString}_{testType}_confusion_matrix.png')
    FigNum.figNum = FigNum.figNum + 1
    print(f'{classifierString} {testType} accuracy : {success_margin}')


def train_classifier(data, label, model_feature, model_name):
    chosen_features_df = data[model_feature]
    train_data, test_data, train_labels, test_labels = train_test_split(chosen_features_df, label, test_size=0.1,
                                                                        random_state=1)
    rf_classifier = RandomForestClassifier(n_estimators=1000, class_weight='balanced',
                                           random_state=np.random.seed(1234))

    rf_classifier.fit(train_data, train_labels)
    validation(rf_classifier, train_data, train_labels, 'train data', model_name)
    validation(rf_classifier, test_data, test_labels, 'test data', model_name)


def main():
    df = pd.read_csv('datasets/dat_cleaned.csv')

    # Get outcome vector and unfiltered set of features
    outcomes = df['Outcome']
    data = df.drop(['Outcome'], axis=1)
    noFeaturesDesired = 10
    examine_different_methods(data, outcomes, noFeaturesDesired)


if __name__ == '__main__':
    main()
