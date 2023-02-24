import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import classification_report
import argparse
import os
import sys
from os.path import join
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold


def create_df_from_file(path):
    data_consolidated = [f for f in os.listdir(path)
                         if not f.startswith('.')
                         and os.path.isfile(join(path, f))]

    # create dataframe
    df = pd.DataFrame()

    for f in data_consolidated:
        data_temp = arff.loadarff(path+'/'+f)
        temp_df = pd.DataFrame(data_temp[0])
        df = pd.concat([df, temp_df], ignore_index=True)

    # fill nans
    for column in df:
        if df[column].isnull().any():
            if (column in df):
                df[column] = df[column].fillna(df[column].mode()[0])
            else:
                df[column] = df[column].fillna(df[column].mean)

    # Turn b'0'/b'1' to 0 and 1
    df['class'] = df['class'].replace([b'0', b'1'], [0, 1])

    return df


def predictions(df):
    X = df.drop(['class'], axis=1)
    y = df['class']

    # Normalize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=5, random_state=888, shuffle=True)

    X_train, X_test, y_train, y_test = \
        train_test_split(X_scaled, y, test_size=0.25, random_state=123)

    # Set Best RF Classifier - based on ROC_AUC
    # given that we have unbalanced data, we set class weight to balanced.

    clf = RandomForestClassifier(class_weight='balanced',
                                 random_state=888,
                                 n_jobs=-1,
                                 n_estimators=100,
                                 min_samples_split=4,
                                 min_samples_leaf=2,
                                 max_features=None,
                                 max_depth=46,
                                 criterion='entropy')

    # Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print()

    print("parameters for the Random Forest:")
    print()
    print(clf.get_params())

    print()
    print("Random Forest Classification Report")
    print()
    print(classification_report(y_test, y_pred))

    print()
    print("Random Forest Cross Validation Score:")
    print()

    cross_array = cross_val_score(clf, X_train, y_train,
                                  cv=cv, scoring='roc_auc')

    print(cross_array)

    print()
    print("Random Forest mean CV Score:")
    print()
    print(np.mean(cross_array))

    print()
    print("-------------------")
    print()

    # XGBoost Classifier portion
    param = {'subsample': .9, 'reg_lambda': None, 'min_child_weight': 3.0,
             'max_depth': 20, 'learning_rate': 0.1, 'gamma': .25,
             'eval_metric': 'auc', 'colsample_bytree': 1,
             'colsample_bylevel': None}
    model = xgb.XGBClassifier()
    model.set_params(**param)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("parameters for the XGBoostClassifier:")
    print()
    print(model.get_params())

    print()
    print("XGBoostClassifier Classification Report")
    print()
    print(classification_report(y_test, y_pred))

    cross_array = cross_val_score(model, X_train, y_train,
                                  cv=cv, scoring='roc_auc')

    print()
    print("XGBoostClassifier Cross Validation Score:")
    print()
    print(cross_array)

    print()
    print("XGBoostClassifier mean CV Score:")
    print()
    print(np.mean(cross_array))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='List files in a directory')
    parser.add_argument('--directory', required=True,
                        help='Directory to list files in')
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"{args.directory} is not a valid directory")
        sys.exit(1)

    New_df = create_df_from_file(args.directory)
    predictions(New_df)
