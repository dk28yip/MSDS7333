import pandas as pd
import os
import sys
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


def create_df_from_file(path, file):

    df = pd.read_csv(path + file)

    print()
    print(df.info())
    print()
    # remove unwanted columns

    lowest_features = ['x0', 'x1', 'x3', 'x4', 'x5', 'x8', 'x9', 'x10',
                       'x11', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19',
                       'x21', 'x22', 'x24', 'x25', 'x26', 'x29', 'x30', 'x31',
                       'x33', 'x34', 'x35', 'x36',  'x39', 'x43',
                       'x44', 'x45', 'x47']

    print("The features that had the lowest importance based off of a",
          "Random Forest were the following and were removed:")
    print(lowest_features)

    df = df.drop(lowest_features, axis=1)

    # convert x32 to a float
    # Fill x32 NA with 0. Get rig of percent sign and convert to float.
    df['x32'].fillna("0.0%", inplace=True)

    df['x32'] = df['x32'].str.rstrip("%").astype(float)/100

    # Fill x37 NA with 0. Get rig of percent sign and convert to float.
    df['x37'].fillna("$0.00", inplace=True)

    df['x37'] = df['x37'].str.lstrip("$").astype(float)

    # Replace remaining NAs with column mean
    df = df.fillna(df.mean())

    df['y'] = df['y'].astype('category')

    print()
    print(df.info())
    print()

    return df


def predictions(df):

    # Split of the features and target variable
    scaler = StandardScaler()

    # All the rows and columns minus the target variable
    x_feat = scaler.fit_transform(df.drop('y', axis=1))

    # Target Variable
    y_target = df['y']

    X_train, X_test, y_train, y_test = train_test_split(x_feat, y_target,
                                                        test_size=0.2,
                                                        random_state=25)

    print(f"No. of X training examples: {X_train.shape[0]}")
    print(f"No. of X testing examples: {X_test.shape[0]}")
    print(f"No. of Y training examples: {y_train.shape[0]}")
    print(f"No. of Y testing examples: {y_test.shape[0]}")

    # Set Best RF Classifier - based on RandomSearchCV

    print()
    print("Based off of a RandomSearchCV the following were",
          " the best params for a RF:")
    params = {'n_estimators': 100, 'min_samples_split': 2,
              'min_samples_leaf': 1,
              'max_features': 5, 'max_depth': 32,
              'criterion': 'entropy'}
    print(params)

    clf = RandomForestClassifier(random_state=None,
                                 n_jobs=-1,
                                 n_estimators=100,
                                 min_samples_split=2,
                                 min_samples_leaf=1,
                                 max_features=5,
                                 max_depth=32,
                                 criterion='entropy')

    # Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train, y_train)

    y_full_pred = clf.predict(x_feat)
    cnf_matrix = confusion_matrix(y_target, y_full_pred)
    print(classification_report(y_target, y_full_pred))
    print()
    print(f"True Negatives: {cnf_matrix[0][0]}")
    print(f"True Positives: {cnf_matrix[1][1]}")
    print(f"False Positives: {cnf_matrix[0][1]}")
    print(f"False Negatives: {cnf_matrix[1][0]}")

    print()
    total_cost = cnf_matrix[1][0]*15 + cnf_matrix[0][1]*35
    print("total cost for this model is : $", total_cost)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='List files in a directory')
    parser.add_argument('--directory', required=True,
                        help='Directory to list files in')
    parser.add_argument('--file',
                        help='filename to be loaded')
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"{args.directory}  is not a valid directory")
        sys.exit(1)

    path = args.directory + args.file

    if not os.path.exists(path):
        print(f"{args.directory}{args.file} does not exist")
        sys.exit(1)

    New_df = create_df_from_file(args.directory, args.file)
    print(New_df)
    predictions(New_df)
