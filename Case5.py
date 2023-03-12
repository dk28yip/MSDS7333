import argparse
import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import ConfusionMatrixDisplay


def create_df_from_file(path):

    df = pd.read_csv(path + 'log2.csv')

    # Create category columns
    df['Action'] = df['Action'].astype('category')
    df['Source Port'] = df['Source Port'].astype('category')
    df['Destination Port'] = df['Destination Port'].astype('category')
    df['NAT Source Port'] = df['NAT Source Port'].astype('category')
    df['NAT Destination Port'] = df['Destination Port'].astype('category')

    # Create Action Labels
    le = LabelEncoder()

    # Using fit.transform function to fit label
    # encoder and return encoded labels
    label = le.fit_transform(df['Action'])

    # Appending the array to the dataframe
    df['Label_Action'] = df['Action']
    df['Action'] = label

    # Select columns to scale
    # do not scale source and destination ports... domain knowledge
    cols_to_scale = ['Bytes', 'Bytes Sent', 'Bytes Received', 'Packets',
                     'Elapsed Time (sec)', 'pkts_sent', 'pkts_received']

    # Scale selected columns
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[cols_to_scale] = scaler.fit_transform(df_scaled[cols_to_scale])

    return df_scaled


def predictions(df_scaled):
    # Split the data into train and test sets
    X = df_scaled.drop(['Action', 'Label_Action'], axis=1)
    y = df_scaled['Label_Action']

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, random_state=42)

    print()
    # Our Best LinearSVC Model below
    print("LinearSVC")
    best_model = LinearSVC(tol=.1, loss='hinge', C=1e-09)
    best_model.fit(X_train, y_train)
    print("Best Cross Val Score for LinearSVC")
    print(cross_val_score(best_model, X_train, y_train,
                          cv=5, scoring='accuracy').mean())

    print()
    print('Detailed classification report:')
    print()
    print('The model is trained on full development set.')
    print('The scores are computed on the full evaluation set')
    print()
    y_true, y_pred = y_test, best_model.predict(X_test)
    print(classification_report(y_true, y_pred, zero_division=1))
    print()
    print()

    p = cross_val_predict(best_model, X_test, y_test, cv=5)
    ConfusionMatrixDisplay.from_predictions(y_test, p)

    print()
    print()

    # Best parameters SGD, #Fill in the best parameters here
    print("SGDClassifier")
    best_SGD = SGDClassifier(tol=1, loss='squared_hinge', alpha=0.001)
    best_SGD.fit(X_train, y_train)
    print("Best Cross Val Score for SGDClassifier")
    print(cross_val_score(best_SGD, X_train, y_train,
                          scoring='accuracy', n_jobs=-1, cv=5).mean())

    print()
    print('Detailed classification report:')
    print()
    print('The model is trained on full development set.')
    print('The scores are computed on the full evaluation set')
    print()
    y_true, y_pred = y_test, best_SGD.predict(X_test)
    print(classification_report(y_true, y_pred, zero_division=1))
    print()
    print()
    p = cross_val_predict(best_SGD, X_test, y_test, cv=5)
    ConfusionMatrixDisplay.from_predictions(y_test, p)

    print()
    print()


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
