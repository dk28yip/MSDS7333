import pandas as pd
import numpy as np
import argparse
import os
import sys
import email
from bs4 import BeautifulSoup as BS4

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans


def prediction(df):
    count = CountVectorizer(ngram_range=(1, 2))
    clf = MultinomialNB(alpha=0.1)
    training_data, testing_data = \
        train_test_split(df, test_size=0.2, random_state=25)
    Xtrain = count.fit_transform(training_data['email_body'])
    Xtrain = Xtrain.toarray()
    cluster_array = np.array(training_data['clusters'])
    Xtrain2 = np.hstack((Xtrain, cluster_array.reshape(-1, 1)))
    clf.fit(Xtrain2, training_data['labels'])
    Xtest = count.transform(testing_data['email_body'])
    Xtest = Xtest.toarray()
    cluster_array = np.array(testing_data['clusters'])
    Xtest2 = np.hstack((Xtest, cluster_array.reshape(-1, 1)))
    preds = clf.predict(Xtest2)
    return print(classification_report(testing_data['labels'], preds))


def create_df_from_file(path):
    file_name = []
    contents = []
    types = []
    labels = []
    labelnames = []
    message = ''
    for root, dirs, files in os.walk(path):
        for name in files:
            with open(os.path.join(root, name),
                      'r', encoding='latin1') as f:
                message = ''
                try:
                    x = email.message_from_file(f)
                except UnicodeDecodeError:
                    print("Error in file: Unknown Error")
                if "multipart" in x.get_content_type():
                    if x.is_multipart():
                        for part in x.get_payload():
                            message = message + part.as_string()
                    contents.append(message.replace("\n", " ")
                                    .replace(r'http\S+', ' ')
                                    .replace("-", " "))
                elif "text/plain" in x.get_content_type():
                    contents.append(x.get_payload()
                                    .replace("\n", " ")
                                    .replace(r'http\S+', ' ')
                                    .replace("-", " "))
                elif "text/html" in x.get_content_type():
                    contents.append(BS4(x.get_payload(), "html.parser")
                                    .get_text()
                                    .replace(r'http\S+', ' ')
                                    .replace("\n", " ")
                                    .replace("-", " "))
                types.append(x.get_content_type())
                if "ham" in root:
                    labelnames.append('ham')
                    labels.append(1)
                elif "spam" in root:
                    labelnames.append('spam')
                    labels.append(0)
                file_name.append(os.path.join(root, name))
    df_NB = pd.DataFrame()
    df_NB['Filename'] = file_name
    df_NB['types'] = types
    df_NB['email_body'] = contents
    df_NB['labelnames'] = labelnames
    df_NB['labels'] = labels
    the_count = CountVectorizer()
    Xtrain = the_count.fit_transform(df_NB['email_body'])
    Xtrain = Xtrain.toarray()
    km = KMeans(n_clusters=11,
                n_init='auto',
                random_state=0)
    clusters = km.fit_predict(Xtrain)
    df_NB['clusters'] = clusters
    return df_NB


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='List files in a directory')
    parser.add_argument('--directory', required=True,
                        help='Directory to list files in')
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"{args.directory} is not a valid directory")
        sys.exit(1)

    New_df = create_df_from_file(args.directory)
    prediction(New_df)
