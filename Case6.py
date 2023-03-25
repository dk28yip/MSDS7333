import pandas as pd
import tensorflow as tf
import os
import sys
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report


def create_df_from_file(path):

    df = pd.read_csv(path + 'all_train.csv')
    df.rename(columns={'# label': 'response'}, inplace=True)
    return df


def predictions(df):

    # Scaling the data for train and test
    scaler = StandardScaler()
    X = df.loc[:, df.columns != 'response'].values
    y = df['response'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42)
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Creating the model using the best params from the Random Search
    num_neurons = 387
    input_shape = [28]
    learning_rate = 0.01

    final_model = tf.keras.models.Sequential()
    final_model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    final_model.add(tf.keras.layers.Dense(num_neurons, activation="relu"))
    final_model.add(tf.keras.layers.Dropout(.2, input_shape=(2,)))
    final_model.add(tf.keras.layers.Dense(num_neurons, activation="relu"))
    final_model.add(tf.keras.layers.Dropout(.2, input_shape=(2,)))
    final_model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.BinaryCrossentropy()
    final_model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    # Print Summary
    final_model.summary()

    early_stop = EarlyStopping(monitor='val_loss',
                               patience=3, min_delta=2e-4)
    final_model.fit(X_train, y_train, epochs=500,
                    validation_data=(X_test, y_test),
                    callbacks=[early_stop], batch_size=1000)

    # Print the model output
    tf.keras.utils.plot_model(final_model, to_file='model.png',
                              show_shapes=True, show_layer_names=True)

    # final prediction
    preds = final_model.predict(X_test)
    preds = (preds > .5).astype(int)
    print(classification_report(y_test, preds, zero_division=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='List files in a directory')
    parser.add_argument('--directory', required=True,
                        help='Directory to list files in')
    args = parser.parse_args()

    if not os.path.isdir(args.directory):
        print(f"{args.directory} is not a valid directory")
        sys.exit(1)

    New_df = create_df_from_file(args.directory)
    print(New_df)
    predictions(New_df)
