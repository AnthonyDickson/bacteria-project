"""The cnn experiment suite runs a series of classification tests on the bacteria
fluorescence spectra data using several classifiers from scikit-learn and a
convolutional neural network.

There are two main classes: CNNExperiment and CNNGramnessExperiment. They share
identical functionality except the former is classifying bacteria species and
the latter is classifying whether a given bacteria sample is gram positive or
gram negative.
"""

from time import time

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, GlobalAvgPool1D

from experiments.sklearn import timed, Experiment


class CNNExperiment(Experiment):
    """Runs a series of classification tests on the bacteria fluorescence
    dataset.

    Tests the following classifiers:
    - GaussianNB (Naive Bayes)
    - SVM
    - RandomForest with decision stumps
    - RandomForest with decision trees
    - AdaBoost with decision stumps
    - AdaBoost with decision trees
    - Convolutional Neural Network.
    """
    def __init__(self, growth_phases='all', n_jobs=-1, random_seed=42):
        self.X_cnn = {}

        super().__init__(growth_phases, n_jobs, random_seed)

        self.tests['cnn'] = \
            lambda integration_time: self.cnn_test(integration_time)

        _, self.W, self.C = self.X_cnn['16ms'].shape
        self.k = len(np.unique(self.y['16ms']))

    def _create_X_y(self):
        super()._create_X_y()

        for it in CNNExperiment.integration_times:
            channels = []

            if isinstance(self.growth_phases, str):
                channel = self.X[it].filter(regex=self.growth_phases)
                channel = channel.values

                channels.append(channel)
            else:
                for growth_phase in self.growth_phases:
                    channel = self.X[it].filter(regex=growth_phase)
                    channel = channel.values

                    channels.append(channel)

            channels = np.array(list(map(lambda c: np.expand_dims(c, axis=2),
                                         channels)))

            self.X_cnn[it] = np.concatenate(channels, axis=2)

    @timed
    def cnn_test(self, integration_time):
        """Run a classification test using a convolutional neural network (CNN)
        classifier.

        Arguments:
            integration_time: The integration time of the data to use for
                              classification. Must be one of '16ms' or '32ms'.

        Returns: A dictionary containing a list of scores for the 'original'
                 (not transformed) X data set and the pca data set.
        """
        status = 'Running CNN tests.'
        print('*' * len(status))
        print(status)
        print('*' * len(status))

        X = self.X_cnn[integration_time]
        y = self.y[integration_time].values.reshape(-1, 1)

        print('Finding suitable number of epochs.')
        n_epochs = self.optimal_cnn_epochs(X, y)
        print('Number of epochs is %d.' % n_epochs)

        n_splits = self.n_splits
        n_repeats = self.n_repeats
        n_total = n_splits * n_repeats

        i = 0
        start = time()

        score_history = []

        def print_progress(epoch, _):
            msg = 'Iteration %d/%d' % (i + 1, n_total)
            msg += ' - '
            msg += 'Epoch %d/%d' % (epoch, n_epochs)
            msg += ' - '

            total_epochs_done = epoch + i * n_epochs
            total_epochs = n_splits * n_repeats * n_epochs
            msg += 'Progress: %d/%d' % (total_epochs_done, total_epochs)
            msg += ' - '

            elapsed = time() - start
            mins = elapsed // 60
            secs = int(elapsed % 60)
            msg += 'Elapsed time: %02dm %02ds' % (mins, secs)

            n_whitespace = max(0, 80 - len(msg))
            msg += ' ' * n_whitespace

            print(msg, end='\r')

        epoch_progress_logger = tf.keras.callbacks.LambdaCallback(
            on_epoch_begin=print_progress
        )

        for train_idx, val_idx in self.cv.split(X, y):
            X_train_cv = X[train_idx]
            y_train_cv = y[train_idx]
            X_val_cv = X[val_idx]
            y_val_cv = y[val_idx]

            # Encoding is not done directly on y so that the k-fold splitter
            # doesn't throw an exception about incorrectly shaped labels array.
            ohe = OneHotEncoder(sparse=False)
            y_train_cv = ohe.fit_transform(y_train_cv)
            y_val_cv = ohe.fit_transform(y_val_cv)

            model = self.get_model()

            history = model.fit(X_train_cv, y_train_cv,
                                epochs=n_epochs,
                                validation_data=(X_val_cv, y_val_cv),
                                callbacks=[epoch_progress_logger],
                                verbose=0)

            score_history.append(history.history['val_acc'][-1])

            i += 1

        score_history = np.array(score_history)
        print('\nAccuracy: %.2f +/- %.2f' % (score_history.mean(),
                                             2 * score_history.std()))
        print('\nPCA Accuracy: N/A')

        return {'original': score_history, 'pca': np.array([])}

    def get_model(self):
        """Get an instance of CNN model.

        Returns: the compiled CNN model.
        """
        model = Sequential()
        model.add(Conv1D(32, kernel_size=3, activation='elu',
                         input_shape=(self.W, self.C)))
        model.add(Conv1D(64, kernel_size=3, activation='elu'))

        model.add(GlobalAvgPool1D())
        model.add(Dense(self.k, activation='softmax'))

        model.compile(loss=tf.keras.losses.categorical_crossentropy,
                      optimizer=tf.keras.optimizers.RMSprop(),
                      metrics=['accuracy'])

        return model

    def optimal_cnn_epochs(self, X, y):
        """Find the 'optimal' number of epochs to train the CNN model.

        Here optimal means the approximate number of epochs before the
        validation loss starts to increase again.

        Arguments:
            X: the feature data to train on.
            y: the labels to classify for.

        Returns: the number of epochs that is estimated to be optimal.
        """
        model = self.get_model()

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
                                                            test_size=0.2,
                                                            random_state=42)

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=10, )

        epoch_progress_logger = tf.keras.callbacks.LambdaCallback(
            on_epoch_begin=lambda epoch, _: print('Epoch %d/1000' % epoch,
                                                  end='\r')
        )

        ohe = OneHotEncoder(sparse=False)
        y_train_encoded = ohe.fit_transform(y_train)
        y_test_encoded = ohe.fit_transform(y_test)

        model.fit(X_train, y_train_encoded,
                  epochs=1000,
                  validation_data=(X_test, y_test_encoded),
                  callbacks=[early_stopping, epoch_progress_logger],
                  verbose=0)

        return CNNExperiment.n_train_epochs(early_stopping.stopped_epoch)

    @staticmethod
    def n_train_epochs(stopped_epoch, to_nearest=100, min_epochs=100):
        """Get the number of epochs the model should be trained for in cross-validation.

        Effectively rounds `stopped_epoch` down to the nearest multiple of `to_nearest`, but always returns a number
        at least as large as `min_epochs`.

        Arguments:
            stopped_epoch: The epoch that training was stopped on by the early stopping callback in keras.
            to_nearest: The multiple to round `stopped_epoch` down to.
            min_epochs: The minimum number to be returned.

        Returns: A multiple of `to_nearest` that is at least `min_epochs`.

        Examples:
        >>> CNNExperiment.n_train_epochs(99, to_nearest=100, min_epochs=100)
        100
        >>> CNNExperiment.n_train_epochs(101, to_nearest=100, min_epochs=100)
        100
        >>> CNNExperiment.n_train_epochs(243, to_nearest=100, min_epochs=100)
        200
        >>> CNNExperiment.n_train_epochs(34, to_nearest=10, min_epochs=20)
        30
        >>> CNNExperiment.n_train_epochs(243, to_nearest=20, min_epochs=100)
        240

        """
        assert stopped_epoch >= 0, 'Argument "stopped_epoch" must be >= 0.'

        result = (stopped_epoch // to_nearest) * to_nearest

        if result < min_epochs:
            return min_epochs
        else:
            return result


class CNNGramnessExperiment(CNNExperiment):
    """Runs a series of classification tests on the bacteria fluorescence
    dataset where the problem is simplified to classifying gramness
    (positive/negative.

    Tests the following classifiers:
    - GaussianNB (Naive Bayes)
    - SVM
    - RandomForest with decision stumps
    - RandomForest with decision trees
    - AdaBoost with decision stumps
    - AdaBoost with decision trees
    - Convolutional Neural Network.
    """

    def _create_X_y(self):
        """Create the X, X_pca and y data sets."""
        super()._create_X_y()

        for it in Experiment.integration_times:
            self.y[it] = self.X[it].reset_index()['gramness']


if __name__ == '__main__':
    e = Experiment()
    e.run()
