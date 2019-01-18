# Classification of Bacteria with Fluorescence Spectra
In this project a series of machine-learning algorithms are used to try classify bacteria based on fluorescence spectra data. Two main series of experiments are done: the first on predicting the species of a given bacteria sample, and the second on predicting whether a given bacteria sample would test positive or negative in the gram stain test.

## Getting Started
1. If you are using conda you can create the conda environment used for the project with command:
    ```shell
    $ conda env create -f environment.yml
    ```

    Otherwise, make sure you python environment is set up with packages listed in the file `environment.yml`
2. Open the jupyter notebook `Data Preparation.ipynb` and run it.
3. Open and run one of the jupyter notebooks in the root directory or in the directory `other_experiments/`.

## Data
The data consists of fluorescence spectra readings from six different species of bacteria: Bacillus cereus, Listeria monocytogenes, Staphylococcus aureus, Salmonella enterica, Escherichia coli, and Pseudomonas aureginosa.
For each bacteria sample there are spectra readings for about 1043 different wavelengths of light and the three growth phases: lag, log, and stat. This means that for each bacteria sample there are 3 * 1043 data points. 
Furthermore, the spectra readings are generated with two different integration times, 16ms and 32ms. 
Integration time is the time spent gathering each fluorescence spectra reading. 
Shorter integration times are preferred.

When looking at a single growth phase, the data for just that growth phase is used as-is. 
However when using all growth phases, the bacteria samples that do not have data for all three growth phases are discarded. 
There are 47, 41, and 47 bacteria samples for the lag, log and stationary growth phases, respectively. 
There are 39 bacteria samples with data for all three growth phases, and thus the number of samples used when looking at all the growth phases together is also 39. 
The class balance within each of the subsets are analysed in the jupyter notebook `Data Analysis.ipynb`.

Within each bacteria species there are a number of replicates. A replicate is a copy of the bacteria species at possibly different concentration levels.

There are some large numbers in the dataset (some spectra readings exceed 25,000).
This poses a problem when training SVM models that use the linear kernel as the linear kernel is very slow for large values. 
For example, a SVM using the rbf kernel would take less than ~0.1 second to train while a SVM using the linear kernel could take up to ~16 minutes to train. 
To mitigate this effect I scaled the data into the interval [0.0, 1.0]. 
It should be noted that scaling is done 'globally', rather than scaling each feature individually as is done in the sklearn scaling libraries. 
This retains the relative scale between features. 
It is important to keep the relative scaling between features because technically all the features in this dataset are readings of the same feature. 
Ignoring relative scale and scaling on a per-feature basis worsens classification peformance.

There are two sets of labels used for classification: 
1.  the species of a given bacteria sample, which are:
    - Bacillus Cereus (bc)
    - Escherichia Coli (ec)
    - Listeria Monocytogenes (lm)
    - Pseudomonas Aureginosa (pa)
    - Staphylococcus Aureus (sa) 
    - Salmonella Enterica (se)

2. the 'gram-ness' of a given bacteria sample, i.e. whether the given bacteria would test positive or negative in the [gram stain test](https://en.wikipedia.org/wiki/Gram_stain). The groupings for the bacteria in the dataset are:
    - Gram-positive
        - Bacillus Cereus 
        - Listeria Monocytogenes
        - Staphylococcus Aureus 

    - Gram-negative
        - Escherichia Coli
        - Pseudomonas Aureginosa
        - Salmonella Enterica

Refer to the notebooks `Data Preparation.ipynb` and `Data Analysis.ipynb` for more details about the dataset and data. 

## Models
The models used in the experiments are:
1. Naive Bayes
2. SVM
3. RandomForest with Decision Stumps
4. RandomForest with Decision Trees
5. AdaBoost with Decision Stumps
6. AdaBoost with Decision Trees.
7. Convolutional Neural Network

Additionally, the parameters 'C', 'gamma', and 'kernel' are optimised for the SVM model via grid search. The score given for the SVM model is the model initialised with the best parameters found in this parameter search.
The decision stumps/trees used with AdaBoost and RandomForest are tested with a max tree depth of 1 (for decision stumps) and 3. 
RandomForest models are tested with 512 classifiers and AdaBoost with 256 classifiers.

The architecture of the convolutional neural network (CNN) model that is used is as follows:

|Layer (type)                |# Kernels |Kernel Shape|Output Shape  |# Params    |
|----------------------------|----------|------------|--------------|------------|
|conv1d (Conv1D)             |32        |3           |(N, 1041, 32) |320         |
|conv1d_1 (Conv1D)           |64        |3           |(N, 1039, 64) |6208        |
|global_average_pooling1d (Gl|-         |-           |(N, 64)       |0           |
|dense (Dense)               |          |            |(N, 6)        |390 (130*)  |
|                                                                                   |
|Total params: 6,918 (6,466*)                                                       |
|Trainable params: 6,918 (6,466*)                                                   |
|Non-trainable params: 0                                                            |
|* *If using a single growth phase.*                                                |

*N in the output shape is the batch size which changes between training and testing.*

A global average pooling layer is used instead of flattening. It significantly reduces the dimensionality of the output of convolutional layers and allows the output to be fed directly into the softmax layer. 
This way fully connected layers are not necessary and the number of parameters in the model are kept to a minimum. 
The CNN is trained with the RMSprop optimiser using the default settings.

## Methodology
For each experiment a sequence of tests which evaluate the performance of various models are run.  
An experiment is run for the entire dataset and again for each subset of the dataset, where a subset is simply the data from a single growth phase. 
In each experiment I run the same series of tests twice, once for each integration time. 

Each model is evaluated using both the original untransformed data, and a PCA transformed version of the data (with the expception of the CNN model, which is only tested on the original data). 
The number of components kept in the PCA transformed data is automatically set to the minimum number of components needed to retain 99% of the variance in the data. 

I utilise cross-validation to make sure the performance achieved by the models are not just achieved by random chance. 
Models are evaluated using repeated stratified k-fold cross validation which does 3-fold cross-validation 20 times. 
Scores are given for both the untransformed data and the PCA data and each score is the mean score over all the 60 indvidual folds +/- two standard deviations for the given model.

To help prevent overfitting the CNN I first train a single instance of the CNN model before performing cross-validation. 
I train the model for 1000 episodes and use early stopping so that training is stopped if the validation loss does not decrease for more than 10 epochs. 
I then round the epoch number that the training stopped on down the nearest 100. 

The random state is set to the same value across different modules (e.g. train_test_split, RandomForest initialisation) to ensure results can be reproduced consistently. 
I have not covered every case or have I even been very thorough with this, but I believe I have done enough to ensure that the data is always split in the same way and ensemble models like RandomForest have the same sub-models.

Brief summaries of the results are given with a table listing the top five configurations (in terms of both data and models) and a bar chart comparing classification scores across each configuration. 
The black lines on the bars in the bar chart indicate the +/- two standard deviation ranges.

The code for these experiments can be found in python package `experiments`.
Results can be found in the jupyter notebooks `Classification-Species.ipynb` and `Classification-Gramness.ipynb`. A summary of these results are given in the last section.

There are also a number of notebooks for ad-hoc experiments that were run separately. The code and results for these experiments can be found in the various notebooks under the directory `other_experiments/`. 
Most of these individual experiments are covered in the two jupyter notebooks mentioned above. The notebooks that may be of interest are the notebooks that deal with upsampling and other CNN architectures, i.e. notebooks that end with `-upsampling.ipynb` or notebooks starting with `CNN-`.

# Other Experiments
I also tried a few different experiments that were not covered by the two main notebooks, `Classification-Species.ipynb` and `Classification-Gramness.ipynb`, or included in final results. 

## Upsampling
The first thing that I tried was upsampling the data so that each class had the same number of instances. 
The dataset was first split into two portions: the first would be used for upsampling, and the second would be used as a test set and held out until the very end. 
Then the SMOTE algorithm would be used on the first portion of the data to create a training set. 
A set of the models would then be tested using the same repeated stratified k-fold cross-validation as described previously. 
The best performing model would be selected and have its performance on the held-out test set.   
The results were worse than classification without upsampling. 

## CNN with Separate Kernels for Each Growth Phase
In this experiment I try treating the data like an image and apply 2D convolution. The idea is that a 1D kernel applies the weights to every growth phase and the kernel may be too general and underfit. Treating the data as a 1043 x 3 image (1043 spectra readings by 3 growth phases) and using a 2D kernel may enable the model to learn kernels that are more specific to each growth phase and end up being more accurate.

The only implementation changes that this required was adding an extra dimension to the feature data, which changed the shape of each sample from 1043 x 3 to 1043 x 3 x 1, and using 2D convolutions with 3 x 3 kernels with 'same' padding.
In my experiments, this method did not provide any significant improvements in performance over using the same kernel for each growth phase. 
The results can be compared in the jupyter notebooks `CNN-Species.ipynb` and `CNN-Species-Separate-Kernels.ipynb` in the directory `other_experiments/`.

# Summary of Results
## Species Classification
Overall, none of models are able to produce good results when classifying bacteria species. The best classification accuracy was 59% (+/- 12%) using a SVM and the 32ms integration time data. 
Since, in the case of using all growth phase data, there are about 12 samples in the majority class out of a total of 39 samples, the best score a classifier could get by consistently guessing the majority class would be around 30%. 
So 59% is quite a bit better than random guessing, however it is still too unreliable for practical use. 

### Top Configurations
#### Lag Growth Phase
|  | integration_time |           classifier |  dataset | mean_score | score_std |
|--|------------------|----------------------|----------|------------|-----------|
|1 |             32ms |                  svm | original |       0.54 |      0.19 |
|2 |             32ms |                  svm |      pca |       0.54 |      0.20 |
|3 |             16ms |                  svm | original |       0.50 |      0.19 |
|4 |             32ms | random_forest_stumps | original |       0.49 |      0.08 |
|5 |             16ms |                  svm |      pca |       0.48 |      0.19 |

#### Log Growth Phase
|  | integration_time |           classifier |  dataset | mean_score | score_std |
|--|------------------|----------------------|----------|------------|-----------|
|1 |             16ms |        random_forest |      pca |       0.50 |      0.17 |
|2 |             16ms |                  svm |      pca |       0.50 |      0.21 |
|3 |             32ms |                  svm |      pca |       0.50 |      0.25 |
|4 |             16ms | random_forest_stumps | original |       0.48 |      0.11 |
|5 |             16ms |        random_forest | original |       0.48 |      0.18 |

#### Stat Growth Phase
|  | integration_time |           classifier |  dataset | mean_score | score_std |
|--|------------------|----------------------|----------|------------|-----------|
|1 |             32ms |                  svm | original |       0.59 |      0.19 |
|2 |             32ms |                  svm |      pca |       0.56 |      0.19 |
|3 |             32ms |        random_forest | original |       0.52 |      0.16 |
|4 |             16ms |        random_forest | original |       0.52 |      0.17 |
|5 |             32ms | random_forest_stumps | original |       0.51 |      0.15 |

#### All Growth Phases
|  | integration_time |           classifier |  dataset | mean_score | score_std |
|--|------------------|----------------------|----------|------------|-----------|
|1 |             32ms |                  svm | original |       0.59 |      0.11 |
|2 |             32ms |                  svm |      pca |       0.59 |      0.11 |
|3 |             16ms |                  svm | original |       0.59 |      0.14 |
|4 |             16ms |                  svm |      pca |       0.59 |      0.14 |
|5 |             32ms |        random_forest | original |       0.59 |      0.18 |

## Gram-ness Classification
Classifying gram-ness seems to alleviate the class imbalance problem encountered when classifying bacteria species.

Overall, the classification scores for gram-ness were much better than the scores for species classification. 
Many models were able to achieve 98% (+/- 2~7%) accuracy on the 16ms integration time log growth phase data. 

### Top Configurations
#### Lag Growth Phase
|  | integration_time |           classifier |  dataset | mean_score | score_std |
|--|------------------|----------------------|----------|------------|-----------|
|1 |             32ms |                  svm | original |       0.92 |      0.13 |
|2 |             32ms |                  svm |      pca |       0.92 |      0.13 |
|3 |             16ms |                  svm | original |       0.91 |      0.13 |
|4 |             16ms |                  svm |      pca |       0.89 |      0.13 |
|5 |             16ms | random_forest_stumps | original |       0.76 |      0.20 |

#### Log Growth Phase
|  | integration_time |           classifier |  dataset | mean_score | score_std |
|--|------------------|----------------------|----------|------------|-----------|
|1 |             16ms |                  svm |      pca |       0.98 |      0.07 |
|2 |             16ms |                  svm |      pca |       0.98 |      0.07 |
|3 |             16ms | random_forest_stumps | original |       0.98 |      0.07 |
|4 |             16ms |        random_forest | original |       0.98 |      0.07 |
|5 |             16ms |             adaboost | original |       0.98 |      0.07 |

#### Stat Growth Phase
|  | integration_time |           classifier |  dataset | mean_score | score_std |
|--|------------------|----------------------|----------|------------|-----------|
|1 |             32ms |                  svm | original |       0.93 |      0.10 |
|2 |             32ms |                  svm |      pca |       0.93 |      0.10 |
|3 |             16ms |                  svm | original |       0.93 |      0.11 |
|4 |             16ms |                  svm |      pca |       0.93 |      0.11 |
|5 |             32ms |                  cnn | original |       0.92 |      0.10 |

#### All Growth Phases
|  | integration_time |           classifier |  dataset | mean_score | score_std |
|--|------------------|----------------------|----------|------------|-----------|
|1 |             16ms |                  svm | original |       0.97 |      0.07 |
|2 |             16ms |                  svm |      pca |       0.97 |      0.07 |
|3 |             32ms |                  svm | original |       0.97 |      0.07 |
|4 |             32ms |                  svm |      pca |       0.97 |      0.07 |
|5 |             32ms |             adaboost |      pca |       0.95 |      0.08 |
