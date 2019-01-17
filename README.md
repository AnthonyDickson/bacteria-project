# Classification of Bacteria with Fluorescence Spectra
In this project a series of machine-learning algorithms are used to try classify bacteria based on fluorescence spectra data.

## Getting Started
1. If you are using conda you can create the conda environment used for the project with command:
    ```shell
    $ conda env create -f environment.yml
    ```

    Otherwise, make sure you python environment is set up with packages listed in the file `environment.yml`
2. Open the jupyter notebook `Data Preparation.ipynb` and run it.
3. Open and run one of the jupyter notebooks in the root directory or in the directory `individual/`.

## Data
The data consists of fluorescence spectra readings from six different species of bacteria: Bacillus cereus, Listeria monocytogenes, Staphylococcus aureus, Salmonella enterica, Escherichia coli, and Pseudomonas aureginosa.
For each bacteria sample there are spectra readings for about 1043 different wavelengths of light and the three growth phases: lag, log, and stat (stationary). This means that for each bacteria sample there are 3 * 1304 data points. Furthermore, the spectra readings are generated with two different integration times (time spent gathering the spectra reading), 16ms and 32ms. 

When looking at a single growth phase, the data is used as-is. However when using all growth phases, the bacteria samples that do not have data for all three growth phases are discarded. There are 47, 41, and 47 bacteria samples for the lag, log and stationary growth phases, respectively. There are 39 bacteria samples with data for all three growth phases. The class balance within each of the subsets are analysed in the jupyter notebook `Data Analysis.ipynb`.

There are some large numbers in the dataset (some spectra readings exceed 25,000). This poses a problem when training SVM models that use the linear kernel as the linear kernel is very slow for large values. For example, a SVM using the rbf kernel would take less than ~0.1 second to train while a SVM using could take up to ~16 minutes to train. To mitigate this effect I scaled the data into the interval [0.0, 1.0]. However, scaling is done 'globally' as opposed to scaling each feature individually as is done in the sklearn scaling libraries. This retains the relative scale between features. It is important to keep the relative scaling between features because technically all the features in this dataset are readings of the same feature. Ignoring relative scale and scaling on a per-feature basis worsens classification peformance.

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
The classifiers used in the experiments are:
1. Naive Bayes
2. SVM
3. RandomForest with Decision Stumps
4. RandomForest with Decision Trees
5. AdaBoost with Decision Stumps
6. AdaBoost with Decision Trees.
7. Convolutional Neural Network

Additionally, the parameters 'C', 'gamma', and 'kernel' are optimised for the SVM model via grid search. The score given for the SVM model is the model initialised with the best parameters found in this parameter search.
The decision stumps/trees used with AdaBoost and RandomForest are tested with a max tree depth of 1 (for decision stumps) and 3. RandomForest models are tested with 512 classifiers and AdaBoost with 256 classifiers.

The architecture of the convolutional neural network (CNN) is as follows:

|Layer (type)                |Output Shape             |Param #     |
|----------------------------|-------------------------|------------|
|conv1d (Conv1D)             |(None, 1041, 32)         |320         |
|conv1d_1 (Conv1D)           |(None, 1039, 64)         |6208        |
|global_average_pooling1d (Gl|(None, 64)               |0           |
|dense (Dense)               |(None, 6)                |390 (130*)  |
|                                                                   |
|Total params: 6,918 (6,466*)                                       |
|Trainable params: 6,918 (6,466*)                                   |
|Non-trainable params: 0                                            |
|* *If using a single growth phase.*

A global average pooling layer is used instead of flattening and it significantly reduces the dimensionality of the input going into the softmax layer. This way fully connected layers are not necessary and the number of parameters in the model are kept to a minimum. 
The CNN is trained with the RMSprop optimiser using the default settings.

## Methodology
An experiment refers to a sequence of tests which evaluate the performance of various models. An experiment is run for the entire dataset and again for each subset of the dataset, where a subset is simply the data from a single growth phase. In each experiment I run the same series of tests twice, once for each integration time. 

Each model is evaluated using both the original, untransformed data and a PCA transformed version of the data (with the expception of the CNN model, which is only tested on the original data). The number of components kept in the PCA transformed data is automatically set to the minimum number of components needed to explain 99% of the variance in the data. 

I utilise cross-validation to make sure the performance achieved by the models are not achieved by just random chance. Models are evaluated using repeated stratified k-fold cross validation which does 3-fold cross-validation 20 times. The scores given for both the untransformed data and the PCA data are mean score over all the 60 indvidual folds +/- two standard deviations.

To help prevent overfitting in CNN I first train a single instance of the CNN model before performing cross-validation. I train the model for 1000 episodes and use early stopping so that training is stopped if the validation loss does not decrease for more than 10 epochs. I then round the epoch that was stopped on down the nearest 100. 

The random state is set to the same value across different modules (e.g. train_test_split, RandomForest initialisation) to ensure results can be reproduced consistently. I have not covered every case or have I even been very thorough with this, but I believe I have done enough to ensure that the data is always split in the same way and ensemble models like RandomForest have the same sub-models.

Brief summaries of the results are given with a table listing the top three configurations (in terms of both data and models) and a bar chart comparing classification scores across each configuration. The black lines on the bars in the bar chart indicate the +/- two standard deviation ranges.

The code for these experiments can be found in python module `experiments`.
Results can be found in the jupyter notebooks `Classification-Species.ipynb` and `Classification-Gramness.ipynb`.

There are also a number of notebooks where experiments were run individually. The code and results for these experiments can be found in the various notebooks under the directory `individual/`.

# Summary of Results
## Species Classification
Overall, none of models are able to produce good results when classifying bacteria species. The best classification accuracy was 59% (+/- 12%) using a CNN. 
Since, in the case of using all growth phase data, there are about 12 samples in the majority class out of a total of 39 samples, the best score a classifier could get by consistently guessing the majority class would be around 30%. So 59% is quite a bit better than random guessing, however it is still too unreliable for practical use. 

## Gram-ness Classification
Classifying gram-ness seems to alleviate the class imbalance problem encountered when classifying bacteria species.

Overall, the classification scores for gram-ness were much better than the scores for species classification. Both the SVM and CNN models were able to achieve 98% (+/- 2~7%) accuracy on the log growth phase data. 
