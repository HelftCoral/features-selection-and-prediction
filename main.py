"""
IMPORTANT NOTES


ACCURACY STATS HISTORY: (test_ratio = 0.2, n_perms = 100, norm_flag = 1, binning_flag = 0)
1. Original - 0.5765
2. After standardization calculation correction (previously was calculated separately on each label) - 0.7324
3. After correcting the line: sorted_features = sorted_features.dropna() - 0.7353
4. After algorithm correction at "select_non_correlated_features" - 0.7465 (fewer features taken caused to reduction
 at final results)
5. Correction of standardization process + correction of abs(R) in the feature selection process - 0.7429
6. After using the correct method of train_test_splitting with shuffling the data - 0.7468
7. Model's threshold from 0.5 to 0.4 - 0.7547 (specificity reduced, sensitivity increased)

MAIN ACTIONS TAKEN IN THE CODE:
26-27/05/2023 -
* Going through the code
* Creating the foundations of class df_preprocessing. The train_test split method was changed to be more memory-efficient -
 instead of creating and keeping N datasets ahead, in each iteration a new shuffled dataset is created.
* Correction of standardization process
* Correction of algorithm at the "select_non_correlated_features" function.
* Correction of syntax mistakes, most not harmful, some affected the correctness of the code (and the results).

28/05/2023 -
* Criterion_flag was removed
* Normalization options were added
* Deletion of code lines associated with is_categorical
* Addition of binning_flag allowing to choose if to apply binning or not (the binning is still in the old method)

29/05/2023 -
* Renaming some variables along the code to be more informing
* Further correction of standardization process - the mean and std is now calculated only for the train dataset. Then,
  the standardization applied on the test dataset using the mean and std of the train (we cannot use the "unseen"-test-dataset
  for the mean and std calculations).
* Since the standardization process is calculated for each permutation, a folder of normalized files is created in the code
  where the normalized data for each permutation is saved separately.
* In feature selection by the correlation there was a mistake, no "abs" applied on the R values.
* Finishing of creating class df_preprocessing and combining it in the code, achieving much lower accuracy scores.
  After deep analysis, it was noticed that the original function created non-shuffled test and train datasets - the data
  was sorted by the label, first the data with label 0 then the label 1. This caused to higher productivity of the model.

30/05/2023 -
* The code creates a folder of models' weights. The name of the folder is controlled by a hyperparameter.
* The code saves models weights.
* The model's coefficients are plotted with the choice of a flag variable (hyperparameter)
* Various score calculations were implemented.
* The total mean test scores of the model are plotted in the end of the process.
* The total test scores for each permuted dataset are plotted as table in the end of the process. In addition, this
  table is saved as csv file.
* The code creates a folder of the selected features names. For each permuted data, a list of the features names is
  saved as a different file.
* The code creates a folder of the scales for each permuted dataset. For each permuted data, its scaler is saved as a
  different file.

31/05/2023 -
* Added to the model a threshold control option on the predictions
* Added to the model a visualization option of the predictions
* Additional minor corrections

01/06/2023 -
* Rewrote the model's functions as a class
* Created (manually) a folder "Data CSV files", containing the original csv data-file and the other csv files for
  checking the code outputs.
* The correlation calculated between the features was pearson, I changed it to spearman.
* In the model's class, created basic foundations of applying LDA or PCA on the data for dimensionally reduction.
* Esthetics - code structuring, renaming variables names to be more logical, adding comments


FURTHER SUGGESTIONS:
1. To consider inputting a balanced dataset of labels 0 and 1 to the ML model. There are more label 0 than 1, thus the
   model is influenced more by the label 0 data and the mean specificity is higher than the mean sensitivity. This is
   important especially if we care more about true detections of positives.
2. The method of using pairwise correlation between the features to the labels should be considered to be changed. The
   method is problematic in many ways:
   * It can exclude features that their combination can produce a valuable meaning to the model.
   * Some high correlated featured that are passed are also correlated to each other (such ash PD_mean, PD_median).
     After that some of them are excluded in the pairwise correlation between features algorithm. In total, those
     features take the place of some other features that can be more valuable. I made an experiment where I created a
     new file "patientdata2 - fewer features" where I removed some of the columns that I saw were correlated to each other,
     and re-ran the code. The majority of the final scores improved a bit. I am sure further work could increase the scores
    even more.

    I would try to keep the first pairwise correlation only for the decision of the algorithm of the second pairwise
    correlation, but not for excluding features. Instead, PCA could be a good try for reducing dimensionality, or even
    better - LDA. When using PCA or LDA it is needed to analyze the components (which is also a free parameter in the
    model).

    Another approach is my previous suggestion of training ML models for the permuted datasets and to analyze the models'
    coefficients to see which features contribute more and less. This is a bit tricky since the whole data should be
    first standardized, and the standardization process suits mostly for data that is gaussian-distributed, which is also
    a question since maybe not all the features are gaussian-distributed. Therefore, the normalization process has to
    be properly suited to our data.
3. To make a deeper analysis of the features - which can be deleted ahead? How are they distributed? Are there any
   outliers in the data? Is it possible to get more useful features?
4. Playing with the hyperparameters, such as the correlation thresholds and with the model's parameters - the model's
   threshold, type of the model (maybe k-means could be a good try for achieving 2 groups? random forest could be also
   a good try since he deals pretty good with small datasets). Make a loop over the various parameters and find the
   best ones according to your mission.
5. The main() function doesn't have any input variables since all the hyperparameters are hard-coded. A cleaner
   implementation could be done with creating separate python files of the hyperparameters, of related functions and of
   the main() function.
"""

# ----------------------------------- Necessary Modules ------------------------------------
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import insert
from pandas import DataFrame
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy import stats
import os
from os import walk, path
import shutil
import pickle

warnings.filterwarnings('ignore')

# ----------------------------------- Hyper-parameters ------------------------------------
# Datasets creation
test_ratio = 0.2  # The proportion of the test vs. train data
n_perms = 100  # How many times to divide the data to test and train datasets
norm_flag = 2  # 0 - no normalization, 1 - standardization, 2 - min_max_normalization

# Pairwise spearman correlation calculation between the features to the H_label, for selecting the most correlated
# features to the labels column
binning_flag = 0  # 0 - no binning, 1 - binning
nbins = 10  # The amount of bins (if binning_flag=1)
R_threshold_perc = 90  # [0-100] The percentile of the corr values, that defines the corr value-threshold which above we keep the feature
P_threshold_val = 0.02  # [0-1] The corr P-value threshold, which below we keep the feature

# Pairwise spearman correlation calculation between the features, for selecting non-correlated features
threshold_of_pairwise_correlation = 0.85  # [0-1] The corr R-value, which bellow we keep both features (higher correlation --> keep only one of each pair)

pairwise_corr_method = 'spearman'
# threshold_of_the_most_selected_features = 1  # [0-N_perm] Which features will be presented in the final results (ONLY for plotting purposes)

# ML model
print_models_coeff_and_features_flag = 0  # 0 - no print, 1 - print
show_models_predictions_visualization_flag = 0  # 0 - no print, 1 - print
show_features_selection_visualization_flag = 0  # 0 - no print, 1 - print
model_prob_threshold = 0.48  # [0-1] controls the TP/TN performances of the ML model

# Files' names
main_file_dir = os.getcwd()
data_file_name = 'logistic_regression_features_males.csv'
data_files_folder = ''
normalized_file_name = 'Normalized_logistic_regression_features_males.csv'
perm_scores_file_name = "perm_scores_males.csv"
R_values_file_name = 'R_values_males.csv'
P_values_file_name = 'P_values_males.csv'
features_importance_file_name = 'Features_importance_after_pairwise_males.csv'
accuracies_data_file_name = 'males_accuracies_across_hyper_parameters'
features_selected_elaborate_file_name = 'features_selected_after_spearman_by_perms_males'

# Folders' names
normalized_files_folder_name = 'Normalized files'
models_weights_folder_name = 'Models weights'
selected_feature_names_folder_name = 'Selected feature names'
scales_folder_name = 'Permuted datasets scales'

header_prefix_str = 'H_'
target_column_name = 'label'
label_name = [header_prefix_str + target_column_name][0]
number_of_random_features = 2
random_features_name = 'random_feature'
coral_organizing_data_flag = True
save_flag = True
train_test_split_method = 0  # 1 is for dividing the data randomly to train and test
                             # 0 is for leave one out - goes through all possible permutation

# coral --> dictionary of the hyperparameters
# hyper_parameters_for_df_preprocessing = dict(main_file_dir=os.getcwd(),
#                                             data_files_folder='',
#                                             data_file_name='logistic_regression_features_females.csv')

# ----------------------------------- Functions ------------------------------------


"""
The class reads the main data file, splits it to train and test datasets and applies the desired normalization method.
"""


class DfPreprocessing:

    def __init__(self, main_file_dir, data_files_folder, data_file_name):
        # Read the csv files
        self.test = None
        self.train = None
        data_path = os.path.join(main_file_dir, data_files_folder, data_file_name)
        self.df_data = pd.read_csv(data_path)
        # self.df_data = pd.read_csv(f'{main_file_dir}\\{data_files_folder}\\{data_file_name}')
        # print(f"Table's shape: {self.df_data.shape}\n")
        # coral
    # END_OF_init

    def coral_organizing_data(self, number_of_header_columns):
        roi_names = ['Cortex', 'Corpus Callosum', 'Hippocampus', 'Thalamus', 'Amygdala', 'Medial Pre Frontal Cortex',
                     'Striatum']
        df_new = pd.DataFrame()
        first = True

        for roi_idx, roi_name in enumerate(roi_names):
            df_tmp = self.df_data[self.df_data.H_ROI_number == roi_idx].copy()
            v = np.array(df_tmp.columns)
            # adding roi_name for every col name from quantitative features stars position
            v[number_of_header_columns:] = roi_name + ' ' + v[number_of_header_columns:]
            df_tmp.columns = v
            df_tmp.drop('H_ROI_number', axis=1, inplace=True)
            if not first:
                df_new = df_new.merge(df_tmp, on=['H_label', 'H_animal number'], how='outer')
            else:
                df_new = df_tmp.copy()
                first = False
        self.df_data = df_new
        number_of_header_columns = number_of_header_columns - 1
        return number_of_header_columns

    def add_randomized_features(self, random_features_name, number_of_random_features):
        nRows = self.df_data.shape[0]
        random_feature = pd.DataFrame(index=range(nRows), columns=range(number_of_random_features))
        new_columns_names = []
        for idx in range(number_of_random_features):
            random_feature.iloc[:, idx] = np.random.rand(nRows, 1)
            new_columns_names.append(f'{random_features_name}_{str(idx)}')

        random_feature.columns = new_columns_names
        self.df_data = self.df_data.join(random_feature)
        return self.df_data

    def calculate_number_of_header_columns(self, header_prefix_str):
        # calculated how many header parameter there are in the data
        for col_number, col in enumerate(self.df_data.columns):
            if not col.startswith(header_prefix_str, 0, len(header_prefix_str)):
                number_of_header_columns = col_number
                break
        return number_of_header_columns

    """
    The function divides the total data to train and test datasets, that are used later in the normalized_data
    function.

    :param test_ratio: (float) The proportion of the test vs. train data.
    :param perm_index: (int) The current index out of the total n_perms. For each perm_index a different
                            divide_the_data_into_train_and_test and shuffling of the data is applied. Each unique 
                            splitting and shuffling remains the SAME by the control of random_state=perm_index.
    """

    def divide_the_data_into_train_and_test(self, test_ratio, perm_index, label_name, train_test_split_method):
        # Separate the data for each label separately
        df_0 = self.df_data[self.df_data[label_name] == 0].copy()
        df_1 = self.df_data[self.df_data[label_name] == 1].copy()

        # Split the data of 'H_label'=0 to train_test:
        features_0 = df_0.iloc[:, 1:]  # All columns except the H_label column
        label_0 = df_0[label_name]
        # Split the data of 'label'=1 to train_test:
        features_1 = df_1.iloc[:, 1:]  # All columns except the label column
        label_1 = df_1[label_name]

        if train_test_split_method:  # 1- is randomizing splitting method
            features_0_train, features_0_test, label_0_train, label_0_test = \
                train_test_split(features_0, label_0, test_size=test_ratio, random_state=perm_index, shuffle=True)

            features_1_train, features_1_test, label_1_train, label_1_test = \
                train_test_split(features_1, label_1, test_size=test_ratio, random_state=perm_index, shuffle=True)
        else:  # 0- is for leave one out - goes through all possible permutation
            perm_index_for_group_0 = int(np.floor(perm_index / 10))
            label_0_test = label_0.iloc[[perm_index_for_group_0]]
            label_0_train = label_0.drop(perm_index_for_group_0, axis=0)
            features_0_test = features_0.iloc[[perm_index_for_group_0]]
            features_0_train = features_0.drop(perm_index_for_group_0, axis=0)
            perm_index_for_group_1 = str(perm_index)
            perm_index_for_group_1 = int(perm_index_for_group_1[-1])
            loc_index_for_group_1 = perm_index_for_group_1 + len(features_0)
            label_1_test = label_1.iloc[[perm_index_for_group_1]]
            label_1_train = label_1.drop(loc_index_for_group_1, axis=0)
            features_1_test = features_1.iloc[[perm_index_for_group_1]]
            features_1_train = features_1.drop(loc_index_for_group_1, axis=0)

        # Combine the total data:
        features_train = pd.concat([features_0_train, features_1_train])
        label_train = pd.concat([label_0_train, label_1_train])
        features_test = pd.concat([features_0_test, features_1_test])
        label_test = pd.concat([label_0_test, label_1_test])

        self.train = features_train.copy()
        self.train.insert(0, label_train.name, label_train.values, allow_duplicates=True)
        self.test = features_test.copy()
        self.test.insert(0, label_test.name, label_test.values, allow_duplicates=True)

        # Shuffle the total data:
        self.train = self.train.sample(frac=1, random_state=perm_index).reset_index(drop=True)
        self.test = self.test.sample(frac=1, random_state=perm_index).reset_index(drop=True)

        # END_OF_divide_the_data_into_train_and_test

    """
    Standardization (norm_flag=1) - (x-mean(x))/std(x)
    Min_Max_Normalization (norm_flag=2) - transforms the data range of values from [x_min,x_max] to [new_min,new_max]

    Assumption -
    The first column of the main patients_data csv file is the label and the second column is patient_ID.

    :param norm_flag: (int) Defines the normalization method to apply on the data. 0 - no normalization,
                            1 - standardization, 2 - min_max_normalization.
    :param normalized_file_name: (str) The file name of the saved normalized data.
    :param perm_index: (int) The current index out of the total n_perms. It is used as a part of the scaler's and
                           normalized file's names.

    :return: self.train: (DataFrame), self.test (DataFrame) - the split normalized train and test datasets.

    Notice -
    If a normalization method is applied on the dataset, the scaler must be saved for a future reuse. The
    normalization is fitted on the train dataset, and using the achieved "learned" coefficients, the test dataset
    is then normalized with those learned coefficients.

    Deleted variables -
    * labels_indices

    Additions -
    * Changed the calculation so that the normalization's coefficients will be learned from the whole column of each
      feature in the train dataset (and not on each label separately as in the older code), and then the normalization
      is applied on both of the train and test datasets.
    * The saved file of the normalized data is now a combination of the train and test datasets with an additional
      column indicating the dataset's type
    * The scales for each permuted data are saved in a folder
    """

    def normalized_data(self, norm_flag, normalized_file_name, perm_index, number_of_header_columns, main_file_dir,
                        scales_folder_name, normalized_files_folder_name, number_of_random_features):

        # Extract the feature names (excluding the label and patient_ID columns)
        feature_names = self.df_data.columns.copy()[
                        number_of_header_columns:-number_of_random_features]

        # Normalize quantitative features
        if norm_flag == 0:
            pass

        # --------------------------------------------
        # Normalize quantitative values
        # --------------------------------------------
        # It's important to calculate the normalization parameters for the train and then apply on both train and test.
        # Similarly, if we have completely new data - it will be normalized in the same way.
        elif norm_flag == 1:
            # Define a scaler containing the mean and SD of each column (i.e., qMRI feature) in the train data.
            scaler = StandardScaler().fit(self.train.iloc[:, number_of_header_columns:-number_of_random_features])
        elif norm_flag == 2:
            # Define a scaler containing the min and max each column(i.e., qMRI feature) in the train data.
            scaler = MinMaxScaler(feature_range=(0, 1)).fit(
                self.train.iloc[:, number_of_header_columns:-number_of_random_features])
        else:
            raise Exception(f"Invalid norm_flag value.")

        self.train.iloc[:, number_of_header_columns:-number_of_random_features] = scaler.transform(
            self.train.iloc[:, number_of_header_columns:-number_of_random_features])
        self.test.iloc[:, number_of_header_columns:-number_of_random_features] = scaler.transform(
            self.test.iloc[:, number_of_header_columns:-number_of_random_features])

        # Save the scaling parameters to a file
        if save_flag:
            filename = f'{main_file_dir}\\{scales_folder_name}\\scaler_{perm_index}.pkl'
            if path.exists(scales_folder_name):
                shutil.rmtree(scales_folder_name)
            os.mkdir(scales_folder_name)
            pickle.dump(scaler, open(filename, 'wb'))

        # [NBE][Lauren] explain this:
        # Create a combined file of the train and test dataset to save as csv file
        if save_flag:
            normalized_train_df_tosave = self.train.copy()
            normalized_train_df_tosave.insert(0, 'dataset type', ['train'] * self.train.shape[0], allow_duplicates=True)
            normalized_test_df_tosave = self.test.copy()
            normalized_test_df_tosave.insert(0, 'dataset type', ['test'] * self.test.shape[0], allow_duplicates=True)
            normalized_df_tosave = pd.concat([normalized_train_df_tosave, normalized_test_df_tosave], ignore_index=True)

            norm_file_name = f'{main_file_dir}\\{normalized_files_folder_name}\\{perm_index}_{normalized_file_name}'
            normalized_df_tosave.to_csv(norm_file_name, index=False)

        # [NBE][Lauren] why permute
        print(f"The new permuted dataset is saved as '{perm_index}_{normalized_file_name}'.")

        return self.train, self.test

    # END_OF_normalized_data


"""
The function performs the extraction and selection of features from a dataset. For n_perms times of data permutations,
the function includes a process of data splitting, normalization, calculation of correlation values and selection of
features based on correlation thresholds. It provides as outputs the most relevant features using a defined threshold
(only for plotting purposes) and the final train and test datasets containing only the selected features as columns.

:param test_ratio:  (float) The proportion of the test vs. train data.
:param norm_flag: (int) Defines the normalization method to apply on the data.
:param n_perms: (int) How many times to divide the data to test and train datasets.
:param binning_flag: (int) Used in the pairwise spearman correlation calculation between the features to the label.
:param nbins: (int) Used in the pairwise spearman correlation calculation between the features to the label.
:param R_threshold_perc: (int/float) The percentile of the corr values, that defines the corr value-threshold which
above we keep the feature.
:param P_threshold_val: (float) The corr P-value threshold, which below we keep the feature.
:param threshold_of_pairwise_correlation: (float) The corr R-value, which belowe we keep both features (higher
correlation --> keep only one of each pair).
:param threshold_of_the_most_selected_features: (int) Which features will be presented in the final results (ONLY for
plotting purposes).
:param data_file_name: (str) Main data file name.
:param normalized_file_name: (str) The file name of the saved normalized data.
:param features_importance_file_name: (str) The file name of the saved total number of times each feature was selected.
:param R_values_file_name: (str) The file name of the saved calculated correlation values between each feature to the
'label' column.
:param P_values_file_name: (str) The file name of the saved calculated p-values between each feature to the 'label' column.
:return most_selected_features (DataFrame) -  The total number of times each feature was selected.
         series_of_train_data_after_features_selection (list(DataFrame,DataFrame, ..)) - The final train datasets containing
                                                                                 only the selected features as columns.
         series_of_test_data_after_features_selection (list(DataFrame,DataFrame, ..)) - The final test datasets containing
                                                                                only the selected features as columns.

Deleted variables -
* labels_indices (no use after the change of standar. calculation)
* filtered_R_P_values = filtered_R_P_values.dropna() - written wrongly, without inplace=True it doesn't do anything.
  It was put correctly inside the inner functions.
* criterion_flag - for criterion_flag=2 the functions were not fully implemented. Also there were unknown variables
  such as roi_names and data's column 'ROI'.

Corrections -
* The total data should first be split to train and test datasets and only then to apply normalization only to the
  train dataset.
* More efficient implementations, e.g. initializing a list for R and P values and only after the loop to concatenate
  the rows of data, instead of initializing dataframes and concatenating the rows of data in each iteration.
"""


def feature_selection(test_ratio,
                      norm_flag,
                      train_test_split_method,
                      random_features_name,
                      number_of_random_features,
                      n_perms,
                      binning_flag,
                      nbins,
                      R_threshold_perc,
                      P_threshold_val,
                      threshold_of_pairwise_correlation,
                      data_file_name,
                      normalized_file_name,
                      features_importance_file_name,
                      R_values_file_name,
                      P_values_file_name,
                      header_prefix_str,
                      label_name,
                      main_file_dir,
                      data_files_folder,
                      scales_folder_name,
                      normalized_files_folder_name,
                      pairwise_corr_method):
    print('Starting features extraction process\n')

    # Read the main csv file
    df_prep = DfPreprocessing(main_file_dir, data_files_folder, data_file_name)
    number_of_header_columns = df_prep.calculate_number_of_header_columns(header_prefix_str)
    if coral_organizing_data_flag:
        number_of_header_columns = df_prep.coral_organizing_data(number_of_header_columns)
    df_prep.add_randomized_features(random_features_name, number_of_random_features)
    series_of_train_data_after_features_selection = []
    series_of_test_data_after_features_selection = []
    R_values_list = []
    P_values_list = []

    # Split the data n_perms times into train and test datasets + apply normalization.
    for perm_index in range(n_perms):
        df_prep.divide_the_data_into_train_and_test(test_ratio, perm_index, label_name, train_test_split_method)
        train, test = df_prep.normalized_data(norm_flag, normalized_file_name, perm_index,
                                              number_of_header_columns, main_file_dir,
                                              scales_folder_name, normalized_files_folder_name,
                                              number_of_random_features)

        # --------------------------------------------------------------------
        # Stage I: Keep only qMRI features with high correlation to the label
        # --------------------------------------------------------------------
        # Find r and p values for the correlation between each feature and the 'label'
        R_values, P_values = calculate_R_and_P_values(train, nbins, binning_flag, number_of_header_columns, label_name)

        # Collect all R and P values for all permutations
        R_values_list.append(R_values)
        P_values_list.append(P_values)

        # Keep the R and P values only for  features with high correlation (r and p) to the 'label' column.
        # From here on we use the ABSOLUTE R-values
        R_values_absolute = abs(R_values)
        R_threshold_val = np.percentile(R_values_absolute, R_threshold_perc)
        if np.isnan(R_threshold_val):
            R_threshold_val = 0

        filtered_R_P_values = filter_features_by_R_and_P_values(R_values, P_values, R_threshold_val, P_threshold_val)

        # Extract from the Train data only qMRI features that passed the correlation to the label
        train_data_after_filtering_by_correlation_to_the_label = train[filtered_R_P_values.index]
        # Add the label to the Train data
        train_data_after_filtering_by_correlation_to_the_label.insert(0, label_name, train[label_name])

        test_data_after_filtering_by_correlation_to_the_label = test[filtered_R_P_values.index]
        test_data_after_filtering_by_correlation_to_the_label.insert(0, label_name, test[label_name])

        # -------------------------------------------------------------------------------------
        # Stage II: Filter out features that correlate with one another (pairwise correlation)
        #           (prevent usage of features that contain similar information)
        # -------------------------------------------------------------------------------------
        # final_features_names = select_non_correlated_features(train_data_after_filtering_by_correlation_to_the_label,
        #                                                       filtered_R_P_values,
        #                                                       threshold_of_pairwise_correlation,
        #                                                       pairwise_corr_method,
        #                                                       show_features_selection_visualization_flag)

        # # Append to the series train and test datasets a new pair containing only the currently selected features
        # series_of_train_data_after_features_selection.append(train[final_features_names])
        # series_of_test_data_after_features_selection.append(test[final_features_names])
        series_of_train_data_after_features_selection.append(train_data_after_filtering_by_correlation_to_the_label)
        series_of_test_data_after_features_selection.append(test_data_after_filtering_by_correlation_to_the_label)
    # END_OF_FORLOOP

    # ----------------------------------------------------------
    # Calculate the total times each feature was selected
    # (plotting / debugging purposes)
    # ----------------------------------------------------------
    all_features_names = train.columns[number_of_header_columns:]
    # Initialize empty dataframe that will contain all features and how many times each one was selected
    features_counter = pd.DataFrame({'features': all_features_names,
                                     'counter': np.zeros(len(all_features_names))})
    features_selected_by_iteration = pd.DataFrame(columns=all_features_names, index=range(n_perms))
    # Loop over all permutations of data and check which permutation contains which feature
    for perm_number, df_tmp in enumerate(series_of_train_data_after_features_selection):
        # features_counter['counter'] += features_counter['features'].isin(df_tmp.columns)
        for count, col in enumerate(features_counter['features']):
            if col in df_tmp.columns:
                features_counter.at[count, 'counter'] = int(features_counter.at[count, 'counter']) + 1
                features_selected_by_iteration.at[perm_number, col] = 1
    if save_flag:
        features_selected_by_iteration.to_csv(f'{features_selected_elaborate_file_name}.csv')

    # Keep only features that were selected more than "threshold_of_the_most_selected_features" times
    # and sort them according to the number of times they were chosen
    features_counter = features_counter.sort_values(by=['counter'], ascending=False, ignore_index=True)
    location_of_random_features = []
    for count, col in enumerate(features_counter['features']):
        if 'random_feature' in col:
            location_of_random_features.append(count)
    threshold_of_the_most_selected_features = features_counter['counter'][min(location_of_random_features)]
    most_selected_features = features_counter[features_counter['counter'] > threshold_of_the_most_selected_features]

    # -----------------------------------------
    # Save information to files
    # -----------------------------------------
    # Save selected features to a file
    if save_flag:
        most_selected_features.to_csv(features_importance_file_name, index=False)
        print(f"\nSelected features count saved as '{features_importance_file_name}'")

    # Save files of the R and p values
    # First , reshape (using Concatenate command) R and P values list of lists into 2D dataframes
    # Second, save to file
    R_values_df = pd.concat(R_values_list, ignore_index=True)
    P_values_df = pd.concat(P_values_list, ignore_index=True)

    if save_flag:
        R_values_df.to_csv(R_values_file_name, index=False)
        print(f"\nR values saved as '{R_values_file_name}'")

        P_values_df.to_csv(P_values_file_name, index=False)
        print(f"P values saved as '{P_values_file_name}'")

    print('\nFeatures selection - Done!\n')

    return most_selected_features, \
        series_of_train_data_after_features_selection, \
        series_of_test_data_after_features_selection


"""
For the train dataset, this function calculates the r and p values between each feature in the data and the label
column.

:param train: (DataFrame) The train dataset.
:param nbins: (int) Used in the pairwise spearman correlation calculation between the features to the label.
:param binning_flag: (int) Used in the pairwise spearman correlation calculation between the features to the label.
:return R_values: (DataFrame: (1,features)) - Correlation values between each feature to the labels column.
        P_values (DataFrame: (1,features)) - P-values between each feature to the labels column.

Deleted variables-
* criterion_flag

Notice -
If the method of calculating the pairwise correlation won't have the option of the binning, the whole related
functions could be written in a more efficient way, which will lead to much faster performances.
"""


def calculate_R_and_P_values(train, nbins, binning_flag, number_of_header_columns, label_name):
    # Extract the names of qMRI features from the train data (excluding 'label' and other header columns)
    header_cols = train.columns[:number_of_header_columns]
    cols = train.drop(header_cols, axis=1).columns
    # DEBUGGING:
    # print(f'calculate_R_and_P_values: Removing header columns {header_cols}') # coral: important for debug --> activated for debugging

    # Extract number of rows in the Train data
    rows_size = train.shape[0]

    # R and P values calculation for each column in comparison to the 'label' column
    R_values = pd.DataFrame(columns=cols)
    P_values = pd.DataFrame(columns=cols)

    for col in cols:
        r_value, p_value = calc_pairwise_spearman_corr_between_two_features(col,
                                                                            label_name,
                                                                            train,
                                                                            binning_flag,
                                                                            nbins,
                                                                            rows_size)
        R_values.loc[0, col] = r_value
        P_values.loc[0, col] = p_value

    return R_values, P_values


"""
For 2 given column vectors from a dataframe, the function calculates the pairwise spearman correlation between them.

:param x_name: (str) Feature's column name.
:param y_name: (str) Label's column name.
:param df: (DataFrame) The full dataframe containing the features and the labels.
:param nbins: (int) for visualization: If binning_flag chosen to be 1, it sets the number of bins in the binning algorithm.
:param binning_flag: (int) Decides whether to apply binning algorithm on the data before calculation the correlation.
:param rows_size: (int) Dataframe's rows size
:return spearman_corr: (float), p_value (float)

Deleted -
* Code lines associated with is_categorial
* Moved the rows_size calculation to the calculate_R_and_P_values since it is always constant.

Additions -
* Ensuring nbins>rows_size.
* binning_flag which allows to choose if to apply binning or not (the old method of binning remained).
"""


def calc_pairwise_spearman_corr_between_two_features(x_name, y_name, df, binning_flag, nbins, n_rows):
    df_small = df[[x_name, y_name]].copy()

    if binning_flag == 0:
        [r_value, p_value] = stats.spearmanr(df_small[x_name].values,
                                             df_small[y_name].values)
    elif binning_flag == 1:
        # Ensure that the number of nbins is smaller or equal to the data's number of rows
        if nbins > n_rows:
            raise Exception(
                f"nbins is larger than the data's number of rows ({n_rows}), please choose a smaller number.")

        # Sort is not needed for calculating correlation or for plotting. It is needed for binning (when used).
        # If inplace=True, perform the sort operation in the current dataframe (df_small).
        df_small.sort_values(x_name, inplace=True)

        x_vals = np.zeros(nbins)
        y_vals = np.zeros(nbins)
        for i in range(nbins):
            fr = int(np.floor(n_rows * (i + 0) / nbins))
            to = int(np.floor(n_rows * (i + 1) / nbins))
            x_vals[i] = np.mean(df_small[x_name].values[fr:to])
            y_vals[i] = np.mean(df_small[y_name].values[fr:to])
        [r_value, p_value] = stats.spearmanr(x_vals, y_vals)
    else:
        raise Exception(f"binning_flag values can be only 0 or 1.")

    return r_value, p_value
    # END_OF_FUNCTION


"""
Based on the calculated R and P values of features, the function keeps only the features who have a higher
correlation value than R_threshold_val and a lower p-value than P_threshold_val.

:param R_values: (DataFrame (1,features)): Correlation values between each feature to the labels column.
:param P_values: (DataFrame (1,features)): P-values between each feature to the labels column.
:param R_threshold_val: (float)
:param P_threshold_val: (float)
:return filtered_R_P_values: (DataFrame: (filtered_features,2))

Corrections -
* Syntax corrected.
* There was no abs on the R_values, which affected on performances. Now more features are saved.
"""


def filter_features_by_R_and_P_values(R_values, P_values, R_threshold_val, P_threshold_val):
    # Initialize a dataframe of R and P values and transpose to columns
    filtered_R_P_values = pd.concat([R_values, P_values]).T
    # add title to each column
    filtered_R_P_values.columns = ['R_values', 'P_values']

    # Drop rows containing nan values
    if filtered_R_P_values.isnull().values.any():
        filtered_R_P_values.dropna(axis=0, inplace=True)
        user_input_str = input('Do you want to continue even though there was NaN in the R/P values? yes/ no')
        if user_input_str == 'no':
            raise Exception(f'the data contained NaN')

    # Filter the dataframe based on threshold values
    filtered_R_P_values.where(abs(filtered_R_P_values.R_values) >= R_threshold_val, inplace=True)
    filtered_R_P_values.where(filtered_R_P_values.P_values <= P_threshold_val, inplace=True)
    filtered_R_P_values = filtered_R_P_values.dropna(axis=0)
    filtered_R_P_values.sort_values(by=['R_values'], ascending=False, inplace=True)

    return filtered_R_P_values


"""
The function calculates the correlation between all of the features. If a correlation between 2 features is higher
than the defined threshold, the function will keep the feature that has a higher correlation to the label column
(which was previously calculated).

:param train_data_after_filtering_by_correlation_to_the_label: (DataFrame) The train dataset containing the feature columns with
                                                         higher correlation to the label column.
:param filtered_R_P_values: (DataFrame (filtered_features,2)): The corr and p values between the filtered features
                                                               of the train dataset to the label column.
:param threshold_of_pairwise_correlation: (float) The threshold which bellow we keep both features. For a higher
                                                  correlation we keep only the one which has a higher correlation
                                                  to the label column.
:return columns_names: (Index: (num_of_selected_columns)) - The column names we keep in the input dataframe

Corrections -
The implementation was wrong in 2 aspects:
1. If i==2 and j==3 for example, and we got that the corr at feature i is less than the corr at feature j, there was
 no update on not taking feature i and break the inner loop, it just kept both of the features.
2. The final assertion of selected_features[j] = 0 in the previous code was wrong! sorted_features.iat[i, 1]
represents its p-value. If the p-value is smaller - there is a stronger correlation, and thus it should be kept!

Deleted variables -
* criterion_flag

Additions -
* train_df_features - for excluding the 'label' column (wasn't previously implemented and worked somehow in certain
conditions).
* The correlation calculated between the features was pearson, I changed it to spearman.
* The 2 last code lines were put here in the function instead of in the outer function.
"""


def select_non_correlated_features(train_data_after_filtering_by_correlation_to_the_label, filtered_R_P_values,
                                   threshold_of_pairwise_correlation, pairwise_corr_method, show_features_selection_visualization_flag):
    train_df_features = train_data_after_filtering_by_correlation_to_the_label.drop(label_name, axis=1)

    # Calculate pairwise correlation between each pair of features
    correlation_matrix = train_df_features.corr(method=pairwise_corr_method)

    if show_features_selection_visualization_flag:
        fig, ax = plt.subplots(figsize=(20, 17))
        _ = sns.heatmap(abs(correlation_matrix), vmin=0, vmax=1, linewidths=.3).set(title='pairwise correlation original')

    # Initialize an array of selected features (0 - features won't be selected, 1 - features will be selected)

    selected_features = np.ones(len(train_df_features.columns))
    if selected_features.shape[0] != correlation_matrix.shape[0]:
        # Sanity check - should not happen
        raise Exception(f"The number of rows of selected_features and correlation_matrix don't match.")

    # Zero all cells along the diagonal and the bottom half of the matrix (which is symmetric, so we don't need both sides)
    for idx in range(correlation_matrix.shape[0]):
        correlation_matrix.iloc[idx:, idx] = 0

    correlation_matrix_abs = abs(correlation_matrix)
    n_rows_temp = correlation_matrix_abs.shape[0]
    n_cols_temp = correlation_matrix_abs.shape[1]
    if n_rows_temp != n_cols_temp:  # sanity
        raise Exception(f'Pairwise correlation matrix is not square')

    max_iterations = int(((n_rows_temp ** 2) - n_rows_temp) / 2)

    for iter_idx in range(max_iterations):
        # Find the maximal correlation value in the matrix
        max_corr_val = 0
        for row_idx in range(n_rows_temp):
            for col_idx in range(row_idx + 1, n_rows_temp, 1):
                cur_corr_val = correlation_matrix_abs.at[
                    correlation_matrix_abs.index[row_idx], correlation_matrix_abs.index[col_idx]]
                if cur_corr_val > max_corr_val:
                    max_corr_val = cur_corr_val
                    max_corr_row = row_idx
                    max_corr_col = col_idx

        if max_corr_val >= threshold_of_pairwise_correlation:
            # iat--> Access a single value for a row/column pair by integer position
            #  at--> Access a single value for a row/column label pair.
            corr_R_val_row = filtered_R_P_values.iat[max_corr_row, 0]
            corr_R_val_col = filtered_R_P_values.iat[max_corr_col, 0]

            if corr_R_val_row > corr_R_val_col:
                delete_loc = max_corr_col
            else:
                delete_loc = max_corr_row

            # Reset to zero the correlation_matrix_abs row of the redundant feature
            correlation_matrix_abs.iloc[delete_loc, :] = 0
            # Reset to zero the correlation_matrix_abs column of the redundant feature
            correlation_matrix_abs.iloc[:, delete_loc] = 0
            # Reset to zero the selected features vector in place of the redundant feature
            selected_features[delete_loc] = 0

        else:
            break

    # Add 1 to selected_features to keep the label column in the datasets
    selected_columns_in_train_df = np.insert(selected_features, 0, 1).astype(dtype=bool)
    # The final column names
    columns_names = train_data_after_filtering_by_correlation_to_the_label.columns[selected_columns_in_train_df]

    return columns_names


"""

Notice -
* Analysing the model's weights is tricky. When its not normalized we cannot to infer directly from the weights
which features are more important, because the weights aren't scaled.
* LogisticRegressionCV is computationally heavier than LogisticRegression (and provides better performances).
If there are too many features, it might throw an error since he exceeds a runtime limitation.

Additions -
* Saving model weights
* Saving the selected features
* Various score calculations
* Threshold control on the predictions
* Visualization of the predictions

Deleted -
indA, indB (no use)

syntax corrected -
* model.fit(X_train, y_train)

"""

"""
:param x_train (DataFrame): Train dataset's features
:param x_test (DataFrame): Test dataset's features
:param y_train (Series): Train dataset's labels
:param y_test (Series): Test dataset's labels
:param perm_index (int): The current index out of the total n_perms.

    Notice -
    * Created in the comments the foundations of applying PCA or LDA on the data. Notice that the PCA converts
      the dataframes to ndarray and then the part of saving the features names will throw an error, so put it
      in comments if you would like to try running the code.
"""


class MLModel:

    def __init__(self, x_train, x_test, y_train, y_test, perm_index):

        self.y_test_predicted = None
        self.y_test_probabilities = None
        self.y_train_predicted = None
        self.y_train_probabilities = None
        self.model = None
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.perm_index = perm_index

        # # Using PCA for dimensionality reduction.
        # pca = PCA(n_components=8)  # reduce to 8 dimensions
        # self.x_train = pca.fit_transform(self.x_train)
        # self.x_test = pca.transform(self.x_test)
        #
        # # Analysis of the PCA components
        # plt.plot(pca.explained_variance_ratio_,'ks-',markerfacecolor='w',markersize=10)
        # plt.ylabel("Percent variance explained")
        # plt.xlabel("Component number")
        # plt.title(f'PCA screen plot - train')
        # plt.show()

        # # Using LDA for dimensionality reduction. N
        # LDA_model = LDA()
        # self.x_train = LDA_model.fit_transform(self.x_train, self.y_train)
        # self.x_test = LDA_model.transform(self.x_test)

    """
    Fitting the ML model on the train data and saving the model and the selected features to disk.
    """

    def ML_model_train(self):
        # Define and fit the model
        self.model = LogisticRegressionCV(
            solver='lbfgs')
        self.model.fit(self.x_train, self.y_train)

        # Save the model to disk
        if save_flag:
            filename = f'{main_file_dir}\\{models_weights_folder_name}\\model_{self.perm_index}.pkl'
            pickle.dump(self.model, open(filename, 'wb'))

            # Save the selected feature names as list:
            feature_names_file_name = f'{main_file_dir}\\{selected_feature_names_folder_name}\\features_names_{self.perm_index}.pkl'
            with open(feature_names_file_name, 'wb') as f:
                pickle.dump(self.x_train.columns.tolist(), f)
        """
    Perform the predictions that are controlled by the choice of a probability threshold.
    :param print_models_coeff_and_features_flag: (int) Choice of showing the model's coefficients.
    :param show_models_predictions_visualization_flag: (int) Choice of showing the visualized model's predictions.
    :param model_prob_threshold: (float) Controls the TP/TN performances of the ML model
        """

    def ML_model_make_prediction(self, model_prob_threshold, print_models_coeff_and_features_flag,
                                 show_models_predictions_visualization_flag):

        # Make predictions
        self.y_train_probabilities   = self.model.predict_proba(self.x_train)
        y_train_label1_probabilities = [p[1] for p in self.y_train_probabilities]  #[NBE] why for??? # probabilities of getting class "1"
        self.y_train_predicted       = [0 if p[1] < model_prob_threshold else 1 for p in self.y_train_probabilities]

        self.y_test_probabilities = self.model.predict_proba(self.x_test)
        y_test_label1_probabilities = [p[1] for p in self.y_test_probabilities]  # probabilities of getting class "1"
        self.y_test_predicted = [0 if p[1] < model_prob_threshold else 1 for p in self.y_test_probabilities]

        # Plot model's features and coefficients:
        if print_models_coeff_and_features_flag == 1:
            # [NBE] Understand what is coef_ (probably weight of each feature)
            #              and what is intercept_
            # AND ADD THIS INFORMATION TO THE DOCUMENTATION OF THIS CODE
            print(f"Model number {self.perm_index} chosen features:       {self.x_test.columns.tolist()}")
            print(f"Model number {self.perm_index} coefficients:          {self.model.coef_}")
            print(f"Model number {self.perm_index} intercept coefficient: {self.model.intercept_}\n")

        # Show the predictions and their actual label
        if show_models_predictions_visualization_flag == 1:
            # The probabilities of getting class "1" where the class actually is "0" (we aim this to be as low as possible)
            plt.figure(figsize=(7, 4))  # [NBE] change this code to print test and not train results
            plt.plot([y_test_label1_probabilities[index] for index in np.where(self.y_test == 0)[0].tolist()],
                     np.zeros(np.sum(self.y_test == 0)) + np.random.normal(0, 1.5, np.sum(self.y_test == 0)),
                     markersize=5, marker='x', linestyle="", alpha=0.6)
            # The probabilities of getting class "1" where the class actually is "1" (we aim this to be as high as possible)
            plt.plot([y_test_label1_probabilities[index] for index in np.where(self.y_test == 1)[0].tolist()],
                     np.zeros(np.sum(self.y_test == 1)) + np.random.normal(0, 1.5, np.sum(self.y_test == 1)),
                     markersize=5, marker='o', linestyle="", alpha=0.6)
            plt.plot([model_prob_threshold] * 50, np.linspace(-7.5, 7.5, num=50), linestyle="--", color='r', alpha=0.6)
            plt.xlim(0, 1)
            plt.ylim(-10, 10)
            plt.title('Test - Prediction Probabilities of Class 1 ')
            plt.legend(['datapoints of class 0', 'datapoints of class 1', 'Threshold'])
            plt.show()

    """
    Scores calculations:
    * Accuracy - (TP + TN)/(TP + TN + FP + FN)
    * Balanced accuracy - (Sensitivity + Specificity)/2
    * Mean model's prediction probabilities for each label
    * AUC - Higher the AUC, the better the model is capable of distinguishing between the classes.
    * Specificity (TNR) - TN/(TN + FP)
    * Sensitivity (TPR) - TP/(TP + FN)

    :return train_scores: (dict) - contains the scores calculations applied on the train dataset.
            test_scores (dict) - contains the scores calculations applied on the test dataset.
    """

    def ML_model_scores(self):
        # Accuracy score
        train_accuracy = accuracy_score(self.y_train, self.y_train_predicted)
        test_accuracy = accuracy_score(self.y_test, self.y_test_predicted)

        # Balanced accuracy score
        # [NBE] do we need this and if so when and do we need to set a hyper parameter flag to use / not use this code?
        train_balanced_accuracy = balanced_accuracy_score(self.y_train, self.y_train_predicted)
        test_balanced_accuracy = balanced_accuracy_score(self.y_test, self.y_test_predicted)

        # Mean model's prediction probabilities for each label
        train_mean_probability_per_label = self.y_train_probabilities.mean(axis=0)
        test_mean_probability_per_label = self.y_test_probabilities  #.mean(axis=0)
        test_mean_probability_per_label = np.reshape(test_mean_probability_per_label, (1, 4))

        # AUC score
        train_auc = roc_auc_score(self.y_train, self.y_train_probabilities[:, 1])
        test_auc = roc_auc_score(self.y_test, self.y_test_probabilities[:, 1])

        # Specificity + Sensitivity scores
        tn_train, fp_train, fn_train, tp_train = confusion_matrix(self.y_train, self.y_train_predicted).ravel()
        train_specificity = tn_train / (tn_train + fp_train)
        train_sensitivity = tp_train / (tp_train + fn_train)

        tn_test, fp_test, fn_test, tp_test = confusion_matrix(self.y_test, self.y_test_predicted).ravel()
        test_specificity = tn_test / (tn_test + fp_test)
        test_sensitivity = tp_test / (tp_test + fn_test)

        # Save the scores in dictionaries
        train_scores = {}
        test_scores = {}

        train_scores['y_train']                          = self.y_train
        train_scores['y_predicted']                      = self.y_train_predicted
        train_scores['train_accuracy']                   = train_accuracy
        train_scores['train_balanced_accuracy']          = train_balanced_accuracy
        train_scores['train_auc']                        = train_auc
        train_scores['train_specificity']                = train_specificity
        train_scores['train_sensitivity']                = train_sensitivity
        train_scores['train_mean_probability_per_label'] = train_mean_probability_per_label

        test_scores['y_test']                            = self.y_test
        test_scores['y_predicted']                       = self.y_test_predicted
        test_scores['test_accuracy']                     = test_accuracy
        test_scores['test_balanced_accuracy']            = test_balanced_accuracy
        test_scores['test_auc']                          = test_auc
        test_scores['test_specificity']                  = test_specificity
        test_scores['test_sensitivity']                  = test_sensitivity
        test_scores['test_mean_probability_per_label']   = test_mean_probability_per_label


        return train_scores, test_scores


"""
Calculate various scores and mean score based on the total permuted datasets.

:param series_of_train_data_after_features_selection: (list(DataFrame,DataFrame, ..)) The final train datasets containing
                                                                               only the selected features as columns.
:param series_of_test_data_after_features_selection: (list(DataFrame,DataFrame, ..)) The final test datasets containing
                                                                             only the selected features as columns.
:param N_perms: (int) How many times to divide the data to test and train datasets.


:param model_prob_threshold: (float) Controls the TP/TN performances of the ML model
:param perm_scores_file_name: (str) The file name of the saved scores of the permuted datasets.
:param print_models_coeff_and_features_flag: (int) Choice of showing the model's coefficients.
:param show_models_predictions_visualization_flag: (int) Choice of showing the visualized model's predictions.

:return test_perm_scores (DataFrame: (N_perms+2,scores_amount)) - The total scores for each permuted test dataset
                                                                   and 2 additional rows of the mean and std scores.


NOTICE -
The model includes score calculations for both test and train. It was implemented in this function to take only the
test scores and make further calculations on them.

Deleted variables-
indA, indB (no use)

Additions -
* Calculations
* The total scores for each permuted dataset are saved as csv file + the mean and std of the score (in the same file)

Corrected -
* If there are empty features in a permuted data, the accuracy for that was kept as in the initiallization (0). Now
it is nan value. The same as for the rest of the scores. That affected the shown accuracy if there were EMPTY features.

"""


def calc_model_mean_scores(series_of_train_data_after_features_selection,
                           series_of_test_data_after_features_selection,
                           model_classification_threshold,
                           ML_Model_scores_fn,
                           print_models_coeff_and_features_flag,
                           show_models_predictions_visualization_flag):
    print('Calculating model\'s scores...\n')
    N_perms = len(series_of_train_data_after_features_selection)

    # Initialize score variables sized for the total N_perms
    test_accuracy_per_perm = np.zeros(N_perms)
    test_balanced_accuracy_per_perm = np.zeros(N_perms)
    test_auc_per_perm = np.zeros(N_perms)
    test_specificity_per_perm = np.zeros(N_perms)
    test_sensitivity_per_perm = np.zeros(N_perms)
    total_test_mean_probability_per_label = []  # np.zeros((N_perms, 2))
    num_of_features_per_perm = np.zeros(N_perms)
    test_y_true = []  # np.zeros(N_perms)
    test_y_predicted = []  # np.zeros(N_perms)

    for perm_index in range(N_perms):
        # Separate the train/test datasets to features (x) and labels (y)
        y_train = series_of_train_data_after_features_selection[perm_index][label_name].copy()
        y_test = series_of_test_data_after_features_selection[perm_index][label_name].copy()

        x_train = series_of_train_data_after_features_selection[perm_index].copy()
        x_test = series_of_test_data_after_features_selection[perm_index].copy()
        x_train.pop(label_name)
        x_test.pop(label_name)

        # Check if the permuted data has no features at all
        if x_train.empty or x_test.empty:
            print('calc_model_mean_scores: EMPTY train or test', perm_index)
            test_accuracy_per_perm[perm_index] = np.nan
            test_balanced_accuracy_per_perm[perm_index] = np.nan
            test_auc_per_perm[perm_index] = np.nan
            test_specificity_per_perm[perm_index] = np.nan
            test_sensitivity_per_perm[perm_index] = np.nan
            total_test_mean_probability_per_label[perm_index, :] = np.array([np.nan, np.nan])
            continue

        # Assign the number of total features in the dataset
        num_of_features_per_perm[perm_index] = len(x_train.columns.tolist())

        # Get the model's scores
        model = MLModel(x_train, x_test, y_train, y_test, perm_index)
        model.ML_model_train()
        model.ML_model_make_prediction(model_classification_threshold,
                                       print_models_coeff_and_features_flag,
                                       show_models_predictions_visualization_flag)
        #       _, test_scores = model.ML_model_scores()
        train_scores, test_scores = model.ML_model_scores()

        # Assign the test scores in the initialized variables
        test_accuracy_per_perm[perm_index] = test_scores['test_accuracy']
        test_balanced_accuracy_per_perm[perm_index] = test_scores['test_balanced_accuracy']
        test_auc_per_perm[perm_index] = test_scores['test_auc']
        test_specificity_per_perm[perm_index] = test_scores['test_specificity']
        test_sensitivity_per_perm[perm_index] = test_scores['test_sensitivity']
        total_test_mean_probability_per_label.append(test_scores['test_mean_probability_per_label'])
        test_y_true.append(test_scores['y_test'])
        test_y_predicted.append(test_scores['y_predicted'])

    # Create a table of test scores of all the permuted datasets
    test_perm_scores_df = pd.DataFrame()
    test_perm_scores_df['num_of_selected_features'] = num_of_features_per_perm
    test_perm_scores_df['test_accuracy'] = test_accuracy_per_perm
    test_perm_scores_df['test_balanced_accuracy'] = test_balanced_accuracy_per_perm
    test_perm_scores_df['test_auc'] = test_auc_per_perm
    test_perm_scores_df['test_specificity'] = test_specificity_per_perm
    test_perm_scores_df['test_sensitivity'] = test_sensitivity_per_perm

    results_proba = np.array(total_test_mean_probability_per_label)
    test_results_probability = np.squeeze(results_proba)
    test_y_true = np.squeeze(np.array(test_y_true))
    test_y_predicted = np.squeeze(np.array(test_y_predicted))


    # Calculate test mean scores of all the models and add to test_perm_scores
    # For each column / field in the final score dataframe calculate the mean and SD values
    # and add them as two last rows
    test_perm_scores_df = test_perm_scores_df.append(test_perm_scores_df.mean(axis=0, skipna=True), ignore_index=True)
    test_perm_scores_df = test_perm_scores_df.append(test_perm_scores_df.iloc[:-1].std(axis=0, skipna=True),
                                                     ignore_index=True)

    # Change index names of the mean and std calculations to 'Mean' and 'SD'
    test_perm_scores_df.rename(index={test_perm_scores_df.shape[0] - 2: 'Mean',
                                      test_perm_scores_df.shape[0] - 1: 'Std'}, inplace=True)

    # Save test_perm_scores as file
    if save_flag:
        ML_Model_scores_path = f'{main_file_dir}\\Test_{ML_Model_scores_fn}'
        test_perm_scores_df.to_csv(ML_Model_scores_path, index=True)

    return test_perm_scores_df, test_results_probability, test_y_true, test_y_predicted


"""
The function prints the final scores and results.

:param test_perm_scores: (DataFrame (N_perms+2,scores_amount)) - The total scores for each permuted test dataset
                                                                 and 2 additional rows of the mean and std scores.
:param most_selected_features: (DataFrame) Data frame with 2 columns (feature name, number of times this features was selected).

Corrections -
* Removed the obvious variable of mean probability for the scores 0 and 1 (which is always 0.5).
* Added 2 tables of test scores
"""


def print_results(test_perm_scores, most_selected_features):
    # Table of test scores for each permuted dataset
    print("The model's test scores for each permuted dataset:\n")
    pd.set_option('display.max_columns', None)
    print(test_perm_scores.iloc[:-2])
    pd.reset_option('display.max_columns')

    # Table of test mean scores of the permuted datasets
    print("\nThe model's test mean scores:\n")
    print(test_perm_scores.iloc[-2:].T)

    if len(most_selected_features) > 0:
        # Features Importance Graph
        sns.set_theme(style="white", palette="copper")
        sns.set(font_scale=1)
        g = sns.catplot(data=most_selected_features, kind="bar", x="counter", y="features",
                        palette="copper", height=25).set(title='Features Importance')
        g.despine(left=True)
        g.set_axis_labels("The amount of times each feature was selected", "Features")
        plt.show()
    else:
        print('\nTHERE WERE NO SELECTED FEATURES AT ALL!')


"""
Main Run Command.
The function doesn't have any input variables since all the hyperparameters are hard-coded.

Deleted variable -
criterion_flag
"""


def main():
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Coral's benchmark
    # R_threshold_perc_arr = range(85, 100, 5)
    # model_prob_threshold_arr = [x / 100 for x in range(46, 56, 2)]
    # P_threshold_val_arr = 0.02
    # threshold_of_pairwise_correlation_arr = [x / 100 for x in range(95, 60, -5)]
    # accuracy_results = pd.DataFrame(columns =['model_prob_threshold', 'accuracy'])
         # columns=['R_threshold_perc', 'P_threshold_val', 'threshold_of_pairwise_correlation', 'accuracy'])
    # ind = 0
    #
    # for model_prob_threshold in model_prob_threshold_arr:
    #     for threshold_of_pairwise_correlation in threshold_of_pairwise_correlation_arr:

            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        np.random.seed(0)

        # Create desired folders. If the folder already exists, the code deletes it and creates a new one.
        folders_names = [normalized_files_folder_name, models_weights_folder_name,
                         selected_feature_names_folder_name]
        for (dirpath, dirnames, filenames) in walk(main_file_dir):
            for i in range(len(folders_names)):
                folder_dir = f'{main_file_dir}\\{folders_names[i]}'
                if path.exists(folder_dir):
                    shutil.rmtree(folder_dir)
                os.mkdir(folder_dir)
            break

        # Create permuted datasets with selected features
        most_selected_features, series_of_train_data_after_features_selection, \
            series_of_test_data_after_features_selection = feature_selection(test_ratio,
                                                                             norm_flag,
                                                                             train_test_split_method,
                                                                             random_features_name,
                                                                             number_of_random_features,
                                                                             n_perms,
                                                                             binning_flag,
                                                                             nbins,
                                                                             R_threshold_perc,
                                                                             P_threshold_val,
                                                                             threshold_of_pairwise_correlation,
                                                                             data_file_name,
                                                                             normalized_file_name,
                                                                             features_importance_file_name,
                                                                             R_values_file_name,
                                                                             P_values_file_name,
                                                                             header_prefix_str,
                                                                             label_name,
                                                                             main_file_dir,
                                                                             data_files_folder,
                                                                             scales_folder_name,
                                                                             normalized_files_folder_name,
                                                                             pairwise_corr_method)

        # Calculate the permuted datasets' models' scores
        test_perm_scores_df, test_results_probability, test_y_true, test_y_predicted = calc_model_mean_scores(series_of_train_data_after_features_selection,
                                                                                                              series_of_test_data_after_features_selection,
                                                                                                              model_prob_threshold,
                                                                                                              perm_scores_file_name,
                                                                                                              print_models_coeff_and_features_flag,
                                                                                                              show_models_predictions_visualization_flag)

        # Show the final results
        print_results(test_perm_scores_df, most_selected_features)

            # ------------------------------------------------------------------
            # Coral's benchmark
            # ------------------------------------------------------------------
            # accuracy_results.at[ind, 'R_threshold_perc'] = R_threshold_perc
            # accuracy_results.at[ind, 'P_threshold_val'] = P_threshold_val
            # accuracy_results.at[ind, 'threshold_of_pairwise_correlation'] = threshold_of_pairwise_correlation
        # accuracy_results.at[ind, 'model_prob_threshold'] = model_prob_threshold
        # accuracy_results.at[ind, 'accuracy'] = test_perm_scores_df['test_accuracy']['Mean']
        # print(f'done with {ind} interation')
        # ind += 1
        # print(test_perm_scores_df)
    # if save_flag:
        # accuracy_results.to_csv(f'{main_file_dir}\\{data_files_folder}\\model_prob_threshold_females_hyperparms.csv')


if __name__ == '__main__':
    main()
