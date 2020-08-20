from numpy import count_nonzero, logical_and, logical_or, concatenate, mean, array_split, poly1d, polyfit, array
from numpy.random import permutation
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt


SVM_DEFAULT_DEGREE = 3
SVM_DEFAULT_GAMMA = 'auto'
SVM_DEFAULT_C = 1.0
ALPHA = 1.5


def prepare_data(data, labels, max_count=None, train_ratio=0.8):
    """
    :param data: a numpy array with the features dataset
    :param labels:  a numpy array with the labels
    :param max_count: max amout of samples to work on. can be used for testing
    :param train_ratio: ratio of samples used for train
    :return: train_data: a numpy array with the features dataset - for train
             train_labels: a numpy array with the labels - for train
             test_data: a numpy array with the features dataset - for test
             test_labels: a numpy array with the features dataset - for test
    """
    if max_count:
        data = data[:max_count]
        labels = labels[:max_count]

    ###########################################################################
    # TODO: Implement the function                                            #
    ###########################################################################

    data_plus_labels = concatenate((data, array([labels]).T), axis=1)
    shuffled_data_plus_labels = permutation(data_plus_labels)

    num_of_train_instunses = int(shuffled_data_plus_labels.shape[0] * train_ratio)
    num_of_test_instunses = shuffled_data_plus_labels.shape[0] - num_of_train_instunses
    
    train_data = array(shuffled_data_plus_labels[0:num_of_train_instunses,0:-1])
    train_labels = array(shuffled_data_plus_labels[0:num_of_train_instunses,-1])
    test_data = array(shuffled_data_plus_labels[num_of_train_instunses:,0:-1])
    test_labels = array(shuffled_data_plus_labels[num_of_train_instunses:,-1])
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return train_data, train_labels, test_data, test_labels


def get_stats(prediction, labels):
    """
    :param prediction: a numpy array with the prediction of the model
    :param labels: a numpy array with the target values (labels)
    :return: tpr: true positive rate
             fpr: false positive rate
             accuracy: accuracy of the model given the predictions
    """

    # ###########################################################################
    # # TODO: Implement the function                                            #
    # ###########################################################################
    
    positive_num = count_nonzero(labels)
    negative_num = labels.shape[0] - positive_num

    tp = 0
    fp = 0
    for i in range(labels.shape[0]):
        if (prediction[i] == 1 and labels[i] == 1):
            tp += 1
        if (prediction[i] == 1 and labels[i] == 0):
            fp +=1
    
    tn = negative_num - fp
    fn = positive_num - tp

    tpr = tp / (tp + fn)
    fpr = fp / (tn + fp)
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return tpr, fpr, accuracy


def get_k_fold_stats(folds_array, labels_array, clf):
    """
    :param folds_array: a k-folds arrays based on a dataset with M features and N samples
    :param labels_array: a k-folds labels array based on the same dataset
    :param clf: the configured SVC learner
    :return: mean(tpr), mean(fpr), mean(accuracy) - means across all folds
    """
    tpr = []
    fpr = []
    accuracy = []

    ###########################################################################
    # TODO: Implement the function                                            #
    ###########################################################################

    curr_train_data = folds_array
    curr_train_label = labels_array
    
    for i in range(len(folds_array)):
        curr_test_data = curr_train_data.pop(0)
        curr_test_label = curr_train_label.pop(0)
        
        conct_train_data = concatenate(curr_train_data)
        conct_train_label = concatenate(curr_train_label)
        
        clf.fit(conct_train_data, conct_train_label)
        curr_tpr, curr_fpr, curr_accuracy = get_stats(clf.predict(curr_test_data), curr_test_label)
        
        tpr.append(curr_tpr)
        fpr.append(curr_fpr)
        accuracy.append(curr_accuracy)
        
        curr_train_data.append(curr_test_data)
        curr_train_label.append(curr_test_label)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return mean(tpr), mean(fpr), mean(accuracy)


def compare_svms(data_array,
                 labels_array,
                 folds_count,
                 kernels_list=('poly', 'poly', 'poly', 'rbf', 'rbf', 'rbf',),
                 kernel_params=({'degree': 2}, {'degree': 3}, {'degree': 4}, {'gamma': 0.005}, {'gamma': 0.05}, {'gamma': 0.5},)):
    """
    :param data_array: a numpy array with the features dataset
    :param labels_array: a numpy array with the labels
    :param folds_count: number of cross-validation folds
    :param kernels_list: a list of strings defining the SVM kernels
    :param kernel_params: a dictionary with kernel parameters - degree, gamma, c
    :return: svm_df: a dataframe containing the results as described below
    """
    svm_df = pd.DataFrame()
    svm_df['kernel'] = kernels_list
    svm_df['kernel_params'] = kernel_params
    svm_df['tpr'] = None
    svm_df['fpr'] = None
    svm_df['accuracy'] = None

    tpr = []
    fpr = []
    accuracy = []
    ###########################################################################
    # TODO: Implement the function                                            #
    ###########################################################################

    fold_data_array = array_split(data_array, folds_count)
    fold_labels_array = array_split(labels_array, folds_count)

    clf = SVC()
    for i in range(len(kernels_list)):

        ker_type = kernels_list[i]
        ker_param = kernel_params[i]

        clf.set_params(**{'kernel' :ker_type, 'C' : SVM_DEFAULT_C, 'gamma' : SVM_DEFAULT_GAMMA, 'degree' : SVM_DEFAULT_DEGREE})
        clf.set_params(**ker_param)
        
        mean_tpr, mean_fpr, mean_accuracy = get_k_fold_stats(fold_data_array, fold_labels_array, clf)

        tpr.append(mean_tpr)
        fpr.append(mean_fpr)
        accuracy.append(mean_accuracy)
    
    svm_df['tpr'] = tpr
    svm_df['fpr'] = fpr
    svm_df['accuracy'] = accuracy
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return svm_df


def get_most_accurate_kernel(df):
    """
    :return: integer representing the row number of the most accurate kernel
    """
    
    best_kernel = list(df).index(max(df))

    return best_kernel


def get_kernel_with_highest_score(df):
    """
    :return: integer representing the row number of the kernel with the highest score
    """
    best_kernel = list(df).index(max(df))

    return best_kernel


def plot_roc_curve_with_score(df, alpha_slope=1.5):
    """
    :param df: a dataframe containing the results of compare_svms
    :param alpha_slope: alpha parameter for plotting the linear score line
    :return:
    """
    x = df.fpr.tolist()
    y = df.tpr.tolist()

    ###########################################################################
    # TODO: Implement the function                                            #
    ###########################################################################

    b = y[get_kernel_with_highest_score(df['score'])] - (alpha_slope * x[get_kernel_with_highest_score(df['score'])])
    straight_line = poly1d([alpha_slope, b])

    plt.title("ROC")
    plt.ylabel('Tpr')
    plt.xlabel('Fpr')

    plt.xlim([0,1.01])
    plt.ylim([0,3])

    plt.plot(x, y, 'ro', ms=5, mec='green')
    plt.plot([0,1], straight_line([0,1]), "-r")

    plt.show()

    # print(x)
    # print(y)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################


def evaluate_c_param(data_array, labels_array, folds_count, kernels_list, best_kernel_params):
    """
    :param data_array: a numpy array with the features dataset
    :param labels_array: a numpy array with the labels
    :param folds_count: number of cross-validation folds
    :return: res: a dataframe containing the results for the different c values. columns similar to `compare_svms`
    """

    ###########################################################################
    # TODO: Implement the function                                            #
    ###########################################################################
    res = compare_svms(data_array, labels_array, folds_count, kernels_list, best_kernel_params)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return res


def get_test_set_performance(train_data, train_labels, test_data, test_labels, best_kernel, best_kernel_params):
    """
    :param train_data: a numpy array with the features dataset - train
    :param train_labels: a numpy array with the labels - train

    :param test_data: a numpy array with the features dataset - test
    :param test_labels: a numpy array with the labels - test
    :return: kernel_type: the chosen kernel type (either 'poly' or 'rbf')
             kernel_params: a dictionary with the chosen kernel's parameters - c value, gamma or degree
             clf: the SVM leaner that was built from the parameters
             tpr: tpr on the test dataset
             fpr: fpr on the test dataset
             accuracy: accuracy of the model on the test dataset
    """

    kernel_type = best_kernel
    kernel_params = best_kernel_params
    clf = SVC(class_weight='balanced', kernel = kernel_type , C = SVM_DEFAULT_C, 
        gamma = SVM_DEFAULT_GAMMA, degree = SVM_DEFAULT_DEGREE)  # TODO: set the right kernel
    tpr = 0.0
    fpr = 0.0
    accuracy = 0.0

    ###########################################################################
    # TODO: Implement the function                                            #
    ###########################################################################
    clf.set_params(**kernel_params)

    clf.fit(train_data, train_labels)
    tpr, fpr, accuracy = get_stats(clf.predict(test_data), test_labels)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return kernel_type, kernel_params, clf, tpr, fpr, accuracy
