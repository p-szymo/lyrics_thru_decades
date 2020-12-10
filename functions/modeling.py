# dataframe packages
import pandas as pd
import numpy as np
import itertools

# visualization package
import matplotlib.pyplot as plt

# string processing
import re
import string

# nlp packages
import nltk
from nltk import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from textblob import TextBlob as tb
import pronouncing

# statistical packages
from scipy.stats import ttest_ind
from sklearn.metrics import accuracy_score, f1_score, classification_report, \
    confusion_matrix
from sklearn.model_selection import cross_val_score


# cap/floor outlier values
def winsorizer(
    data,
    min_percentile=0.05,
    max_percentile=0.95
):

    '''
    Function that uses winsorization method of capping and
    flooring outlier values on both ends of a distribution.


    Input
    -----
    data : Pandas Series
        Values to be transformed.


    Optional input
    --------------
    min_percentile : float
        Percentile with minimum value to floor data
        (default=0.05, i.e. 5th percentile).

    max_percentile : float
        Percentile with maximum value to cap data
        (default=0.95, i.e. 95th percentile).


    Output
    ------
    capped : array
        NumPy array with capped and floored values.
        `data = winsorizer(data)` will overwrite input
        Series.

    '''

    # calculate thresholds
    min_thresh = data.quantile(min_percentile)
    max_thresh = data.quantile(max_percentile)

    # floor outlier values below mean
    capped = np.where(
        data < min_thresh,
        min_thresh,
        data)

    # cap outlier values above mean
    capped = np.where(
        capped > max_thresh,
        max_thresh,
        capped)

    # transformed values
    return capped


# contractions conversions
def load_dict_contractions():

    '''
    Dictionary of contractions as keys and their expanded words
    as values.


    [Code modified from]:
    https://stackoverflow.com/questions/19790188/expanding-english-\
    language-contractions-in-python

    '''

    return {
        "ain't": "is not",
        "amn't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "cuz": "because",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "could've": "could have",
        "daren't": "dare not",
        "daresn't": "dare not",
        "dasn't": "dare not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "d'you": "do you",
        "e'er": "ever",
        "em": "them",
        "'em": "them",
        "everyone's": "everyone is",
        "finna": "fixing to",
        "gimme": "give me",
        "gonna": "going to",
        "gon't": "go not",
        "gotta": "got to",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he he will have",
        "he's": "he is",
        "how'd": "how would",
        "how'll": "how will",
        "how're": "how are",
        "how's": "how is",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i'm'a": "i am about to",
        "i'm'o": "i am going to",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "i've": "i have",
        "kinda": "kind of",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "may've": "may have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "might've": "might have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "must've": "must have",
        "needn't": "need not",
        "needn't've": "need not have",
        "ne'er": "never",
        "o'": "of",
        "o'clock": "of the clock",
        "o'er": "over",
        "ol'": "old",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shalln't": "shall not",
        "shan't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "should've": "should have",
        "so's": "so as",
        "so've": "so have",
        "somebody's": "somebody is",
        "someone's": "someone is",
        "something's": "something is",
        "that'd": "that would",
        "that'd've": "that would have",
        "that'll": "that will",
        "that're": "that are",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there'll": "there will",
        "there're": "there are",
        "there's": "there is",
        "these're": "these are",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "this's": "this is",
        "those're": "those are",
        "to've": "to have",
        "'tis": "it is",
        "tis": "it is",
        "'twas": "it was",
        "twas": "it was",
        "wanna": "want to",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "weren't": "were not",
        "we've": "we have",
        "what'd": "what did",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where're": "where are",
        "where's": "where is",
        "where've": "where have",
        "which's": "which is",
        "will've": "will have",
        "who'd": "who would",
        "who'd've": "who would have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who're": "who are",
        "who's": "who is",
        "who've": "who have",
        "why'd": "why did",
        "why're": "why are",
        "why've": "why have",
        "why's": "why is",
        "won't": "will not",
        "won't've": "will not have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "would've": "would have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have",
        }


# simpler text cleaning
def simple_process(text):

    '''
    Function similar to `clean_text` function without removing
    stop words or words less than three characters.

    Makes text lowercase, removes punctuation, and converts
    contractions to full words.


    Input
    -----
    text : str
        Text to be cleaned.


    Output
    ------
    text : str
        Lowercase text without punctuation or contractions.

    '''

    # make text lowercase and convert some punctuation
    text = text.lower().replace("’", "'").replace('—', ' ').\
        replace('-', ' ')

    # remove punctuation other than apostrophes
    text = text.translate(str.maketrans(
        '', '', string.punctuation.replace("'", "")))

    # tokenization to remove non-words
    tokenizer = RegexpTokenizer("([a-z]+(?:'[a-z]+)?)")
    words = tokenizer.tokenize(text)

    # convert contractions
    contractions = load_dict_contractions()
    words = [contractions[word] if word in contractions else
             word for word in words]

    # stringify and remove leftover apostrophes
    text = ' '.join(words)
    text = text.replace("'", "")

    return text


def cv_plotter(
    model,
    features,
    target,
    scoring='f1_weighted',
    cv=10,
    n_jobs=-1
):

    '''
    Function to calculate and visualize cross-validation
    scores. Plots a graph with the title as the mean of
    the scores.

    Uses sklearn's cross_val_score. For documentation,
    visit:
    `https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html`

    Input
    -----
    model : sklearn or similar object
        Unfitted machine learning model.

    features : Pandas DataFrame
        Features for each data point.

    target : Pandas Series
        Target class for each data point.


    Optional input
    --------------
    scoring : str
        Metric to be calculated (default='f1_weighted')
        For list of metrics, visit:
        `https://scikit-learn.org/stable/modules/model_evaluation.html`

    cv : int
        Number of k-folds (default=10).

    n_jobs : int
        Number of computer cores to use (default=-1, i.e.
        all cores).


    Output
    ------
    cv_scores : list (float)
        List of calculated scores with length equal to
        input value for cv.

    Plots a histogram depicting scores.

    '''

    # calculate on k-folds
    cv_scores = cross_val_score(
        model, features, target, 
        scoring=scoring, cv=cv, n_jobs=n_jobs)

    # graph
    plt.hist(cv_scores)
    plt.title(f'Average score: {np.mean(cv_scores)}')
    plt.show()

    return cv_scores


# model predictions and printout
def predict(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    classes,
    to_print=True
):

    '''
    Function that predicts on training and testing data
    and returns predictions.

    Optionally prints out accuracy score, F1 score,
    classification report, and confusion matrix
    (defaults to printing).


    Input
    -----
    model : sklearn or similar object
        Fitted machine learning model.

    X_train : Pandas DataFrame
        Train set features.

    y_train : Pandas Series
        Train set classes.

    X_test : Pandas DataFrame
        Test set features.

    y_test : Pandas DataFrame
        Test set classes.

    classes : list (str)
        Target classes (in the order in which
        they appear in DataFrame).


    Optional input
    --------------
    to_print : bool
        Whether or not to print metrics and reports
        (default=True).
        Set `to_print=False` to not print.


    Output
    ------
    train_preds : list (str, int, or float)
        List of model's predictions for the training
        set.
    test_preds : list (str, int, or float)
        List of model's predictions for the testing
        set.

    Optional printout (defaults to printing).

    '''

    # predict class for the train and test sets
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    # accuracy and f1 scores for train and test sets
    acc_train = accuracy_score(y_train, train_preds)
    acc_test = accuracy_score(y_test, test_preds)

    # binary scores
    if len(classes) == 2:
        f1_train = f1_score(y_train, train_preds)
        f1_test = f1_score(y_test, test_preds)
    # multiclass scores
    else:
        f1_train = f1_score(y_train, train_preds, average='weighted')
        f1_test = f1_score(y_test, test_preds, average='weighted')

    # print metrics, classification report, and confusion matrix
    if to_print:
        print('-----TRAIN-----')
        print(f'Accuracy: {acc_train}')
        print(f'F1 score: {f1_train}')
        print('\n-----TEST-----')
        print(f'Accuracy: {acc_test}')
        print(f'F1 score: {f1_test}')

        print('\n' + '-' * 100 + '\n')

        # print out report for test predictions
        print(classification_report(y_test,
                                    test_preds,
                                    target_names=classes))

        print('\n' + '-' * 100 + '\n')

        # print out confusion matrix
        print("CONFUSION MATRIX:\n")
        print(confusion_matrix(y_test, test_preds))

    return train_preds, test_preds


# confusion matrix plotter
def plot_confusion_matrix(
    cm,
    classes,
    normalize=False,
    title='Confusion matrix',
    cmap=plt.cm.Blues
):

    '''
    Function that prints and plots a model's confusion matrix.


    Input
    -----
    cm : sklearn confusion matrix
        `sklearn.metrics.confusion_matrix(y_true, y_pred)`

    classes : list (str)
        Names of target classes.


    Optional input
    --------------
    normalize : bool
        Whether to apply normalization (default=False).
        Normalization can be applied by setting `normalize=True`.

    title : str
        Title of the returned plot.

    cmap : matplotlib color map
        For options, visit:
        `https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html`


    Output
    ------
    Prints a stylized confusion matrix.

    '''

    # number of classes, for dimensions
    n = len(classes)

    # convert to percentage, if normalize set to True
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # plot
    plt.figure(figsize=(2*n, 1.5*n))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # format true positives and others
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize=16,
                 horizontalalignment="center", verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # add axes labels
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# naive bayes feature importances printer
def print_nb_features(model, df, label_names, num_features=10):

    '''
    Function to print feature importances for Bernoulli and
    Multinomial Naive Bayes models, sorted by measure of importance.


    Input
    -----
    model : Naive Bayes model
        `sklearn.naive_bayes.BernoulliNB()`
        `sklearn.naive_bayes.MultinomialNB()`

    df : Pandas DataFrame
        Features used in model.


    Optional input
    --------------
    num_features : int
        The number of features to print (default=10).
        All feature importances can be shown by setting
        `num_features=df.shape[1]`.


    Output
    ------
    Prints labels and a list of features.

    '''

    # loop through each label
    for i, label in enumerate(label_names):
        # sorted features per class by importance
        prob_sorted = model.feature_log_prob_[i, :].argsort()

        # prettified labels
        label_pretty = label.replace("_", "-").title()

        # printable features
        features = ", ".join(list(np.take(
            df.columns,
            prob_sorted[:num_features])))

        # printout class features
        print(f'{label_pretty}:\n{features}\n')


# decision tree feature importances plotter
def plot_tree_features(
    model,
    df,
    num_features=10,
    to_print=True,
    to_save=False,
    file_name=None
):

    '''
    This function plots feature importances for Decision Tree models
    and optionally prints a list of tuples with features and their
    measure of importance.


    Input
    -----
    model : Decision Tree model
        `sklearn.tree.DecisionTreeClassifier()`

    df : Pandas DataFrame
        Features used in model.


    Optional input
    --------------
    num_features : int
        The number of features to plot/print (default=10).
        All feature importances can be shown by setting
        `num_features=df.shape[1]`.

    to_print : bool
        Whether to print list of feature names and their impurity
        decrease values (default=True).
        Printing can be turned off by setting `to_print=False`.

    file_name : str
        Path and name to save a graph (default=None).
        If `file_name=None`, the graph will not be saved.


    Output
    ------
    Prints a bar graph and optional list of tuples.

    '''

    features_dict = dict(zip(df.columns, model.feature_importances_))
    sorted_d = sorted(
        features_dict.items(),
        key=lambda x: x[1],
        reverse=True)[
        :num_features]

    # top 10 most important features
    tree_importance = [x[1] for x in sorted_d]

    # prettify the graph
    plt.figure(figsize=(12, 10))
    plt.title('Decision Tree Feature Importances', fontsize=25, pad=15)
    plt.xlabel('')
    plt.ylabel('Gini Importance', fontsize=22, labelpad=15)
    plt.ylim(bottom=sorted_d[-1][1]/1.75, top=sorted_d[0][1]*1.05)
    plt.xticks(rotation=80, fontsize=20)
    plt.yticks(fontsize=20)

    # plot
    plt.bar([x[0] for x in sorted_d], tree_importance)

    # prepare to display
    plt.tight_layout()

    if file_name:
        # save plot
        plt.savefig(file_name, bbox_inches='tight', transparent=True)

    # show plot
    plt.show()

    if to_print:
        # print a list of feature names and their impurity decrease value in
        # the decision tree
        print('\n\n\n')
        print(sorted_d)


# random forest feature importances plotter
def plot_forest_features(
    model,
    X,
    num_features=10,
    to_print=True
):

    '''
    This function plots feature importances for Random Forest models
    and optionally prints a list of tuples with features and their
    measure of importance.


    Input
    -----
    model : Random Forest model
        `sklearn.ensemble.RandomForestClassifier()`

    X : Pandas DataFrame
        Features used in model.


    Optional input
    --------------
    num_features : int
        The number of features to plot/print (default=15).
        All feature importances can be shown by setting
        `num_features=X.shape[1]`.

    to_print : bool
        Whether to print list of feature names and their impurity
        decrease values (default=True).
        Printing can be turned off by setting `to_print=False`.


    Output
    ------
    Prints a bar graph and optional list of tuples.

    '''

    # list of tuples (column index, measure of feature importance)
    imp_forest = model.feature_importances_

    # sort feature importances in descending order, slicing top number of
    # features
    indices_forest = np.argsort(imp_forest)[::-1][:num_features]

    # rearrange feature names so they match the sorted feature importances
    names_forest = [X.columns[i] for i in indices_forest]

    # create plot, using num_features as a dimensional proxy
    plt.figure(figsize=(num_features * 1.5, num_features))
    plt.bar(range(num_features), imp_forest[indices_forest])

    # prettify plot
    plt.title('Random Forest Feature Importances', fontsize=30, pad=15)
    plt.ylabel('Average Decrease in Impurity', fontsize=22, labelpad=20)
    # add feature names as x-axis labels
    plt.xticks(range(num_features), names_forest, fontsize=20, rotation=90)
    plt.tick_params(axis="y", labelsize=20)

    # Show plot
    plt.tight_layout()
    plt.show()

    if to_print:
        # print a list of feature names and their impurity decrease value in
        # the forest
        print('\n\n\n')
        print([
            (i, j) for i, j in zip(names_forest, imp_forest[indices_forest])
            ])


def svm_features(
    model,
    col_names,
    num_features=10,
    title='Feature Importances'
):

    '''
    Function to plot most important features in an SVM model.


    Input
    -----
    model : linear SVC model
        `sklearn.svm.LinearSVC` OR
        `sklearn.svm.SVC(kernel='linear')`

    col_names : list (str)
        List of column names.

    num_features : int
        Number of features to include in graph
        (default=10).

    title : str
        Title of returned graph.


    Output
    ------
    Prints a horizontal bar graph.

    '''

    # prettify graph
    plt.figure(figsize=(num_features/2,
                        0.8*num_features))
    plt.title(title, fontsize=2*num_features, pad=1.5*num_features)
    plt.xlabel('Coefficient (absolute value)',
               fontsize=1.5*num_features, labelpad=num_features)

    # plot top ten features
    pd.Series(abs(model.coef_[0]), index=col_names)\
        .nlargest(num_features).plot(kind='barh')
