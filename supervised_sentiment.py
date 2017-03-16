import sys
import collections
import sklearn.naive_bayes
import sklearn.linear_model
import nltk
import random
random.seed(0)
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
from collections import Counter
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
#nltk.download("stopwords")          # Download the stop words from nltk


# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
if len(sys.argv) != 3:
    print "python sentiment.py <path_to_data> <0|1>"
    print "0 = NLP, 1 = Doc2Vec"
    exit(1)
path_to_data = sys.argv[1]
method = int(sys.argv[2])



def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)

    if method == 0:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
    if method == 1:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)
    print "Naive Bayes"
    print "-----------"
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)
    print ""
    print "Logistic Regression"
    print "-------------------"
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)



def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir+"train-pos.txt", "r") as f:
        for i,line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_pos.append(words)
    with open(path_to_dir+"train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_neg.append(words)
    with open(path_to_dir+"test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_pos.append(words)
    with open(path_to_dir+"test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_neg.append(words)

    return train_pos, train_neg, test_pos, test_neg



def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # English stopwords from nltk
    stopwords = set(nltk.corpus.stopwords.words('english'))

    # Determine a list of words that will be used as features.
    # This list should have the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.
    # YOUR CODE HERE

    print "Entered into NLP featuring"
    stopwords = list(stopwords)
    features = list()

    positive_diction = dict()
    negative_diction = dict()

    #Counter is useful to get the count of redundant words, and keep a single entry for every word into the list/dict
    positive_diction = Counter(getAllWordList(train_pos))
    negative_diction = Counter(getAllWordList(train_neg))

    #Trial ccode to check values
    '''
    for keys,values in positive_diction.items():
        print(keys)
        print(values)

    print "Length : %d" %len(positive_diction)
    '''

    #List of words into features from positive dictionary: fullfillying all properties
    for key, value in positive_diction.iteritems():
        if(key not in features and (value >= len(train_pos)/100 or negative_diction.get(key) > len(train_neg)/100) and value >= 2*negative_diction.get(key) and key not in stopwords):
            features.append(key)

    #List of words from negative dictionary: fullfillying all properties
    for key, value in negative_diction.iteritems():
        if(key not in features and (value >= len(train_neg)/100 or positive_diction.get(key) > len(train_pos)/100) and value >= 2*positive_diction.get(key) and key not in stopwords):
            features.append(key)

    #print "Features" + str(len(features))              #To check length of the features list

    # Using the above words as features, construct binary vectors for each text in the training and test set.
    # These should be python lists containing 0 and 1 integers.
    # YOUR CODE HERE

    train_pos_vec = map(lambda x: map(lambda y: 1 if y in x else 0, features), train_pos)
    train_neg_vec = map(lambda x: map(lambda y: 1 if y in x else 0, features), train_neg)
    test_pos_vec = map(lambda x: map(lambda y: 1 if y in x else 0, features), test_pos)
    test_neg_vec = map(lambda x: map(lambda y: 1 if y in x else 0, features), test_neg)

    print "Featuring successfully executed."
    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec

#Function to get all words, as many as times they occur
def getAllWordList(training_data):
    all_words_list = []
    for words in training_data:
        ongoing_word_list = []
        for word in words:
            if word not in ongoing_word_list:
                ongoing_word_list.append(word)
                all_words_list.append(word)
    return all_words_list


def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # Doc2Vec requires LabeledSentence objects as input.
    # Turn the datasets from lists of words to lists of LabeledSentence objects.
    # YOUR CODE HERE
    print "Entered into Doc2Vec Featuring"

    labeled_train_pos = labeledDataSetToList(train_pos, 'train_pos')
    labeled_train_neg = labeledDataSetToList(train_neg, 'train_neg')
    labeled_test_pos = labeledDataSetToList(test_pos, 'test_pos')
    labeled_test_neg = labeledDataSetToList(test_neg, 'test_neg')

    # Initialize model
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
    model.build_vocab(sentences)

    # Train the model
    # This may take a bit to run
    for i in range(5):
        print "Training iteration %d" % (i)
        random.shuffle(sentences)
        model.train(sentences)

    # Use the docvecs function to extract the feature vectors for the training and test data
    # YOUR CODE HERE
    train_pos_vec = list()
    train_neg_vec = list()
    test_pos_vec = list()
    test_neg_vec = list()

    for i in range(len(train_pos)):
        train_pos_vec.append(model.docvecs['train_pos'+str(i)])
    for i in range(len(train_neg)):
        train_neg_vec.append(model.docvecs['train_neg'+str(i)])
    for i in range(len(test_pos)):
        test_pos_vec.append(model.docvecs['test_pos'+str(i)])
    for i in range(len(test_neg)):
        test_neg_vec.append(model.docvecs['test_neg'+str(i)])

    print "Featuring completed successfully."

    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec

#Funtion to get list from labeled data set
def labeledDataSetToList(obj_list, type_label):
    labeled = []
    for i in range(len(obj_list)):
        label = type_label + str(i)
        labeled.append(LabeledSentence(obj_list[i], [label]))
    return labeled


def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
    # For BernoulliNB, use alpha=1.0 and binarize=None
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE

    #Total positive and negative trained words (0 or 1)
    A = train_pos_vec + train_neg_vec
    nb_model = BernoulliNB(alpha = 1.0, binarize = None).fit(A, Y)
    lr_model = LogisticRegression().fit(A, Y)

    print "Models build successfully."
    return nb_model, lr_model



def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE

    #Total positive and negative trained words (0 or 1)
    B = train_pos_vec + train_neg_vec
    nb_model = GaussianNB().fit(B, Y)
    lr_model = LogisticRegression().fit(B, Y)

    print "Models build successfully."

    return nb_model, lr_model



def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    # Use the predict function and calculate the true/false positives and true/false negative.
    # YOUR CODE HERE

    test_data = ["pos"]*len(test_pos_vec) + ["neg"]*len(test_neg_vec)
    #print "Length of Test Data: " + str(len(test_data))    #Get the length of data

    prediction = model.predict(test_pos_vec + test_neg_vec)

    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
    match = 0

    #Loop to compute true/false positives and true/false negatives
    for i in range(len(test_data)):
        if test_data[i] == prediction[i]:
            match = match + 1
            if test_data[i]=='pos':                 #Match for all true positives
                true_pos = true_pos + 1
            else:                                   #Match for all true negatives
                true_neg = true_neg + 1
        else:
            if test_data[i] == 'pos':               #Match for all false negatives
                false_neg = false_neg + 1
            else:                                   #Match for all false positives
                false_pos = false_pos + 1

    #calculating accuracy using above values
    accuracy = float((float)(match)/(float)(len(test_data)))


    if print_confusion:
        print "predicted:\tpos\tneg"
        print "actual:"
        print "pos\t\t%d\t%d" % (true_pos, false_neg)
        print "neg\t\t%d\t%d" % (false_pos, true_neg)
    print "accuracy: %f" % (accuracy)



if __name__ == "__main__":
    main()
