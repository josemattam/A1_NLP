# models.py

from sentiment_data import *
from utils import *

from collections import Counter
import re
import numpy as np
import spacy
import nltk
from nltk import bigrams
from nltk.corpus import stopwords

nlp = spacy.load('en_core_web_sm', disable=['tagger', 'parser', 'ner'])
nltk.download('stopwords')
stopWords = set(stopwords.words('english'))

# sltk and spacy



class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        # take input
        # Split into tokens
        # one word at a time
        features = Counter()
        joinedSentence = nlp(" ".join(sentence))

        for word in joinedSentence:
            wordStr = word.text.lower()
            if word.is_alpha:                
                if wordStr in self.indexer.objs_to_ints or add_to_indexer:
                    index = self.indexer.add_and_get_index(wordStr)
                    features[index] += 1
        return features

        
        # working old version:
        # for word in sentence:
        #     # removes special characters
        #     wordNoChar = re.sub(r"[^a-zA-Z0-9\-]", "", word)
        #     wordNoChar = wordNoChar.capitalize()
        #     if word in self.indexer.objs_to_ints or add_to_indexer:
        #         index = self.indexer.add_and_get_index(word)
        #         features[index] += 1
        # return features
        

        # old version (doesnt work):
        #     if wordNoChar:      # if not empty
        #         if add_to_indexer or not self.indexer.contains(wordNoChar):
        #             index = self.indexer.add_and_get_index(wordNoChar)
        #         else:
        #             index = self.indexer.index_of(wordNoChar)
        #         features[index] += 1

        # return features
      


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer
    
    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        features = Counter()
        joinedSentence = nlp(" ".join(sentence))
        tokens = [token.text for token in joinedSentence]
        bigramList = list(bigrams(tokens))

        for bigram in bigramList:
            words = '_'.join(bigram)
            if words in self.indexer.objs_to_ints or add_to_indexer:
                index = self.indexer.add_and_get_index(words)
                features[index] += 1
        return features 
                    



class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer):
        raise Exception("Must be implemented")


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")
    
    def get_feature_vector_from_counter(self, features):
        # print("Weights size:", len(self.weights))
        # print("Features:", features)

        fvec = np.zeros(len(self.weights))
        #print("###############\nfvec count:",len(fvec), "\nfeatures count:", len(features))
        for i, count in features.items():
            fvec[i] = count
        return fvec


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, feat_extractor: FeatureExtractor, learning_rate = 0.001):
        self.feat_extractor = feat_extractor
        self.learning_rate = learning_rate
        self.weights = np.zeros(feat_extractor.get_indexer().__len__())

    
    def predict(self, sentence: List[str]) -> int:
        features = self.feat_extractor.extract_features(sentence) # doesnt add to indexer
        fvec = self.get_feature_vector_from_counter(features)
        dotprod = np.dot(fvec, self.weights) # prediction value = dot product
        if dotprod > 0:
            return 1
        return 0
    
    def update_weights(self, y, features):
        fvec = self.get_feature_vector_from_counter(features)
        y_hat = np.dot(fvec, self.weights)
        # change weights if discrepancy between y and y_hat
        if (y_hat > 0 and y == 0) or (y_hat <= 0 and y == 1):
            if y == 1:
                self.weights += self.learning_rate * fvec
            else:
                self.weights -= self.learning_rate * fvec



class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """
    def __init__(self, feat_extractor: FeatureExtractor):
        self.feat_extractor = feat_extractor
        self.learning_rate = 0.001
        self.weights = np.zeros(feat_extractor.get_indexer().__len__())

    
    def logistic_regression(self, x):
        return 1/(1 + np.exp(-x))


    def predict(self, sentence: List[str]) -> int:
        features = self.feat_extractor.extract_features(sentence) # doesnt add to indexer
        fvec = self.get_feature_vector_from_counter(features)
        logreg = self.logistic_regression(np.dot(fvec, self.weights)) # prediction value = dot product
        if logreg > 0.5:
            return 1
        return 0
    

    def update_weights(self, y, features):
        fvec = self.get_feature_vector_from_counter(features)
        y_hat = np.dot(fvec, self.weights)
        grad = fvec * (y - y_hat)
        self.weights += self.learning_rate * grad    

    


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    for ex in train_exs:
        feat_extractor.extract_features(ex.words, add_to_indexer=True)

    classifier = PerceptronClassifier(feat_extractor)
    for ex in train_exs:
        features = feat_extractor.extract_features(ex.words, add_to_indexer=False)
        #print("*****Indexer size after processing training data:", len(feat_extractor.get_indexer().objs_to_ints))
        classifier.update_weights(ex.label, features)
        
    return classifier



def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    for ex in train_exs:
        feat_extractor.extract_features(ex.words, add_to_indexer=True)

    classifier = LogisticRegressionClassifier(feat_extractor)
    for ex in train_exs:
        features = feat_extractor.extract_features(ex.words, add_to_indexer=False)
        #print("*****Indexer size after processing training data:", len(feat_extractor.get_indexer().objs_to_ints))
        classifier.update_weights(ex.label, features)
    
    return classifier


def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model