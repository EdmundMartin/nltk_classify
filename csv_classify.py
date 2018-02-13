import re
import nltk
from random import shuffle
import pickle


class ClassifierCSV:

    def __init__(self, csv_file, featureset_size=1000, test_ratio=0.1):
        self.csv_file = csv_file
        self.documents = []
        self.words = []
        self.featureset_size = featureset_size
        self.test_ratio = test_ratio
        self.feature_words = None
        self.classifier = None

    def __document_features(self, document):
        document_words = set(document)
        features = {}
        for word in self.feature_words:
            features['contains({})'.format(word)] = (word in document_words)
        return features

    def _read_csv(self):
        with open(self.csv_file, 'r') as input_csv:
            for item in input_csv:
                item = item.split(',')
                doc, label = re.findall('\w+', ''.join(item[:-1]).lower()), item[-1].strip()
                for word in doc:
                    self.words.append(word.lower())
                self.documents.append((doc, label))

    def _generate_word_features(self):
        frequency_dist = nltk.FreqDist()
        for word in self.words:
            frequency_dist[word] += 1
        self.feature_words = list(frequency_dist)[:self.featureset_size]

    def train_classifier(self):
        self._read_csv()
        self._generate_word_features()
        shuffle(self.documents)
        feature_sets = [(self.__document_features(d), c) for (d, c) in self.documents]
        cutoff = int(len(feature_sets) * self.test_ratio)
        train_set, test_set = feature_sets[cutoff:], feature_sets[:cutoff]
        self.classifier = nltk.NaiveBayesClassifier.train(train_set)
        print('Achieved {0:.2f}% accuracy against training set'.format(nltk.classify.accuracy(self.classifier, train_set)))
        print('Achieved {0:.2f}% accuracy against test set'.format(nltk.classify.accuracy(self.classifier, test_set)))

    def classify_new_sentence(self, sentence):
        test_features = {}
        for word in self.feature_words:
            test_features['contains({})'.format(word.lower())] = (word.lower() in nltk.word_tokenize(sentence))
        return self.classifier.classify(test_features)

    def save_model(self, filename):
        save_classifier = open(filename, "wb")
        pickle.dump(self.classifier, save_classifier)
        save_classifier.close()

    def load_model(self, filename):
        classifier_f = open(filename, "rb")
        self.classifier = pickle.load(classifier_f)
        classifier_f.close()


if __name__ == '__main__':
    c = ClassifierCSV('tweets-with-sentiment.csv', featureset_size=10000)
    c.train_classifier()
    print(c.classify_new_sentence('What an amazing movie'))