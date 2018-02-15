# nltk_classify
Scripts for text classification with NLTK

## CSV Classify

The ClassifierCSV class contained in csv_classify.py aims to make it easy to train and save a classifier. You simple need to parse a CSV file with two columns, your text content in the left hand column and your labels on the right hand side. It is possible to train using either NLTK's basic naive bayes classifier or a number of the classifier's availiable in Sklearn.

## Using the standard classifier
```python3
c = ClassifierCSV('example-dataset.csv', featureset_size=2000)
c.train_naive_bayes_classifier()
print(c.classify_new_sentence('What an amazing movie'))
```
To train a classifier with the standard Bayes classifier, we simply pass our dataset CSV and the feature set size you want to use. The size of the feature set you opt to use will depend on your computing resources and the size of your dataset. With larger datasets you can typically get better results by using a larger feature set.

## Using a Sklearn classifier
```python3
from sklearn.naive_bayes import BernoulliNB
c = ClassifierCSV('example-dataset.csv', featureset_size=2000)
c.train_sklearn_classifier(BernoulliNB)
print(c.classify_new_sentence('What an amazing movie'))
```
The sklearn classifiers support the same interface but require us to pass in the classifier in question to the train_sklearn_classifier. 

## Comparing Classifiers
```
Using the Naives Bayes Classifier
Achieved 84.09% accuracy against training set  # Feature Set Size - 5000
Achieved 72.89% accuracy against test set
Using the BernouliNB Classifier
Achieved 83.93% accuracy against training set # Feature Set Size - 5000
Achieved 79.78% accuracy against test set
```
