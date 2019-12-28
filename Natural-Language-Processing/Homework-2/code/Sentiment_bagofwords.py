import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sgd import load_saved_params, sgd
from softmaxreg import accuracy
from collections import Counter
from data_utils import *

# Load the dataset
dataset = StanfordSentiment()
tokens = dataset.tokens() # a dictionary
nWords = len(tokens)

# Set the feature to be the dimension of the words

dimVectors = nWords

# Load the train set
trainset = dataset.getTrainSentences()
nTrain = len(trainset)
trainFeatures = np.zeros((nTrain, dimVectors))
trainLabels = np.zeros((nTrain,), dtype=np.int32)
for i in range(nTrain):
    words, trainLabels[i] = trainset[i]
    wordTokens = [tokens[word] for word in words]
    tokenCount = Counter(wordTokens)
    for wordToken in wordTokens:
        trainFeatures[i, wordToken] = tokenCount[wordToken]

# Prepare dev set features
devset = dataset.getDevSentences()
nDev = len(devset)
devFeatures = np.zeros((nDev, dimVectors))
devLabels = np.zeros((nDev,), dtype=np.int32)
for i in range(nDev):
    words, devLabels[i] = devset[i]
    wordTokens = [tokens[word] for word in words]
    tokenCount = Counter(wordTokens)
    for wordToken in wordTokens:
        devFeatures[i, wordToken] = tokenCount[wordToken]

# Prepare test set features
testset = dataset.getTestSentences()
nTest = len(testset)
testFeatures = np.zeros((nTest, dimVectors))
testLabels = np.zeros((nTest,), dtype=np.int32)
for i in range(nTest):
    words, testLabels[i] = testset[i]
    wordTokens = [tokens[word] for word in words]
    tokenCount = Counter(wordTokens)
    for wordToken in wordTokens:
        testFeatures[i, wordToken] = tokenCount[wordToken]

# Try our regularization parameters
results = []

# 1. Multinomial Naive Bayes + Bag of Words

# Test on train set
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
clf.fit(trainFeatures,trainLabels)

# Test on train set
pred = clf.predict(trainFeatures)
trainAccuracy = accuracy(trainLabels, pred)
print("Train accuracy (%%): %f" % trainAccuracy)

# Test on dev set
pred = clf.predict(devFeatures)
devAccuracy = accuracy(devLabels, pred)
print("Dev accuracy (%%): %f" % devAccuracy)

# Test on test set
pred = clf.predict(testFeatures)
testAccuracy = accuracy(testLabels, pred)
print("Test accuracy (%%): %f" % testAccuracy)

# Save the results and weights
results.append({
    "method":"Multinomial Naive Bayes + Bag of Words",
    "train" : trainAccuracy, 
    "dev" : devAccuracy,
    "test": testAccuracy})

# 2. Logistic Regression + Bag of Words

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
clf.fit(trainFeatures,trainLabels)


# Test on train set
pred = clf.predict(trainFeatures)
trainAccuracy = accuracy(trainLabels, pred)
print("Train accuracy (%%): %f" % trainAccuracy)

# Test on dev set
pred = clf.predict(devFeatures)
devAccuracy = accuracy(devLabels, pred)
print("Dev accuracy (%%): %f" % devAccuracy)

# Test on test set
pred = clf.predict(testFeatures)
testAccuracy = accuracy(testLabels, pred)
print("Test accuracy (%%): %f" % testAccuracy)



# Save the results and weights
results.append({
    "method":"Logistic Regression + Bag of Words",
    "train" : trainAccuracy, 
    "dev" : devAccuracy,
    "test": testAccuracy})


# 3. Multinomial Naive Bayes + TF-IDF

x_train = []
y_train = []
x_dev = []
y_dev = []
x_test = []
y_test = []
for i in range(nTrain):
    words, label = trainset[i]
    x_train.append(" ".join(words))
    y_train.append(label)

for i in range(nDev):
    words, label = devset[i]
    x_dev.append(" ".join(words))
    y_dev.append(label)

for i in range(nTest):
    words, label = testset[i]
    x_test.append(" ".join(words))
    y_test.append(label)

from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer()
trainFeatures = tf.fit_transform(x_train)
devFeatures = tf.transform(x_dev)
testFeatures = tf.transform(x_test)

clf = MultinomialNB()
clf.fit(trainFeatures,trainLabels)

# Test on train set
pred = clf.predict(trainFeatures)
trainAccuracy = accuracy(trainLabels, pred)
print("Train accuracy (%%): %f" % trainAccuracy)

# Test on dev set
pred = clf.predict(devFeatures)
devAccuracy = accuracy(devLabels, pred)
print("Dev accuracy (%%): %f" % devAccuracy)

# Test on test set
pred = clf.predict(testFeatures)
testAccuracy = accuracy(testLabels, pred)
print("Test accuracy (%%): %f" % testAccuracy)

# Save the results and weights
results.append({
    "method":"Multinomial Naive Bayes + TF-IDF",
    "train" : trainAccuracy, 
    "dev" : devAccuracy,
    "test": testAccuracy})

# 4. Logistic Regression + TF-IDF


clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
clf.fit(trainFeatures,trainLabels)

# Test on train set
pred = clf.predict(trainFeatures)
trainAccuracy = accuracy(trainLabels, pred)
print("Train accuracy (%%): %f" % trainAccuracy)

# Test on dev set
pred = clf.predict(devFeatures)
devAccuracy = accuracy(devLabels, pred)
print("Dev accuracy (%%): %f" % devAccuracy)

# Test on test set
pred = clf.predict(testFeatures)
testAccuracy = accuracy(testLabels, pred)
print("Test accuracy (%%): %f" % testAccuracy)

# Save the results and weights
results.append({
    "method":"Logistic Regression + TF-IDF",
    "train" : trainAccuracy, 
    "dev" : devAccuracy,
    "test": testAccuracy})

for result in results:
    print(result)