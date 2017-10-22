from preprocess import *
from sklearn import model_selection, cross_validation
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing, svm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion


#plotting confusion matrix to show predicted vs actual values
test_result = []
gold_result = []
pos,neu,neg = 0,0,0

class MultiColumnLabelEncoder:
    def __init__(self, encoder, columns = None):
        self.columns = columns # array of column names to encode
        self.encoder = encoder
        
    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = self.encoder.fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = self.encoder.fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)
        
df = pd.read_csv('../Data/bitcoinparser2MFilteredEqual.csv')
X = df[['FilteredTweet', 'Closing', 'VWAP', 'Volume']]
#xfull = df['FilteredTweet']
y = df['Label']


#Read the tweets one by one and process it
with open('../Data/bitcoinparser2MFilteredEqual.csv', 'rb') as csvfile:
    df = pd.read_csv(csvfile)
    i = 0
    xfull, yfull = [], []
    for row in df['Tweet']:
        sentiment = df['Label'][i]
        yfull.append(sentiment)
        if sentiment == 0:
            neu = neu+1
        elif sentiment == 1:
            pos = pos+1
        elif sentiment == -1:
            neg = neg+1
        tweet = df['Tweet'][i]
        processedTweet = processTweet(tweet)
        featureVector = getFeatureVector(processedTweet)
        xfull.append(processedTweet)
        featureList.extend(featureVector)
        tweets.append((featureVector, sentiment))
        xset.append(" ".join(featureVector))
        i=i+1
csvfile.close()
print xfull

mb = MultiLabelBinarizer()
le = preprocessing.LabelEncoder()
lx = MultiLabelBinarizer()
st = StandardScaler()

#spliting train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
#X_train, X_test, y_train, y_test = X[:1191], X[1191:], y[:1191], y[1191:]
vectorizer = CountVectorizer(ngram_range=(5, 5), analyzer='char_wb')
transformer = TfidfVectorizer(min_df=1)
t_full = transformer.fit_transform(X['FilteredTweet'])
t_train = transformer.fit_transform(X_train['FilteredTweet'])
t_test = transformer.transform(X_test['FilteredTweet'])
X_train, X_test = t_train, t_test
"""
f_full = st.fit_transform(X[['Closing', 'Volume']])
f_train = st.fit_transform(X_train[['Closing', 'Volume']])
f_test = st.transform(X_test[['Closing','Volume']])

# combine countvectorizer column with standard scaler..
X =  np.hstack((t_full.todense(),f_full))
X_train = np.hstack((t_train.todense(),f_train))
X_test = np.hstack((t_test.todense(),f_test))
"""
# Select cross validation method
kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=10)

#tweet_train = pd.DataFrame(vectorizer.transform(X_train).todense(), columns=vectorizer.get_feature_names())
#tweet_test = pd.DataFrame(vectorizer.transform(X_test).todense(), columns=vectorizer.get_feature_names())

#X_train, X_test, y_train, y_test = train_test_split(df_x,df_y,test_size=0.2)
#df_train = pd.DataFrame(st.fit_transform(X_train).toarray())
#df_test = pd.DataFrame(st.transform(X_test).toarray())
#print pd.DataFrame(vectorizer.fit_transform(X_train['FilteredTweet']).toarray())

#X_train = MultiColumnLabelEncoder(vectorizer, columns = ['FilteredTweet']).fit_transform(X_train)
#X_test = MultiColumnLabelEncoder(vectorizer, columns = ['FilteredTweet']).fit_transform(X_test)
#print X_train
#X_train_a= vectorizer.fit_transform(X_train['FilteredTweet'])
#X_test_a= pd.DataFrame(vectorizer.transform(X_test['FilteredTweet']).todense())
#print type(X_train_a)
#X_train_a.columns =['FilteredTweet']
#X_test_a.columns = ['FilteredTweet']
#X_train_f = pd.concat([X_train_a['FilteredTweet'], X_train[['Closing','VWAP','Volume']]], axis=1)
#X_test_f = pd.concat([X_test_a['FilteredTweet'], X_test[['Closing','VWAP','Volume']]], axis=1)
#print X_train_f

"""
# Train Multinomial Naive Bayes
clf = MultinomialNB()
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
score = clf.score(X_test, y_test)
confusion = pd.crosstab(y_test, predictions)
normalized = confusion/confusion.sum(axis=1)
plot_confusion_matrix('MNBcm.png', confusion)
plot_classification_report('MNBcr.png', classification_report(predictions,y_test))
print "Multinomial Naive Bayes Classifier Accuracy: ", '{:.2%}'.format(score)
cross_val = cross_val_score(clf,X_train, y_train, cv=10)
print "Multinomial Naive Bayes Classifier Cross Validation Accuracy: " , '{:.2%}'.format(cross_val.mean()), "+/-",  '{:.2%}'.format(cross_val.std())
print "Mean Absolute Error: %s" % '{:.2%}'.format(mean_absolute_error(y_test, predictions))
"""


# Train SVC(Linear) and outputs results...
clf = svm.SVC(kernel='linear', probability=True).fit(X_train, y_train)
predictions = clf.predict(X_test)
#plot_svm('../Output/SVCLinearcart.png', clf, X_train, y_train, 'Linear SVC')
score = clf.score(X_test, y_test) 
confusion = pd.crosstab(y_test, predictions)
normalized = confusion/confusion.sum(axis=1)
plot_confusion_matrix('../Output/SVCLinearcm.png', confusion)
plot_classification_report('../Output/SVCLinearcr.png', classification_report(predictions,y_test))
print "SVC Linear Classifier Accuracy: ", '{:.2%}'.format(score)
cross_val = cross_val_score(clf, X_train, y_train, cv=kfold, scoring='accuracy')
#mae = cross_val_score(clf, X_train, y_train, cv=kfold, scoring='neg_mean_absolute_error')
#r2 = cross_val_score(clf, X_train, y_train, cv=kfold, scoring='r2')
#logloss = cross_val_score(clf, X_train, y_train, cv=kfold, scoring='neg_log_loss')
#auc = cross_val_score(clf, X_train, y_train, cv=kfold, scoring='roc_auc')
print "SVC Linear Classifier Cross Validation Accuracy: " , '{:.2%}'.format(cross_val.mean()), "+/-",  '{:.2%}'.format(cross_val.std())
#print "Mean Absolute Error: ", '{:.2%}'.format(mae.mean()), "+/-",  '{:.2%}'.format(mae.std())
#print "R^2: ", '{:.2%}'.format(r2.mean()), "+/-",  '{:.2%}'.format(r2.std())
#print "Logarithmic Loss: ", '{:.2%}'.format(logloss.mean()), "+/-",  '{:.2%}'.format(logloss.std())
#print "Area Under Curve: ", '{:.2%}'.format(auc.mean()), "+/-",  '{:.2%}'.format(auc.std())
plot_roc('../Output/SVCLinearroc.png', clf, X[:1191], y[:1191])


# Train SVC and outputs results...
clf = svm.SVC(probability=True).fit(X_train, y_train)
predictions = clf.predict(X_test)
score = clf.score(X_test, y_test) 
confusion = pd.crosstab(y_test, predictions)
normalized = confusion/confusion.sum(axis=1)
plot_confusion_matrix('../Output/SVCcm.png', confusion)
plot_classification_report('../Output/SVCcr.png', classification_report(predictions,y_test))
print "SVC Classifier Accuracy: ", '{:.2%}'.format(score)
cross_val = cross_val_score(clf, X_train, y_train, cv=kfold, scoring='accuracy')
#mae = cross_val_score(clf, X_train, y_train, cv=kfold, scoring='neg_mean_absolute_error')
#r2 = cross_val_score(clf, X_train, y_train, cv=kfold, scoring='r2')
#logloss = cross_val_score(clf, X_train, y_train, cv=kfold, scoring='neg_log_loss')
#auc = cross_val_score(clf, X_train, y_train, cv=kfold, scoring='roc_auc')
print "SVC Classifier Cross Validation Accuracy: " , '{:.2%}'.format(cross_val.mean()), "+/-",  '{:.2%}'.format(cross_val.std())
#print "Mean Absolute Error: ", '{:.2%}'.format(mae.mean()), "+/-",  '{:.2%}'.format(mae.std())
#print "R^2: ", '{:.2%}'.format(r2.mean()), "+/-",  '{:.2%}'.format(r2.std())
#print "Logarithmic Loss: ", '{:.2%}'.format(logloss.mean()), "+/-",  '{:.2%}'.format(logloss.std())
#print "Area Under Curve: ", '{:.2%}'.format(auc.mean()), "+/-",  '{:.2%}'.format(auc.std())
plot_roc('../Output/SVCroc.png', clf, X[:1191], y[:1191])


# Train Logistic Regression
clf = OneVsRestClassifier(LogisticRegression())
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
score = clf.score(X_test, y_test)
confusion = pd.crosstab(y_test, predictions)
normalized = confusion/confusion.sum(axis=1)
plot_confusion_matrix('../Output/Logisticcm.png', confusion)
plot_classification_report('../Output/Logisticcr.png', classification_report(predictions,y_test))
print "Logistic Regression Classifier Accuracy: ", '{:.2%}'.format(score)
cross_val = cross_val_score(clf, X_train, y_train, cv=kfold, scoring='accuracy')
mae = cross_val_score(clf, X_train, y_train, cv=kfold, scoring='neg_mean_absolute_error')
r2 = cross_val_score(clf, X_train, y_train, cv=kfold, scoring='r2')
logloss = cross_val_score(clf, X_train, y_train, cv=kfold, scoring='neg_log_loss')
#auc = cross_val_score(clf, X_train, y_train, cv=kfold, scoring='roc_auc')
print "Logistic Regression Classifier Cross Validation Accuracy: " , '{:.2%}'.format(cross_val.mean()), "+/-",  '{:.2%}'.format(cross_val.std())
print "Mean Absolute Error: ", '{:.2%}'.format(mae.mean()), "+/-",  '{:.2%}'.format(mae.std())
print "R^2: ", '{:.2%}'.format(r2.mean()), "+/-",  '{:.2%}'.format(r2.std())
print "Logarithmic Loss: ", '{:.2%}'.format(logloss.mean()), "+/-",  '{:.2%}'.format(logloss.std())
#print "Area Under Curve: ", '{:.2%}'.format(auc.mean()), "+/-",  '{:.2%}'.format(auc.std())
plot_roc('../Output/Logisticroc.png',clf, X[:1191], y[:1191])


# Train Bernoulli Naive Bayes
clf = OneVsRestClassifier(BernoulliNB())
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
score = clf.score(X_test, y_test)
confusion = pd.crosstab(y_test, predictions)
normalized = confusion/confusion.sum(axis=1)
plot_confusion_matrix('../Output/BNBcm.png', confusion)
plot_classification_report('../Output/BNBcr.png', classification_report(predictions,y_test))
print "Bernoulli Naive Bayes Classifier Accuracy: ", '{:.2%}'.format(score)
cross_val = cross_val_score(clf, X_train, y_train, cv=kfold, scoring='accuracy')
mae = cross_val_score(clf, X_train, y_train, cv=kfold, scoring='neg_mean_absolute_error')
r2 = cross_val_score(clf, X_train, y_train, cv=kfold, scoring='r2')
logloss = cross_val_score(clf, X_train, y_train, cv=kfold, scoring='neg_log_loss')
#auc = cross_val_score(clf, X_train, y_train, cv=kfold, scoring='roc_auc')
print "Bernoulli Naive Bayes Classifier Cross Validation Accuracy: " , '{:.2%}'.format(cross_val.mean()), "+/-",  '{:.2%}'.format(cross_val.std())
print "Mean Absolute Error: ", '{:.2%}'.format(mae.mean()), "+/-",  '{:.2%}'.format(mae.std())
print "R^2: ", '{:.2%}'.format(r2.mean()), "+/-",  '{:.2%}'.format(r2.std())
print "Logarithmic Loss: ", '{:.2%}'.format(logloss.mean()), "+/-",  '{:.2%}'.format(logloss.std())
#print "Area Under Curve: ", '{:.2%}'.format(auc.mean()), "+/-",  '{:.2%}'.format(auc.std())
plot_roc('../Output/BNBroc.png',clf, X[:1191], y[:1191])


# Train MLP using four layers...
clf = OneVsRestClassifier(MLPClassifier(hidden_layer_sizes=(20,20,20,20),max_iter=1000))
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
score = clf.score(X_test, y_test)
confusion = pd.crosstab(y_test, predictions)
normalized = confusion/confusion.sum(axis=1)
plot_confusion_matrix('../Output/MLPcm.png', confusion)
plot_classification_report('../Output/MLPcr.png', classification_report(predictions,y_test))
print "MLP 4 layers Classifier Accuracy: ", '{:.2%}'.format(score)
cross_val = cross_val_score(clf, X_train, y_train, cv=kfold, scoring='accuracy')
mae = cross_val_score(clf, X_train, y_train, cv=kfold, scoring='neg_mean_absolute_error')
r2 = cross_val_score(clf, X_train, y_train, cv=kfold, scoring='r2')
logloss = cross_val_score(clf, X_train, y_train, cv=kfold, scoring='neg_log_loss')
#auc = cross_val_score(clf, X_train, y_train, cv=kfold, scoring='roc_auc')
print "MLP 4 Layers Classifier Cross Validation Accuracy: " , '{:.2%}'.format(cross_val.mean()), "+/-",  '{:.2%}'.format(cross_val.std())
print "Mean Absolute Error: ", '{:.2%}'.format(mae.mean()), "+/-",  '{:.2%}'.format(mae.std())
print "R^2: ", '{:.2%}'.format(r2.mean()), "+/-",  '{:.2%}'.format(r2.std())
print "Logarithmic Loss: ", '{:.2%}'.format(logloss.mean()), "+/-",  '{:.2%}'.format(logloss.std())
#print "Area Under Curve: ", '{:.2%}'.format(auc.mean()), "+/-",  '{:.2%}'.format(auc.std())
plot_roc('../Output/MLProc.png',clf, X[:1191], y[:1191])



# Remove featureList duplicates
featureList = list(set(featureList))
# Extract feature vector for all tweets and split testing and training sets randomly
#random.shuffle(tweets)
training_set = nltk.classify.util.apply_features(extract_features, tweets[:1200])
testing_set = nltk.classify.util.apply_features(extract_features, tweets[1200:])

#linear SVC
LinearSVCClassifier = SklearnClassifier(LinearSVC())
LinearSVCClassifier.train(training_set)
print "Linear SVC Classifier Accuracy: %s" % '{:.2%}'.format(nltk.classify.accuracy(LinearSVCClassifier, testing_set))
cv = cross_validation.KFold(len(training_set), n_folds=10, shuffle=False, random_state=None)
print "Proceed by cross validation using 10 splits:"
i = 0
l = []
for traincv, testcv in cv:
    i = i + 1
    classifier = LinearSVCClassifier.train(training_set[traincv[0]:traincv[len(traincv)-1]])
    ac = nltk.classify.util.accuracy(classifier, training_set[testcv[0]:testcv[len(testcv)-1]])
    l.append(ac)
    print 'Split %s accuracy: '% i, ac
scores = np.array(l)
print "Accuracy from cross validation: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)

# Test the Naive Baiyes classifier
NBClassifier = nltk.NaiveBayesClassifier.train(training_set)
testTweet = 'US market just crashed economy is down'
processedTestTweet = processTweet(testTweet)
print "Show Results: \n"
print NBClassifier.classify(extract_features(getFeatureVector(processedTestTweet)))
print NBClassifier.show_most_informative_features(10)

#getting ready to return a normalized confusion matrix and classification report
for i in range(len(testing_set)):
    test_result.append(NBClassifier.classify(testing_set[i][0]))
    gold_result.append(testing_set[i][1])
y_actu = pd.Series(gold_result, name='Actual')
y_pred = pd.Series(test_result, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred)
df_normalized = df_confusion / df_confusion.sum(axis=1)
plot_confusion_matrix(df_normalized)
cr = classification_report(gold_result,test_result)
plot_classification_report(cr)
print "Naive Bayes Classifier Accuracy: %s" % '{:.2%}'.format(nltk.classify.accuracy(NBClassifier, testing_set))
print "Mean Absolute Error: %s" % '{:.2%}'.format(mean_absolute_error(gold_result, test_result))
print "Proceed by cross validation using 10 splits:"
i = 0
l = []
for traincv, testcv in cv:
    i = i + 1
    classifier = NBClassifier.train(training_set[traincv[0]:traincv[len(traincv)-1]])
    ac = nltk.classify.util.accuracy(classifier, training_set[testcv[0]:testcv[len(testcv)-1]])
    l.append(ac)
    print 'Split %s accuracy: '% i, ac
scores = np.array(l)
print "Accuracy from cross validation: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
new_gold = [el for el in gold_result if el != 0]
new_test = [el for el in test_result if el != 0]
n1 = len(new_gold)
n2 = len(new_test)
n = min(n1,n2)
average_precision = average_precision_score(new_gold[:n], new_test[:n])
precision, recall, _ = precision_recall_curve(new_gold[:n], new_test[:n])
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, facecolor='b', alpha=0.2, interpolate=True)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AUC={0:0.2f}'.format(
          average_precision))
plt.savefig('precision.png')


#best linear SVC ever...
#predictionsLinearSVC = LinearSVCClassifier.classify(X_test)
#print(confusion_matrix(y_test,predictionsLinearSVC))
#print(classification_report(y_test,predictionsLinearSVC))
#print "Mean Absolute Error: %s" % '{:.2%}'.format(mean_absolute_error(predictionsLinearSVC, y_test))


#Train the SVM classifiers
result = getSVMFeatureVectorAndLabels(tweets[:1200], featureList)
testResult = getSVMFeatureVectorAndLabels(tweets[1200:], featureList)
SVCClassifier = SklearnClassifier(SVC())
SVCClassifier.train(training_set)
test_result, gold_result = [], []
for i in range(len(testing_set)):
    test_result.append(SVCClassifier.classify(testing_set[i][0]))
    gold_result.append(testing_set[i][1])
print "SVC Classifier Accuracy: %s" % '{:.2%}'.format(nltk.classify.accuracy(SVCClassifier, testing_set))
print "Mean Absolute Error: %s" % '{:.2%}'.format(mean_absolute_error(gold_result, test_result))
print "Proceed by cross validation using 10 splits:"
i = 0
l = []
for traincv, testcv in cv:
    i = i + 1
    classifier = SVCClassifier.train(training_set[traincv[0]:traincv[len(traincv)-1]])
    ac = nltk.classify.util.accuracy(classifier, training_set[testcv[0]:testcv[len(testcv)-1]])
    l.append(ac)
    print 'Split %s accuracy: '% i, ac
scores = np.array(l)
print "Accuracy from cross validation: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)


#Train the Logistic
LogisticClassifier = SklearnClassifier(LogisticRegression())
LogisticClassifier.train(training_set)
test_result, gold_result = [], []
for i in range(len(testing_set)):
    test_result.append(LogisticClassifier.classify(testing_set[i][0]))
    gold_result.append(testing_set[i][1])
print "Logistic Classifier Accuracy: %s" % '{:.2%}'.format(nltk.classify.accuracy(LogisticClassifier, testing_set))
print "Mean Absolute Error: %s" % '{:.2%}'.format(mean_absolute_error(gold_result, test_result))
print "Proceed by cross validation using 10 splits:"
i = 0
l = []
for traincv, testcv in cv:
    i = i + 1
    classifier = LogisticClassifier.train(training_set[traincv[0]:traincv[len(traincv)-1]])
    ac = nltk.classify.util.accuracy(classifier, training_set[testcv[0]:testcv[len(testcv)-1]])
    l.append(ac)
    print 'Split %s accuracy: '% i, ac
scores = np.array(l)
print "Accuracy from cross validation: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)

#Bernoulli
BNBClassifier = SklearnClassifier(BernoulliNB())
BNBClassifier.train(training_set)
test_result, gold_result = [], []
for i in range(len(testing_set)):
    test_result.append(BNBClassifier.classify(testing_set[i][0]))
    gold_result.append(testing_set[i][1])
print "Bernoulli Naive Bayes Classifier Accuracy: %s" % '{:.2%}'.format(nltk.classify.accuracy(BNBClassifier, testing_set))
print "Mean Absolute Error: %s" % '{:.2%}'.format(mean_absolute_error(gold_result, test_result))
print "Proceed by cross validation using 10 splits:"
i = 0
l = []
for traincv, testcv in cv:
    i = i + 1
    classifier = BNBClassifier.train(training_set[traincv[0]:traincv[len(traincv)-1]])
    ac = nltk.classify.util.accuracy(classifier, training_set[testcv[0]:testcv[len(testcv)-1]])
    l.append(ac)
    print 'Split %s accuracy: '% i, ac
scores = np.array(l)
print "Accuracy from cross validation: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)


# multi binomial..
MNBClassifier = SklearnClassifier(MultinomialNB())
MNBClassifier.train(training_set)
test_result, gold_result = [], []
for i in range(len(testing_set)):
    test_result.append(MNBClassifier.classify(testing_set[i][0]))
    gold_result.append(testing_set[i][1])
print "Multinomial Naive Bayes Classifier Accuracy: %s" % '{:.2%}'.format(nltk.classify.accuracy(MNBClassifier, testing_set))
print "Mean Absolute Error: %s" % '{:.2%}'.format(mean_absolute_error(gold_result, test_result))
print "Proceed by cross validation using 10 splits:"
i = 0
l = []
for traincv, testcv in cv:
    i = i + 1
    classifier = MNBClassifier.train(training_set[traincv[0]:traincv[len(traincv)-1]])
    ac = nltk.classify.util.accuracy(classifier, training_set[testcv[0]:testcv[len(testcv)-1]])
    l.append(ac)
    print 'Split %s accuracy: '% i, ac
scores = np.array(l)
print "Accuracy from cross validation: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)

# Test the Maximum Entropy classifier
MaxEntClassifier = nltk.classify.maxent.MaxentClassifier.train(training_set, 'GIS', max_iter = 10)
processedTestTweet = processTweet(testTweet)
print MaxEntClassifier.classify(extract_features(getFeatureVector(processedTestTweet)))
print MaxEntClassifier.show_most_informative_features(10)
print "Maximum Entropy Model Accuracy: %s" % '{:.2%}'.format(nltk.classify.accuracy(MaxEntClassifier, testing_set))