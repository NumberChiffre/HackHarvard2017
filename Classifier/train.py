from preprocess import *
from sklearn import model_selection, cross_validation
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing, svm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.python.framework import ops
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import unicodedata
import pandas as pd

# Not used, already implemented in libraries..
class MultiColumnLabelEncoder:
    def __init__(self, encoder, columns = None):
        self.columns = columns
        self.encoder = encoder
        
    def fit(self,X,y=None):
        return self

    def transform(self,X):
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

# read from file containing our data
df = pd.read_csv('../Data/bitcoinparser2MFilteredEqualRNN.csv',encoding = "ISO-8859-1")
X = df[['FilteredTweet', 'Closing', 'VWAP', 'Volume']]
y = df['Label']

# set the label encoders
vectorizer = CountVectorizer(analyzer=text_process, stop_words="english", min_df=2, max_df = 0.7)
transformer = TfidfVectorizer(analyzer=text_process, stop_words="english",min_df=2, use_idf =True, sublinear_tf=True)
svd = TruncatedSVD(n_components=200)
mb = MultiLabelBinarizer()
le = preprocessing.LabelEncoder()
lx = MultiLabelBinarizer()
st = StandardScaler()

#spliting train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# applying encoding to our text features into floats and normalize our numerical features
t_full = transformer.fit_transform(X['FilteredTweet'])
t_train = transformer.fit_transform(X_train['FilteredTweet'])
t_test = transformer.transform(X_test['FilteredTweet'])
f_full = st.fit_transform(X[['Closing', 'Volume']])
f_train = st.fit_transform(X_train[['Closing', 'Volume']])
f_test = st.transform(X_test[['Closing','Volume']])
X, X_train, X_test = t_full, t_train, t_test

# combine countvectorizer column with standard scaler..
X =  np.hstack((t_full.todense(),f_full))
X_train = np.hstack((t_train.todense(),f_train))
X_test = np.hstack((t_test.todense(),f_test))

# Select cross validation method
kfold = model_selection.StratifiedKFold(n_splits=10, random_state=10)

# Train SVC(Linear) and outputs results...X[:1191], y[:1191]
title = 'Learning Curve [Support Vector Machine Classifier]'
name = '[Support Vector Machine Classifier]'
clf = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True)).fit(X_train, y_train)
predictions = clf.predict(X_test)
#plot_svm('../Output/SVCLinearchart.png', clf, X_train, y_train, 'Linear SVC')
score = clf.score(X_test, y_test) 
confusion = pd.crosstab(y_test, predictions)
normalized = confusion/confusion.sum(axis=1)
plot_confusion_matrix('../Output/SVCLinearcm.png', confusion, name)
plot_classification_report('../Output/SVCLinearcr.png', classification_report(predictions,y_test), name)
print("SVC Linear Classifier Accuracy: ", '{:.2%}'.format(score))
cross_val = cross_val_score(clf, X_train, y_train, cv=kfold, scoring='accuracy')
print("SVC Linear Classifier Cross Validation Accuracy: " , '{:.2%}'.format(cross_val.mean()), "+/-",  '{:.2%}'.format(cross_val.std()))
plot_roc('../Output/SVCLinearroc.png', clf, kfold, X, y, name)
plot_learning_curve('../Output/SVCLinearlearning.png', clf, title,  X[:1191], y[:1191], ylim=(0.4, 1.01), cv=kfold, n_jobs=4)


# Train Logistic Regression
title = 'Learning Curve [Logistic Regression]'
name = '[Logistic Regression]'
clf = OneVsRestClassifier(LogisticRegression())
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
score = clf.score(X_test, y_test)
confusion = pd.crosstab(y_test, predictions)
normalized = confusion/confusion.sum(axis=1)
plot_confusion_matrix('../Output/Logisticcm.png', confusion, name)
plot_classification_report('../Output/Logisticcr.png', classification_report(predictions,y_test), name)
print ("Logistic Regression Classifier Accuracy: ", '{:.2%}'.format(score))
cross_val = cross_val_score(clf, X_train, y_train, cv=kfold, scoring='accuracy')
print ("Logistic Regression Classifier Cross Validation Accuracy: " , '{:.2%}'.format(cross_val.mean()), "+/-",  '{:.2%}'.format(cross_val.std()))
plot_roc('../Output/Logisticroc.png', clf, kfold, X[:1191], y[:1191], name)
plot_learning_curve('../Output/Logisticlearning.png', clf, title, X[:1191], y[:1191], ylim=(0.4, 1.01), cv=kfold, n_jobs=4)



# Train Bernoulli Naive Bayes
title = 'Learning Curve [Multinomial Naive Bayes]'
name = '[Multinomial Naive Bayes]'
clf = OneVsRestClassifier(MultinomialNB())
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
score = clf.score(X_test, y_test)
confusion = pd.crosstab(y_test, predictions)
normalized = confusion/confusion.sum(axis=1)
plot_confusion_matrix('../Output/MNBcm.png', confusion, name)
plot_classification_report('../Output/MNBcr.png', classification_report(predictions,y_test), name)
print ("Multinomial Naive Bayes Classifier Accuracy: ", '{:.2%}'.format(score))
cross_val = cross_val_score(clf, X_train, y_train, cv=kfold, scoring='accuracy')
print ("Multinomial Naive Bayes Classifier Cross Validation Accuracy: " , '{:.2%}'.format(cross_val.mean()), "+/-",  '{:.2%}'.format(cross_val.std()))
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=10)
plot_roc('../Output/MNBroc.png', clf, kfold, X[:1191], y[:1191], name)
plot_learning_curve('../Output/MNBlearning.png', clf, title, X[:1191], y[:1191], ylim=(0.4, 1.01), cv=kfold, n_jobs=4)


# Train Bernoulli Naive Bayes
title = 'Learning Curve [Bernoulli Naive Bayes]'
name =  '[Bernoulli Naive Bayes]'
clf = OneVsRestClassifier(BernoulliNB())
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
score = clf.score(X_test, y_test)
confusion = pd.crosstab(y_test, predictions)
normalized = confusion/confusion.sum(axis=1)
plot_confusion_matrix('../Output/BNBcm.png', confusion, name)
plot_classification_report('../Output/BNBcr.png', classification_report(predictions,y_test), name)
print ("Bernoulli Naive Bayes Classifier Accuracy: ", '{:.2%}'.format(score))
cross_val = cross_val_score(clf, X_train, y_train, cv=kfold, scoring='accuracy')
print ("Bernoulli Naive Bayes Classifier Cross Validation Accuracy: " , '{:.2%}'.format(cross_val.mean()), "+/-",  '{:.2%}'.format(cross_val.std()))

cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=10)
plot_roc('../Output/BNBroc.png', clf, kfold, X[:1191], y[:1191], name)
plot_learning_curve('../Output/BNBlearning.png', clf, title,  X,y,  ylim=(0.4, 1.01), cv=cv, n_jobs=4)

# Train MLP using four layers...
title = 'Learning Curve [Neural Networks Classifier]'
name = '[Neural Networks Classifier]'
clf = OneVsRestClassifier(MLPClassifier(activation='logistic',solver='lbfgs',hidden_layer_sizes=(10,5),alpha=1e-1))
clf.fit(X_train,y_train)
predictions = clf.predict(X_test)
score = clf.score(X_test, y_test)
confusion = pd.crosstab(y_test, predictions)
normalized = confusion/confusion.sum(axis=1)
plot_confusion_matrix('../Output/MLPcm.png', confusion, name)
plot_classification_report('../Output/MLPcr.png', classification_report(predictions,y_test), name)
print ("MLP 4 layers Classifier Accuracy: ", '{:.2%}'.format(score))
cross_val = cross_val_score(clf, X_train, y_train, cv=kfold, scoring='accuracy')
print ("MLP 4 Layers Classifier Cross Validation Accuracy: " , '{:.2%}'.format(cross_val.mean()), "+/-",  '{:.2%}'.format(cross_val.std()))
plot_roc('../Output/MLProc.png', clf, kfold, X, y, name)
