import tensorflow as tf
from tensorflow.python.framework import ops
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import unicodedata
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing as mp
from itertools import repeat


# uses tensorflow's rnn modules
# used twitter API, Bitstamp API, and built a Thomson Reuters trkd API to parse historical minute data
class RNN():
    
    def __init__(self, num_epochs=500, batch_size=1, total_series_length=0, truncated_backprop_length=3, state_size=10, num_features=6, num_batches=2, num_classes=1, min_test_size=100):
                
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.total_series_length = total_series_length
        self.truncated_backprop_length = truncated_backprop_length
        self.state_size = state_size
        self.num_features = num_features
        self.num_batches = total_series_length//batch_size//truncated_backprop_length
        self.num_classes = num_classes
        self.min_test_size = min_test_size
        self.data_source = '../Data/bitcoinparser2MFilteredEqualFinal.csv'
        self.data_encoding = "ISO-8859-1"
        
        
    def loadRawData(self):
        df = pd.read_csv(self.data_source,encoding = self.data_encoding)
        self.total_series_length = len(df.index)
        
        #define dataset
        new_df = pd.DataFrame()
        new_df[['Closing', 'USDJPY_Closing', 'Volume', 'Label']] = df[['Closing', 'USDJPY_Closing', 'Volume', 'Label']] 
        df['Username'] = df['Timestamp']        
        
        #use nltk to transform into floats
        #nltk.download_shell()
        df.set_index('Username', inplace=True)
        df.index = pd.to_datetime(df.index)
        df = df.reset_index()
        new_df['compound'] = ''
        new_df['neg'] = ''
        new_df['pos'] = ''
        new_df['neu'] = ''
        new_df['Spread'] = df['Ask'] - df['Bid'] 
        new_df['Ratio'] = np.log(df['Closing']) - np.log(df['USDJPY_Closing'])
        return df, new_df
    
    
    def normalizeRawData(self, df, new_df):
        sentiment = SentimentIntensityAnalyzer()
        for date, row in df.T.iteritems():
            try:
                sentence = unicodedata.normalize('NFKD', df.loc[date, 'FilteredTweet'])
                score = sentiment.polarity_scores(sentence)
                new_df.set_value(date, 'compound', score['compound'])
                new_df.set_value(date, 'neg', score['neg'])
                new_df.set_value(date, 'neu', score['neu'])
                new_df.set_value(date, 'pos', score['pos'])
               
            except TypeError:
                print (df.loc[date, 'FilteredTweet'])
                print (date)   
        datasetNorm = (new_df - new_df.mean()) / (new_df.max() - new_df.min())
        datasetNorm['Next_Closing'] = datasetNorm['Closing'].shift(-1)
        datasetNorm.reset_index(inplace=True)
        del datasetNorm['index']
        return datasetNorm
    
    
    def splitTrainTestData(self, datasetNorm):
        self.num_batches = self.total_series_length//self.batch_size//self.truncated_backprop_length
        datasetTrain = datasetNorm[datasetNorm.index < self.num_batches*self.batch_size*self.truncated_backprop_length]
        
        for i in range(self.min_test_size,len(datasetNorm.index)):    
            if(i % self.truncated_backprop_length*self.batch_size == 0):
                test_first_idx = len(datasetNorm.index)-i
                break
        datasetTest =  datasetNorm[datasetNorm.index >= test_first_idx]
        # determine training set
        xTrain, yTrain = datasetTrain[['Closing', 'Ratio', 'Spread', 'USDJPY_Closing', 'Volume', 'compound', 'neu', 'neg', 'pos']].as_matrix(), datasetTrain['Next_Closing'].as_matrix()
        xTest, yTest = datasetTest[['Closing', 'Ratio', 'Spread', 'USDJPY_Closing', 'Volume', 'compound', 'neu', 'neg', 'pos']].as_matrix(), datasetTest['Next_Closing'].as_matrix()     
        return xTrain, yTrain, xTest, yTest
    
    
    def executeRNN(self, xTrain, yTrain, xTest, yTest, const=0.1, learning_rate=0.001):
        #define batchholders
        print("Creating batch holders")
        batchX_placeholder = tf.placeholder(dtype=tf.float32,shape=[None,self.truncated_backprop_length, self.num_features], name='data_ph')
        batchY_placeholder = tf.placeholder(dtype=tf.float32,shape=[None,self.truncated_backprop_length, self.num_classes], name='target_ph')
        
        #2 initialize pairs of w and b series randomly
        print("Initialize variables for RNN")
        W2 = tf.Variable(initial_value=np.random.rand(self.state_size,self.num_classes),dtype=tf.float32)
        b2 = tf.Variable(initial_value=np.random.rand(1,self.num_classes),dtype=tf.float32)
        
        # determine state series
        # use RNN classifier 
        labels_series = tf.unstack(batchY_placeholder, axis=1)
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.state_size)
        states_series, current_state = tf.nn.dynamic_rnn(cell=cell,inputs=batchX_placeholder,dtype=tf.float32)
        states_series = tf.transpose(states_series,[1,0,2])
        
        # Adapt learning rate and set up final variables
        # Apply backward feed
        last_state = tf.gather(params=states_series,indices=states_series.get_shape()[0]-1)
        last_label = tf.gather(params=labels_series,indices=len(labels_series)-1)
        weight = tf.Variable(tf.truncated_normal([self.state_size,self.num_classes]))
        bias = tf.Variable(tf.constant(const,shape=[self.num_classes]))
        prediction = tf.matmul(last_state,weight) + bias
        loss = tf.reduce_mean(tf.squared_difference(last_label,prediction)) 
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        
        # start train/testing each epoch
        loss_list, test_pred_list, test_pred_total = [], [], []
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            
            # nested iteration through number of epochs and batches for each step
            for epoch_idx in range(self.num_epochs):
                print('Training for Epoch %d' % epoch_idx)
                for batch_idx in range(self.num_batches):
                    start_idx = batch_idx * self.truncated_backprop_length
                    end_idx = start_idx + self.truncated_backprop_length * self.batch_size
                    batchX = xTrain[start_idx:end_idx,:].reshape(self.batch_size,self.truncated_backprop_length,self.num_features)
                    batchY = yTrain[start_idx:end_idx].reshape(self.batch_size,self.truncated_backprop_length,1)
                    feed = {batchX_placeholder : batchX, batchY_placeholder : batchY}
                    
                    # Train our datafeed through iterated batches
                    _loss,_train_step,_pred,_last_label,_prediction = sess.run(
                        fetches=[loss,train_step,prediction,last_label,prediction],
                        feed_dict = feed
                    )
                    loss_list.append(_loss)
                    
                    if(batch_idx % self.min_test_size == 0):
                        print('At step %d - Loss RNN: %.6f' % (batch_idx,_loss))
                        
            # Ready for output
            print('Testing between BTC Historical Prices vs RNN Predictions')
            for test_idx in range(len(xTest) - self.truncated_backprop_length):
                testBatchX = xTest[test_idx:test_idx+self.truncated_backprop_length,:].reshape((1,self.truncated_backprop_length,self.num_features))    
                testBatchY = yTest[test_idx:test_idx+self.truncated_backprop_length].reshape((1,self.truncated_backprop_length,1))
                feed = {batchX_placeholder : testBatchX,
                    batchY_placeholder : testBatchY}
                _last_state,_last_label,test_pred = sess.run([last_state,last_label,prediction],feed_dict=feed)
                test_pred_list.append(test_pred[0][0])
        return yTest, test_pred_list, loss_list
    
    
    # use multiprocessing to deal with nested for loop
    # TODO: deal with thread and session init with tensorflow
    def multiprocess_loss(self, epoch_idx, sess, loss, train_step, prediction, last_label, batchX_placeholder, batchY_placeholder, loss_list, xTrain, yTrain, xTest, yTest):
        
        # nested iteration through number of epochs and batches for each step
        print('Training for Epoch %d' % epoch_idx)
        for batch_idx in range(self.num_batches):
            start_idx = batch_idx * self.truncated_backprop_length
            end_idx = start_idx + self.truncated_backprop_length * self.batch_size
            batchX = xTrain[start_idx:end_idx,:].reshape(self.batch_size,self.truncated_backprop_length,self.num_features)
            batchY = yTrain[start_idx:end_idx].reshape(self.batch_size,self.truncated_backprop_length,1)
            feed = {batchX_placeholder : batchX, batchY_placeholder : batchY}
            
            # Train our datafeed through iterated batches
            _loss,_train_step,_pred,_last_label,_prediction = sess.run(
                fetches=[loss,train_step,prediction,last_label,prediction],
                feed_dict = feed
            )
            loss_list.append(_loss)
            
            # output loss error at each batch step
            if(batch_idx % self.min_test_size == 0):
                print('Step %d - Loss: %.6f' % (batch_idx,_loss))
                
        return loss_list
    

if __name__ == "__main__":
    rnn = RNN(num_epochs=10, num_features=9)
    df, new_df = rnn.loadRawData()
    datasetNorm = rnn.normalizeRawData(df, new_df)
    xTrain, yTrain, xTest, yTest = rnn.splitTrainTestData(datasetNorm)
    yTest, test_pred_list, loss_list = rnn.executeRNN(xTrain, yTrain, xTest, yTest)
    
    # generate quick and dirty plot for predictions and error loss
    plt.figure(figsize=(21,7))
    plt.xlim(0, len(test_pred_list)-1)
    plt.plot(yTest[:len(test_pred_list)],label='Price of BTC', color='blue')
    plt.plot(test_pred_list,label='RNN Prediction', color='red')
    plt.title('BTC Actual Historical Prices vs RNN Predicted Prices [Normalized]')
    plt.legend(loc='upper right')
    plt.savefig('../Output/RNNPredict.png')
    plt.clf()
    
    plt.title('Loss Function [RNN]')
    plt.xlim(0, len(test_pred_list)-1)
    plt.scatter(x=np.arange(0,len(loss_list)), y=loss_list)
    plt.xlabel('Epochs Size')
    plt.ylabel('Loss')
    plt.savefig('../Output/RNNLoss.png')
    print("End of RNN")    