# HackHarvard2017
Hudson Rivers Trading Challenge 2017


"Interesting Use of Data! from Hudson River Trading
Any project that gathers and uses data to infer non-obvious relationships is eligible. Some concrete examples would be twists on collaborative filtering, or (for example), identifying shifts in a corpus over time. Prize: Oculus Rift for each team member!"

- Using Bitcoin Tweets, BTC Closing Prices, USDJPY Closing Prices, BTC Volume, BTC/USDJPY Log Ratio to train a set of algorithms to find which are the most accurate models
- Tweets are collected through Twitter API, stored into MySQL
- Applied bag of words/tf-idf vectorizer and scaler + feature extraction
- Compared each ML algorithm based on confusion matrix, classification report, ROC/AUC curves, Learning Curves, and both applying stratified k-fold cross-validation
- Constructed a pseudo Bitcoin Bullishness indicator and apply statistical tests for correlation, cointegration including ADF and Granger Causality Analysis vs the daily returns of BTC
- Applied Recursive Neural Nets to the final section to show actual BTC prices vs RNN predicted series


# Checkout HackHarvard branch!

