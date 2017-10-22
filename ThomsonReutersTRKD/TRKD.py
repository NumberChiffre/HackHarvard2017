"""
Created by Terence Liu, llebox@hotmail.com
"""
import requests, json
import datetime as dt
from tools import *
import pandas as pd
import numpy as np
import sys, time
from os import path
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from pandas_datareader import data

# Lets user generate new auth token with loggin info
class ThomsonReuterAccInfo():
    
    def __init__(self, username, password, appid):
        self.username = username
        self.password = password
        self.appid = appid
        self.headers = {'content-type': 'application/json;charset=utf-8'}
    
    def getAuthHeaders(self):
        return self.headers
    
    def getAuthData(self):
        return {'CreateServiceToken_Request_1': {'ApplicationID': self.appid, 'Username': self.username, 'Password': self.password}}
    
    def getAuthURL(self):
        return 'https://api.trkd.thomsonreuters.com/api/TokenManagement/TokenManagement.svc/REST/Anonymous/TokenManagement_1/CreateServiceToken_1'#'https://www.trkd.thomsonreuters.com/SupportSite/TestApi/Op?svc=TokenManagement_1&op=CreateServiceToken_1'
    

    
# main class for loading historical prices
# TODO: 
# Create base class and allow inheritance for sub classes
# Save json into dataframe
# Can have a base class which serves as requesting data
# Can have wrapper to use multithreading for a list of symbols
# Must decide what to do with each set of dataframe, separate or combine

class ThomsonReuterURLReader():
    
    # set up to handle RESTful API calls
    def __init__(self, symbol=None, start=None, end=None, interval="DAILY", timeout=0.0001, retry_count=5, session=None):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.interval = interval.upper()
        self.timeout = timeout
        self.retry_count = retry_count
        self.session = requests.Session()
        self.headers = {'X-Trkd-Auth-Token': 'D5748CF607B5E520DD8C90E2D7EFEBEF630FD8F889F8001BBB6E4C03385C9DEB177C0ED01BACAA3B77F295F651B2B11E80192604954CD2AB0152477273D69544BF354EB6356D90D570700D10A0110C1DB31BD3C70C3A11D5CA1CD1E4E1776409', 'X-Trkd-Auth-ApplicationID': 'trkddemoappwm', 'Content-Type': 'application/json'}
        self.data_definition = path.expanduser('~/Documents/Projects/prometheus/ThomsonReutersAPI/data_definitions.json')
        
        # check if interval is properly set
        # in API calls, interval string must be capital
        if self.interval.lower() not in ['minute', '5minutes', '30minutes', 'hour','daily', 'weekly', 'monthly', 'quarterly', 'annual']:
            raise ValueError("Invalid interval: ", self.interval, ", valid interval values include: daily, weekly, monthly, quarterly, annual")
    
    def close(self):
        self.session.close()
    
    def setHeadersToken(self, newToken):
        self.headers['X-Trkd-Auth-Token'] = newToken
    
    def getHeaders(self):
        return self.headers    
    
    def base_search_url(self):
        return str("http://api.trkd.thomsonreuters.com/api/Search/Search.svc/REST")
    
    def base_timeseries_url(self):
        return str("http://api.trkd.thomsonreuters.com/api/TimeSeries/TimeSeries.svc/REST")    
    
    def fund_quote_url(self):
        return self.base_search_url() + str("/FundQuote_1/GetFundQuote_1")
    
    def equity_quote_url(self):
        return self.base_search_url() + str("/EquityQuote_1/GetEquityQuote_1")
    
    def historical_prices_url(self):
        return self.base_timeseries_url() + str("/TimeSeries_1/GetInterdayTimeSeries_4")
    
    def intraday_prices_url(self):
        return self.base_timeseries_url() + str("/TimeSeries_1/GetIntradayTimeSeries_4")
        
    
    # finds by array index and dictionary key
    def setFundQuoteJSON(self):
        with open(self.data_definition, 'r') as file:
            df = json.load(file)["fund_quote"]
            df["GetFundQuote_Request_1"]['Query'][-1]['TickerSymbol']['SymbolValue'][-1]['Value'] = self.symbol
        return df
    
    
    # finds by array index and dictionary key
    def setEquityQuoteJSON(self):
        with open(self.data_definition, 'r') as file:
            df = json.load(file)["equity_quote"]
            df["GetEquityQuote_Request_1"]['Query'][-1]['TickerSymbol']['SymbolValue'][-1]['Value'] = self.symbol
        return df
        
    
    # GetInterdayTimeSeries_Request_4 is the field to request historical daily data
    # Daily parameter can be changed later
    def setHistoricalPricesJSON(self):
        with open(self.data_definition, 'r') as file:
            df = json.load(file)["hist_prices"]
            df["GetInterdayTimeSeries_Request_4"]['StartTime'] = self.start.strftime("%Y-%m-%d") + "T00:00:00"
            df["GetInterdayTimeSeries_Request_4"]['EndTime'] = self.end.strftime("%Y-%m-%d") + "T23:59:00"
            df["GetInterdayTimeSeries_Request_4"]['Interval'] = self.interval 
            df["GetInterdayTimeSeries_Request_4"]['Symbol'] = self.symbol
        return df
    
    
    # Intraday use Minute, Hour, 5Minutes, 30Minutes
    def setIntradayPricesJSON(self):
        with open(self.data_definition, 'r') as file:
            df = json.load(file)["intraday_prices"]
            df["GetIntradayTimeSeries_Request_4"]['StartTime'] = self.start.strftime("%Y-%m-%d") + "T00:00:00"
            df["GetIntradayTimeSeries_Request_4"]['EndTime'] = self.end.strftime("%Y-%m-%d") + "T23:59:00"
            df["GetIntradayTimeSeries_Request_4"]['Interval'] = self.interval
            df["GetIntradayTimeSeries_Request_4"]['Symbol'] = self.symbol 
            print(df)
        return df
          
    
    # allow multiple retries for query
    # generates new auth token, update headers
    # return POST request given total url, json data and headers
    def getPOSTResponse(self, url, json_data, headers):
        self.generateNewAuth()
        for i in range(self.retry_count):
            output = self.session.post(url=url, json=json_data, headers=headers)
            if output.status_code == requests.codes.ok:
                print(output.json(), output.status_code, requests.codes.ok)
                return output
            
            # set a pause, increase pause, and do POST again
            time.sleep(self.timeout)
            self.timeout *= self.timeout * 10
    
    
    # for test API
    # generate new token and set it to our headers
    def generateNewAuth(self):
        with open('config.json', 'r') as output:
            file = json.load(output)
            accInfo = ThomsonReuterAccInfo(file["Username"], file["Password"] ,file["ApplicationID"])
            output = requests.post(url=accInfo.getAuthURL(), data=json.dumps(accInfo.getAuthData()), headers=accInfo.getAuthHeaders())
            self.setHeadersToken(output.json()['CreateServiceToken_Response_1']['Token'])
    
        
    # specifically works for fund quote POST
    # outputs TR Ticker that will be used to obtain historical prices
    def getFundQuoteData(self):
        output = self.getPOSTResponse(self.fund_quote_url(), self.setFundQuoteJSON(), self.getHeaders())
        data = output.json()
        return data['GetFundQuote_Response_1']['Result']['Hit'][0]['RIC']
    
    
    # specifically works for equity quote POST
    def getEquityQuoteData(self):
        output = self.getPOSTResponse(self.equity_quote_url(), self.setEquityQuoteJSON(), self.getHeaders())
        data = output.json()
        return data['GetEquityQuote_Response_1']['Result']['Hit'][0]['RIC']

    
    # Returns a dataframe containing historical prices
    def getHistoricalPrices(self):
        total_df = None
        try:
            output = self.getPOSTResponse(self.historical_prices_url(), self.setHistoricalPricesJSON(), self.getHeaders())
            data = output.json()   
            total_df = pd.DataFrame.from_dict(data['GetInterdayTimeSeries_Response_4']['Row'])
            total_df['TIMESTAMP'] = total_df['TIMESTAMP'].map(lambda x: x.split("T", 1)[0])
            total_df.set_index('TIMESTAMP', inplace=True) 
            total_df.index = pd.to_datetime(total_df.index)
        except Exception as e:
            raise("ValueError JSON Response: getHistoricalPrices", e.args)
        finally:
            return total_df
    
    
    # Returns a dataframe containing intraday prices for a given timeframe    
    def getIntradayHistoricalPrices(self):
        total_df = None
        try:
            output = self.getPOSTResponse(self.intraday_prices_url(), self.setIntradayPricesJSON(), self.getHeaders())
            data = output.json()
            print(data)
            total_df = pd.DataFrame.from_dict(data['GetIntradayTimeSeries_Response_4']['R'])
            total_df.set_index('TIMESTAMP', inplace=True)
            total_df.index = pd.to_datetime(total_df.index)
        except Exception as e:
            raise("ValueError JSON Response: getIntradayHistoricalPrices", e.args)
        finally:
            return total_df
    
    # get historical prices for multiple tickers
    def getMultiHistoricalPrices(self):
        pass
    
    
# Calls ThomsonReuterURLReader class for specific methods
def TRDataReader(symbol, method, start=None, end=None):
    # temporary iterator just to run map iterators
    if method == 'hist_prices':
        return ThomsonReuterURLReader(symbol, start, end).getHistoricalPrices()
    elif method == 'intraday_prices':
        return ThomsonReuterURLReader(symbol, start, end, interval = "MINUTE").getIntradayHistoricalPrices()
    elif method == 'fund_quote':
        return ThomsonReuterURLReader(symbol).getFundQuoteData()
    elif method == 'equity_quote':
        return ThomsonReuterURLReader(symbol).getEquityQuoteData()


# compare speed of yahoo API vs Trkd API
def compare_execution_speed():
    t1, t2 = [], []
    for i in range(40):
        starttime = time.time()  
        with ThreadPoolExecutor(max_workers=20) as pool:
            pool.map(data.DataReader, repeat('XIU.TO'), repeat('yahoo'), repeat(dt.datetime(2014,1,3)), repeat(dt.datetime(dt.datetime.now().year,dt.datetime.now().month,dt.datetime.now().day)), range(20))
            pool.shutdown()         
        diff = time.time() - starttime
        t1.append(diff)
        print('Time taken for Yahoo Historical Daily Data ', round(diff, 2), ' seconds')
        
        starttime2 = time.time()   
        with ThreadPoolExecutor(max_workers=20) as pool:
            pool.map(TRDataReader, range(20), repeat('XIU.TO'), repeat('hist_daily'), repeat(dt.datetime(2014,1,3)), repeat(dt.datetime(dt.datetime.now().year,dt.datetime.now().month,dt.datetime.now().day)))
            pool.shutdown()  
        diff2 = time.time() - starttime2
        t2.append(diff2)
        print('Time taken for TR Historical Daily Data ', round(diff2, 2), ' seconds')
    
    print("Yahoo AVG TIME: ", round(np.mean(t1), 2), " seconds")
    print("TR AVG TIME: ", round(np.mean(t2), 2), " seconds")


def testingQueries():
    df = TRDataReader('SPY', 'intraday_prices', dt.datetime(2017,6,6), dt.datetime(2017,6,15))
    print(df)  
    with pd.ExcelWriter('Intraday_Prices.xlsx') as output:
        df.to_excel(output, 'sheet1')
        output.save()
    #df2 = TRDataReader("BBD.B", 'equity_quote', dt.datetime(2007,1,3), dt.datetime(dt.datetime.now().year,dt.datetime.now().month,dt.datetime.now().day))
    #print(df2)
   
    #df3 = TRDataReader("BBD.B", 'intraday_prices', dt.datetime(2017,10,13), dt.datetime(dt.datetime.now().year,dt.datetime.now().month,dt.datetime.now().day))
    #print(df3)   
    
    """
    # test for multiple queries
    symbols = ['AZN','PBD', 'XSH', 'XIC']
    i = 0
    for i in range(len(symbols)):
        symbols[i] = TRDataReader(symbols[i], 'equity_quote')
        
    for symbol in symbols:
        df1 = TRDataReader(symbol, 'hist_prices', dt.datetime(2007,1,3), dt.datetime(dt.datetime.now().year,dt.datetime.now().month,dt.datetime.now().day))
        print(df1)     
    """
    
if __name__=="__main__":  
    #compare_execution_speed()
    testingQueries()