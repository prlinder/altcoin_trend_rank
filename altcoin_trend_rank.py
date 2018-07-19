
# Program: altcoin_trend_rank.py

'''
Coinmarketcap_com coin download and analysis

This notebook uses "coinmarketcap.com" to download coin data. 
    1) It uses the documented API to download a specified number of coins ranked according to the highest market cap
    2) It uses the undocumented graphing URLs to download slightly less than the last 30 days of data about those coins with a 15-minute data resolution.
    
Then a data summary dataframe is created which contains the pricing data for each coin quoted in both BTC and USD$ about 15 minutes ago, one hour ago, one day ago, 7 days ago, and slightly less than one month ago, along with calculated percentage changes from the present BTC and USD$ prices to those historical prices.

The coin data is then ranked according to BTC or USD$ price changes over the chosen interval and printed.

This is useful for keeping track of how much coins are changing in price and which coins are the big movers
'''

#-----------------------------
import os
import sys
import numpy as np
import pandas as pd
import pickle
import time
from datetime import datetime, timedelta
import json

#Third Party
import requests     # we need the whole module in order to get at "requests.packages.urllib3"; also requests.post
from requests.exceptions import RequestException
from urllib.parse import urlencode as _urlencode
from retry import retry     # https://pypi.python.orgpypi/retry/       https://github.com/invl/retry
    # Note that the module above is general purpose. But you counld also write your own routine for requests only around "urllib3.util.retry".
    #   See https://urllib3.readthedocs.io/en/latest/reference/urllib3.util.html#module-urllib3.util.retry
    #   and https://www.peterbe.com/plog/best-practice-with-retries-with-requests

from retry import retry
from retry.api import retry_call
    # for info see "https://pypi.python.org/pypi/retry/" & "https://github.com/invl/retry"

https_timeout = 10
# http_timeout = int time in sec to wait for an api response (otherwise 'requests.exceptions.Timeout' is raised)
#
# Retry Example Usage- precede the function containing the calls to retry with:
#   @retry(requests.exceptions.RequestException, tries=4, delay=10, backoff=4, logger=mylogger)

'''
A discussion of pricing per US$ and per BTC on Coinmarketcap_com and elsewhere
------------------------------------------------------------------------------

-On the coinmarketcap_com website, prices for the alcoins are displayed in US$, and a drop-down menu allows the choice
 to display the price in BTC or ETH instead. The 24h price change % is also displayed, and it changes for US$ versus BTC
 
-Given a US$ price for BTC, one might expect the $-per-altcoin price to be directly related to the BTC-per-altcoin 
 price using the formulas:
    $/eth * btc/$ = btc/eth =>  eth/btc = $/btc / $/eth
 Likewise, you might expect the percentage changes to be similarly related using the formula:
     %change_in_eth/btc = ((1 + %change_in_$/btc) / (1 + %change_in_$/eth)) - 1
     
-But these formulas do not hold, either on coinmarketcap_com or on exchanges. The BTC/altcoin price is determined
 independently in the market based on future expectations, and since it varies, so does the ratio of the change
 percentages.
 
-These facts are further complicated by the fact that most altcoins can be traded on most exchanges only against
 BTC. Only select few coins such as BTC, ETH, and BCH can be traded against fiat on limited exchanges. Although
 some no-fiat exchanges have trading in USDT (Tether) as a proxy for US$, even on those exchanges there are only
 a small number of coins which can trade against USDT and the rest trade only against BTC. 
 
-Coinmarketcap_com does not specify how they calculate their posted prices in terms of averaging over exchanges,
 nor how they calculate the fiat prices since most altcoins can only be traded against BTC. Their public API 
 such as "https://api.coinmarketcap.com/v1/ticker/?limit=10" provides price in US$ and price in BTC, but 
 unfortunately, it provides the 1h, 24h, and 7d % price change numbers only in US%
 
-Given that most altcoins can only be traded against BTC, it is best to track those coin values and their
 percentage price changes over time primarily in BTC. BTC of course can be tracked in US$.
  
-Doing this using coinmarketcap_com data requires that we download the US$ and BTC prices for the time durations of
 interest, and calculate the % changes ourselves. This can be done using the non-official API URLs which are
 used by coinmarketcap_com's individual altcoin graphing pages. This currently takes the form:
     http://graphs2.coinmarketcap.com/currencies/{}/{}000/{}000/
     -the first {} is the coin name
     -the 2nd and 3rd {} are the beginning and ending times in unix timestamp format
     
    the numbers are unix timestamps x1000 (three appended zeros)
    the timestamps are in UTC time
    The data is returned in ranges ending at 1 day, 7 days, 1 month, 3 months, 1 year, YTD, or ALL.
    The best resolution is data with 5 minute increments, which is achieved when the difference between the timestamps is < 24 hours.
    Longer time ranges return lower resolution data.

    The data comes back in JSON format as:

    {"market_cap_by_available_supply": [[1515981841000, 1283917901], [1515982141000, 1287030483]], 
    "price_btc": [[1515981841000, 8.21694e-07], [1515982141000, 8.25958e-07]], 
    "price_usd": [[1515981841000, 0.0113848], [1515982141000, 0.0114124]],
    "volume_usd": [[1515981841000, 68077800], [1515982141000, 68254400]]}

    Occasionally, sfor some coins, there will be an additional column representing the price in some
        other base currency if the coin is a derivative of another coin.
        
    In case there are errors fetching any of the coin info (for example, I had trouble with one coin for which
    one of the column contained data over fewer timestamps than the other columns which generated errors
    when converting the json into a dataframe), I do each coin fetch in a try, and just skip over that coin
    if there are errors
        
-The real-time nature of the coinmarketcap_com data varies. That is, the most recent price quote may be just 
    a few minutes old or it may by up to an hour old depending on how busy the website is. Also, the time
    increment from that most recent quote to the next older quote can be less than or equal to the standard
    time increment for that fetched time range size. Subsequent older data points are at a fixed timestamp 
    increment, which depends on the size of the time range of the requested fetch. For fetch requests with a 
    time ranges 
        less than 24 hours, the data comes back in increments of 5 minutes
        less than ~30 days, the data comes back in increments of 15 minutes
        less than 90 days, the data comes back in increments of 1 hour
        less than 180 days, the data comes back in increments of 2 hours
        less than 365 days, the data comes back in increments of 24 hour

    Remember to ignore the most recent timestamp point because its time delta to the previous timestamp
        varies randomly
        
-Since the quotes could be delayed up to an hour, it makes sense to do the fetch using an increment of slightly
    less than ~30 days, and then report 1-hour, 1-day, 7-day, and ~30-day changes
    
-If desired, a separate fetch could be done for a time range < 24 hours to calculate 1 hour price change
    statistics based on 5-minute resolution fetches
    
'''

#the coinmarketcap_com API url includes the number of coins we want
#  with '0' denoting all coins.
#  eg: 'https://api.coinmarketcap.com/v1/ticker/?limit=0' for all coins or
#      'https://api.coinmarketcap.com/v1/ticker/?limit=400' for the 400 coins with largest market cap

cmc_allcoins_ranking_base_url = 'https://api.coinmarketcap.com/v1/ticker/?limit='
cmc_ranking_data = {}

base_cmc_chart_url = 'http://graphs2.coinmarketcap.com/currencies/{}/{}000/{}000/'
cmc_chart_data = {}

cmc_summary_data = {}

total_sample_count = 0
delta_tick = 0
range_tick = 0
hour_sample_count = 0
day_sample_count = 0
day_7_sample_count = 0
day_approx30_sample_count = 0

def get_allcoins_ranking(count_of_ranked_coins):
    # 'count_of_ranked_coins' is a string integer reoresenting how many coins we want in the ranking.
    #     eg: '400' for the 400 coins with largest market cap, or '0' for all coins
    cmc_allcoins_ranking_url = cmc_allcoins_ranking_base_url + count_of_ranked_coins
    try:
        ret = requests.get(cmc_allcoins_ranking_url, timeout=https_timeout)
    except Exception as e:
        raise e
    finally:
        pass      
    ret_text = ret.text
    ret_list = json.loads(ret_text)
    df_ranking_data = pd.DataFrame.from_dict(ret_list)
    return df_ranking_data

def get_df_chart_data(cmc_chart_url):
    try:
        ret = requests.get(cmc_chart_url, timeout=https_timeout)
    except Exception as e:
        raise e
    finally:
        pass      
    ret_text = ret.text
    ret_list = json.loads(ret_text)
    df_chart_data = pd.DataFrame.from_dict(ret_list)
    return df_chart_data

def create_cmc_chart_data(coin_name):
    cmc_chart_url = base_cmc_chart_url.format(coin_name, str(int(unix_timestamp_start)), str(int(unix_timestamp_end)))
    cmc_chart_data[coin_name] = retry_call(get_df_chart_data, fargs=[cmc_chart_url], fkwargs=None, exceptions=requests.exceptions.RequestException, tries=4, delay=10, backoff=4)

def print_requested_data_range():
    print('Start of data requested: \t' + str(unix_timestamp_start) + '\t' + datetime.utcfromtimestamp(int(unix_timestamp_start)).strftime('%Y-%m-%d %H:%M:%S'))
    print('End of data requested: \t\t' + str(unix_timestamp_end) + '\t' + datetime.utcfromtimestamp(int(unix_timestamp_end)).strftime('%Y-%m-%d %H:%M:%S'))

    range_tick = unix_timestamp_end - unix_timestamp_start
    print('\nLength of range of data requested:')
    print('      {:8.4f} \t days'.format((range_tick / (60*60*24))))
    print('    {:10.4f} \t hours'.format((range_tick / (60*60))))
    print('  {:12.4f} \t minutes'.format((range_tick / 60)))
    print('{:14.4f} \t seconds'.format((range_tick) ))

def compute_tick_values(this_coin_id):    
    global total_sample_count
    global delta_tick
    global range_tick
    total_sample_count = cmc_chart_data[this_coin_id]['price_btc'].count()
    delta_tick = ((((cmc_chart_data[this_coin_id]['price_btc'][total_sample_count-2][0])/1000) - ((cmc_chart_data[this_coin_id]['price_btc'][0][0])/1000)) / (total_sample_count-2))
    range_tick = ((cmc_chart_data[this_coin_id]['price_btc'][total_sample_count-1][0])/1000) - ((cmc_chart_data[this_coin_id]['price_btc'][0][0])/1000)
    #
    global hour_sample_count
    global day_sample_count
    global day_7_sample_count
    global day_approx30_sample_count
    hour_sample_count = int(round(60*60 / delta_tick))
    day_sample_count = int(round(60*60*24 / delta_tick))
    day_7_sample_count = int(round(60*60*24*7 / delta_tick))
    day_approx30_sample_count = int(round(60*60*24*30 / delta_tick)) - 5
    
def print_tick_values():    
    print('The value of total_sample_count is:\t\t' + str(total_sample_count))
    print('The value of delta_tick is:\t\t\t' + str(delta_tick))
    print('The value of range_tick is:\t\t\t' + str(range_tick))
    #
    print('The value of hour_sample_count is:\t\t' + str(hour_sample_count))
    print('The value of day_sample_count is:\t\t' + str(day_sample_count))
    print('The value of day_7_sample_count is:\t\t' + str(day_7_sample_count))
    print('The value of day_approx30_sample_count is:\t' + str(day_approx30_sample_count))   
    
def print_downloaded_data_range_and_resolution(this_coin_id):
    for tick in range(0,4):
    #    print(str(((cmc_chart_data[this_coin_id]['price_btc'][tick][0])/1000)))
    #    print(datetime.utcfromtimestamp((cmc_chart_data[this_coin_id]['price_btc'][tick][0])/1000).strftime('%Y-%m-%d %H:%M:%S')))
        print(str(((cmc_chart_data[this_coin_id]['price_btc'][tick][0])/1000)) + '\t' + datetime.utcfromtimestamp((cmc_chart_data[this_coin_id]['price_btc'][tick][0])/1000).strftime('%Y-%m-%d %H:%M:%S'))
    print('...')
    for tick in range(total_sample_count-4,total_sample_count):
        print(str(((cmc_chart_data[this_coin_id]['price_btc'][tick][0])/1000)) + '\t' + datetime.utcfromtimestamp((cmc_chart_data[this_coin_id]['price_btc'][tick][0])/1000).strftime('%Y-%m-%d %H:%M:%S'))

    print('\nNumber of time samples: \t' + str(total_sample_count))

    print('\nResolution of time samples:')
    print('      {:8.4f} \t days'.format((delta_tick / (60*60*24))))
    print('    {:10.4f} \t hours'.format((delta_tick / (60*60))))
    print('  {:12.4f} \t minutes'.format((delta_tick / 60)))
    print('{:14.4f} \t seconds'.format((delta_tick) ))


    print('\nTime range of data downloaded from start to end:')
    print('      {:8.4f} \t days'.format((range_tick / (60*60*24))))
    print('    {:10.4f} \t hours'.format((range_tick / (60*60))))
    print('  {:12.4f} \t minutes'.format((range_tick / 60)))
    print('{:14.4f} \t seconds'.format((range_tick) ))

def print_downloaded_data_timestamp_locations(this_coin_id):
    print('The number of samples in 1 hour:\t' + str(hour_sample_count))
    print('The number of samples in 1 day:\t\t' + str(day_sample_count))
    print('The number of samples in 7 days:\t' + str(day_7_sample_count))
    print('The number of samples in ~30 days:\t' + str(day_approx30_sample_count))
    
    #print('The datestamp from 1 hour ago was:\t' + datetime.utcfromtimestamp((cmc_chart_data[this_coin_id]['price_btc'][total_sample_count - hour_sample_count][0])/1000).strftime('%Y-%m-%d %H:%M:%S'))
    #print('The datestamp from 1 day ago was:\t\t' + datetime.utcfromtimestamp((cmc_chart_data[this_coin_id]['price_btc'][total_sample_count - day_sample_count][0])/1000).strftime('%Y-%m-%d %H:%M:%S'))
    #print('The datestamp from 7 days ago was:\t' + datetime.utcfromtimestamp((cmc_chart_data[this_coin_id]['price_btc'][total_sample_count - day_7_sample_count][0])/1000).strftime('%Y-%m-%d %H:%M:%S'))
    #print('The datestamp from ~30 days ago was:\t' + datetime.utcfromtimestamp((cmc_chart_data[this_coin_id]['price_btc'][total_sample_count - day_approx30_sample_count][0])/1000).strftime('%Y-%m-%d %H:%M:%S'))
    print(' ')
    print('The timestamp of the requested data was:\t\t' + datetime.utcfromtimestamp(int(unix_timestamp_end)).strftime('%Y-%m-%d %H:%M:%S'))
    print('The timestamp of the 2nd-to-most-recent data was:\t' + datetime.utcfromtimestamp((cmc_chart_data[this_coin_id]['price_btc'][total_sample_count - 2][0])/1000).strftime('%Y-%m-%d %H:%M:%S'))
    try:
        print('The timestamp from 1 hour ago was:\t\t\t' + str(cmc_chart_data[this_coin_id]['price_btc'][total_sample_count - 2 - hour_sample_count][0]) + '\t' + datetime.utcfromtimestamp((cmc_chart_data[this_coin_id]['price_btc'][total_sample_count - 2 - hour_sample_count][0])/1000).strftime('%Y-%m-%d %H:%M:%S'))
    except:
        print('The timestamp from 1 hour ago is beyond the range of the downloaded data')
    try:
        print('The timestamp from 1 day ago was:\t\t\t' + str(cmc_chart_data[this_coin_id]['price_btc'][total_sample_count - 2 - day_sample_count][0]) + '\t' + datetime.utcfromtimestamp((cmc_chart_data[this_coin_id]['price_btc'][total_sample_count - 2 - day_sample_count][0])/1000).strftime('%Y-%m-%d %H:%M:%S'))
    except:
        print('The timestamp from 1 day ago is beyond the range of the downloaded data')
    try:
        print('The timestamp from 7 days ago was:\t\t\t' + str(cmc_chart_data[this_coin_id]['price_btc'][total_sample_count - 2 - day_7_sample_count][0]) + '\t' + datetime.utcfromtimestamp((cmc_chart_data[this_coin_id]['price_btc'][total_sample_count - 2 - day_7_sample_count][0])/1000).strftime('%Y-%m-%d %H:%M:%S'))
    except:
        print('The timestamp from 7 days ago is beyond the range of the downloaded data')
    try:
        print('The timestamp from ~30 days ago was:\t\t\t' + str(cmc_chart_data[this_coin_id]['price_btc'][total_sample_count - 2 - day_approx30_sample_count][0]) + '\t' + datetime.utcfromtimestamp((cmc_chart_data[this_coin_id]['price_btc'][total_sample_count - 2 - day_approx30_sample_count][0])/1000).strftime('%Y-%m-%d %H:%M:%S'))
    except:
        print('The timestamp from ~30 days ago is beyond the range of the downloaded data')
   
#==================

try:
    type(cmc_ranking_data)
except:
    pass
else:
    del cmc_ranking_data
cmc_ranking_data = {}
#
try:
    type(cmc_ranking_data_indexed)
except:
    pass
else:
    del cmc_ranking_data_indexed
cmc_ranking_data_indexed = {}

#==================
# Download the summary and ranking data from coinmarketcap.com

#count_of_ranked_coins = '0'
count_of_ranked_coins = '90'
cmc_ranking_data = get_allcoins_ranking(count_of_ranked_coins)

#==================
# Some useful commands for understanding the cmc_ranking_data structure

#cmc_ranking_data['id']
#cmc_ranking_data['id'].head(5)

#print(list(cmc_ranking_data['id']))
#print(list(cmc_ranking_data['id'].head(5)))

#cmc_ranking_data
#cmc_ranking_data.head(5)
#cmc_ranking_data.tail(5)

#==================

poloniex_coin_list = ['BCN','BELA','BLK','BTCD','BTM','BTS','BURST','CLAM','DASH','DGB','DOGE','EMC2','FLDC','FLO','GAME','GRC','HUC','LTC','MAID','OMNI','NAV','NEOS','NMC','NXT','PINK','POT','PPC','RIC','STR','SYS','VIA','XVC','VRC','VTC','XBC','XCP','XEM','XMR','XPM','XRP','ETH','SC','BCY','EXP','FCT','RADS','AMP','DCR','LSK','LBC','STEEM','SBD','ETC','REP','ARDR','ZEC','STRAT','NXC','PASC','GNT','GNO','BCH','ZRX','CVC','OMG','GAS','STORJ']
bittrex_coin_list = ["LTC","DOGE","VTC","PPC","FTC","RDD","NXT","DASH","POT","BLK","EMC2","XMY","AUR","EFL","GLD","SLR","PTC","GRS","NLG","RBY","XWC","MONA","THC","ENRG","ERC","VRC","CURE","XMR","CLOAK","START","KORE","XDN","TRUST","NAV","XST","BTCD","VIA","PINK","IOC","CANN","SYS","NEOS","DGB","BURST","EXCL","SWIFT","DOPE","BLOCK","ABY","BYC","XMG","BLITZ","BAY","FAIR","SPR","VTR","XRP","GAME","COVAL","NXS","XCP","BITB","GEO","FLDC","GRC","FLO","NBT","MUE","XEM","CLAM","DMD","GAM","SPHR","OK","SNRG","PKB","CPC","AEON","ETH","GCR","TX","BCY","EXP","INFX","OMNI","AMP","AGRS","XLM","CLUB","VOX","EMC","FCT","MAID","EGC","SLS","RADS","DCR","BSD","XVG","PIVX","XVC","MEME","STEEM","2GIVE","LSK","PDC","BRK","DGD","WAVES","RISE","LBC","SBD","BRX","ETC","STRAT","UNB","SYNX","TRIG","EBST","VRM","SEQ","REP","SHIFT","ARDR","XZC","NEO","ZEC","ZCL","IOP","GOLOS","UBQ","KMD","GBG","SIB","ION","LMC","QWARK","CRW","SWT","MLN","ARK","DYN","TKS","MUSIC","DTB","INCNT","GBYTE","GNT","NXC","EDG","LGD","TRST","WINGS","RLC","GNO","GUP","LUN","APX","HMQ","ANT","SC","BAT","ZEN","1ST","QRL","CRB","PTOY","MYST","CFI","BNT","NMR","SNT","DCT","XEL","MCO","ADT","FUN","PAY","MTL","STORJ","ADX","OMG","CVC","PART","QTUM","BCC","DNT","ADA","MANA","SALT","TIX","RCN","VIB","MER","POWR","BTG","ENG","UKG"]

#==================

# To find the coins traded on an exchange which have the smallest market cap

#coin_filter_list = bittrex_coin_list
#cmc_ranking_coin_symbol_list = list(cmc_ranking_data['symbol'])
#filtered_cmc_ranking_data = cmc_ranking_data.copy()
#for this_symbol in cmc_ranking_coin_symbol_list:
#    if not this_symbol in coin_filter_list:
#        filtered_cmc_ranking_data = filtered_cmc_ranking_data[filtered_cmc_ranking_data.symbol != this_symbol]
#filtered_cmc_ranking_data

#==================

#type(cmc_ranking_data['price_btc'][0])

#==================

# Clean up the cmc_ranking_data dataframe by deleting coins which are missing the required price_btc or price_usd fields

#len(cmc_ranking_data.index)

cmc_ranking_data = cmc_ranking_data.drop(cmc_ranking_data[cmc_ranking_data.price_btc.isnull() | cmc_ranking_data.price_usd.isnull()].index)
#df = df.drop( df[ (df.X == x) & (df.Y==y)  & (df.Z==Z)].index )

#len(cmc_ranking_data.index)

#==================

# Set the time range of data to download

#-------
unix_timestamp_end = datetime.now().timestamp()
#print(str(unix_timestamp_end))

#string_time = '2017-07-01 00:05:00'
#unix_timestamp_start = datetime.strptime(string_time, '%Y-%m-%d %H:%M:%S').timestamp()

unix_timestamp_start = (unix_timestamp_end - 60*60*24*30 + 60*15*2)    # ~30 days - 2*15 minutes
#print(str(unix_timestamp_start))
#-------

try:
    type(cmc_chart_data)
except:
    pass
else:
    del cmc_chart_data
cmc_chart_data = {}

#==================

# Download the detailed data over time for each coin from coinmarketcap.com

cmc_ranking_coin_id_list = list(cmc_ranking_data['id'])
#cmc_ranking_coin_id_list = list(cmc_ranking_data['id'].head(5))
coin_count = 0
for this_id in cmc_ranking_coin_id_list:
    coin_count = coin_count + 1
    try:
        print(str(coin_count) + '\tCreating the chart data for coin '+this_id)
        create_cmc_chart_data(this_id)
    except:
        print('Cancelling the chart data for coin '+this_id)
        pass

#==================

# Some useful commands for understanding the cmc_chart_data structure

#cmc_chart_data['ethereum']['price_btc'].count()
#cmc_chart_data['ethereum']['price_btc'][0]
#cmc_chart_data['ethereum']['price_btc'][0][1]
#cmc_chart_data['ethereum']['price_btc'][0][0]
#datetime.utcfromtimestamp((cmc_chart_data['ethereum']['price_btc'][0][0])/1000).strftime('%Y-%m-%d %H:%M:%S')

#print(list(cmc_chart_data))

#cmc_chart_data

#==================

# Uncomment to check the requested time range of the data

#print_requested_data_range()
# Uncomment to check the various dates, number of samples, and resolutions of the downloaded data

#this_coin_id = 'ethereum'
#compute_tick_values(this_coin_id)
#print_tick_values()
#print('\n=========================================\n')
#print_downloaded_data_range_and_resolution(this_coin_id)
#print('\n=========================================\n')
#print_downloaded_data_timestamp_locations(this_coin_id)

#==================

try:
    type(cmc_summary_data_df_template)
except:
    pass
else:
    del cmc_summary_data_df_template
cmc_summary_data_df_template = {}
#
try:
    type(cmc_summary_data_df)
except:
    pass
else:
    del cmc_summary_data_df
cmc_summary_data_df = {}
#
try:
    type(cmc_summary_data_df_indexed)
except:
    pass
else:
    del cmc_summary_data_df_indexed
cmc_summary_data_df_indexed = {}

#==================

# Create the template for the summary dataframe

cmc_summary_data_dictA = {'id': "", 'name': "", 'symbol': "", 'price_btc': 0, 'price_usd': 0, '24h_volume_usd': 0, 'market_cap_usd': 0, 'hour_ago_price_btc': 0, 'hour_change_btc': 0, 'hour_ago_price_usd': 0, 'hour_change_usd': 0}
cmc_summary_data_dictB = {'day_ago_price_btc': 0, 'day_change_btc': 0, 'day_ago_price_usd': 0, 'day_change_usd': 0, 'days_7_ago_price_btc': 0, 'days_7_change_btc': 0, 'days_7_ago_price_usd': 0, 'days_7_change_usd': 0}
cmc_summary_data_dictC = {'days_approx30_ago_price_btc': 0, 'days_approx30_change_btc': 0, 'days_approx30_ago_price_usd': 0, 'days_approx30_change_usd': 0}
cmc_summary_data_dict = cmc_summary_data_dictA.copy()
cmc_summary_data_dict.update(cmc_summary_data_dictB)
cmc_summary_data_dict.update(cmc_summary_data_dictC)
#
cmc_summary_data_df_template =  pd.DataFrame([cmc_summary_data_dict])
cmc_summary_data_df = cmc_summary_data_df_template.copy()

#cmc_summary_data_dict
#cmc_summary_data_df_template

#==================

# Create the list of coins that we will include in the summary dataframe

cmc_chart_data_list = list(cmc_chart_data)
cmc_chart_data_list.remove('bitcoin')
#cmc_chart_data_list
#len(cmc_chart_data_list)

#==================

# Create the summary dataframe with an empty row for each coin

cmc_chart_data_list = list(cmc_chart_data)
cmc_chart_data_list.remove('bitcoin')
#cmc_chart_data_list
cmc_summary_data_df['id'] = 'bitcoin'
for this_coin_id in cmc_chart_data_list:
    cmc_summary_data_df_temp = cmc_summary_data_df_template.copy()
    cmc_summary_data_df_temp['id'] = this_coin_id
    cmc_summary_data_df = pd.concat([cmc_summary_data_df, cmc_summary_data_df_temp])
#cmc_summary_data_df
#type(cmc_summary_data_df)
#len(cmc_summary_data_df)

#==================

# Populate the summary dataframe with the summary of info for each coin computed from the downloaded data

cmc_ranking_data_indexed = cmc_ranking_data.set_index('id')
cmc_summary_data_df_indexed = cmc_summary_data_df.set_index('id')
#
cmc_summary_data_df_indexed_id_list = list(cmc_summary_data_df_indexed.index)
#cmc_summary_data_df_indexed_id_list

for this_coin_id in cmc_summary_data_df_indexed_id_list:
    print('Creating cmc_summary_data_df entries for coin: ' + this_coin_id)
    compute_tick_values(this_coin_id)
    cmc_summary_data_df_indexed.loc[this_coin_id, 'name'] = cmc_ranking_data_indexed.loc[this_coin_id, 'name']
    cmc_summary_data_df_indexed.loc[this_coin_id, 'symbol'] = cmc_ranking_data_indexed.loc[this_coin_id, 'symbol']
    cmc_summary_data_df_indexed.loc[this_coin_id, 'price_btc'] = cmc_chart_data[this_coin_id]['price_btc'][total_sample_count - 2][1]
    cmc_summary_data_df_indexed.loc[this_coin_id, 'price_usd'] = cmc_chart_data[this_coin_id]['price_usd'][total_sample_count - 2][1]
    cmc_summary_data_df_indexed.loc[this_coin_id, '24h_volume_usd'] = cmc_ranking_data_indexed.loc[this_coin_id, '24h_volume_usd']
    cmc_summary_data_df_indexed.loc[this_coin_id, 'market_cap_usd'] = cmc_ranking_data_indexed.loc[this_coin_id, 'market_cap_usd']
    #
    try:
        cmc_summary_data_df_indexed.loc[this_coin_id, 'hour_ago_price_btc'] = cmc_chart_data[this_coin_id]['price_btc'][total_sample_count - 2 - hour_sample_count][1]
        cmc_summary_data_df_indexed.loc[this_coin_id, 'hour_change_btc'] = cmc_chart_data[this_coin_id]['price_btc'][total_sample_count - 2][1] / cmc_chart_data[this_coin_id]['price_btc'][total_sample_count - 2 - hour_sample_count][1]
        cmc_summary_data_df_indexed.loc[this_coin_id, 'hour_ago_price_usd'] = cmc_chart_data[this_coin_id]['price_usd'][total_sample_count - 2 - hour_sample_count][1]
        cmc_summary_data_df_indexed.loc[this_coin_id, 'hour_change_usd'] = cmc_chart_data[this_coin_id]['price_usd'][total_sample_count - 2][1] / cmc_chart_data[this_coin_id]['price_usd'][total_sample_count - 2 - hour_sample_count][1]
    except:
        print('Error- There is no hour_ago cmc_chart_data for coin: ' + this_coin_id)
        cmc_summary_data_df_indexed.loc[this_coin_id, 'hour_ago_price_btc'] = np.NaN
        cmc_summary_data_df_indexed.loc[this_coin_id, 'hour_change_btc'] = np.NaN
        cmc_summary_data_df_indexed.loc[this_coin_id, 'hour_ago_price_usd'] = np.NaN
        cmc_summary_data_df_indexed.loc[this_coin_id, 'hour_change_usd'] = np.NaN
    #
    try:
        cmc_summary_data_df_indexed.loc[this_coin_id, 'day_ago_price_btc'] = cmc_chart_data[this_coin_id]['price_btc'][total_sample_count - 2 - day_sample_count][1]
        cmc_summary_data_df_indexed.loc[this_coin_id, 'day_change_btc'] =  cmc_chart_data[this_coin_id]['price_btc'][total_sample_count - 2][1] / cmc_chart_data[this_coin_id]['price_btc'][total_sample_count - 2 -day_sample_count][1]
        cmc_summary_data_df_indexed.loc[this_coin_id, 'day_ago_price_usd'] = cmc_chart_data[this_coin_id]['price_usd'][total_sample_count - 2 - day_sample_count][1]
        cmc_summary_data_df_indexed.loc[this_coin_id, 'day_change_usd'] =  cmc_chart_data[this_coin_id]['price_usd'][total_sample_count - 2][1] / cmc_chart_data[this_coin_id]['price_usd'][total_sample_count - 2 - day_sample_count][1]
    except:
        print('Error- There is no day_ago cmc_chart_data for coin: ' + this_coin_id)
        cmc_summary_data_df_indexed.loc[this_coin_id, 'day_ago_price_btc'] = np.NaN
        cmc_summary_data_df_indexed.loc[this_coin_id, 'day_change_btc'] = np.NaN
        cmc_summary_data_df_indexed.loc[this_coin_id, 'day_ago_price_usd'] = np.NaN
        cmc_summary_data_df_indexed.loc[this_coin_id, 'day_change_usd'] = np.NaN
    #
    try:
        cmc_summary_data_df_indexed.loc[this_coin_id, 'days_7_ago_price_btc'] = cmc_chart_data[this_coin_id]['price_btc'][total_sample_count - 2 - day_7_sample_count][1]
        cmc_summary_data_df_indexed.loc[this_coin_id, 'days_7_change_btc'] =  cmc_chart_data[this_coin_id]['price_btc'][total_sample_count - 2][1] / cmc_chart_data[this_coin_id]['price_btc'][total_sample_count - 2 -day_7_sample_count][1]
        cmc_summary_data_df_indexed.loc[this_coin_id, 'days_7_ago_price_usd'] = cmc_chart_data[this_coin_id]['price_usd'][total_sample_count - 2 - day_7_sample_count][1]
        cmc_summary_data_df_indexed.loc[this_coin_id, 'days_7_change_usd'] =  cmc_chart_data[this_coin_id]['price_usd'][total_sample_count - 2][1] / cmc_chart_data[this_coin_id]['price_usd'][total_sample_count - 2 - day_7_sample_count][1]
    except:
        print('Error- There is no days_7_ago cmc_chart_data for coin: ' + this_coin_id)
        cmc_summary_data_df_indexed.loc[this_coin_id, 'days_7_ago_price_btc'] = np.NaN
        cmc_summary_data_df_indexed.loc[this_coin_id, 'days_7_change_btc'] = np.NaN
        cmc_summary_data_df_indexed.loc[this_coin_id, 'days_7_ago_price_usd'] = np.NaN
        cmc_summary_data_df_indexed.loc[this_coin_id, 'days_7_change_usd'] = np.NaN
    #
    try:
        cmc_summary_data_df_indexed.loc[this_coin_id, 'days_approx30_ago_price_btc'] = cmc_chart_data[this_coin_id]['price_btc'][total_sample_count - 2 - day_approx30_sample_count][1]
        cmc_summary_data_df_indexed.loc[this_coin_id, 'days_approx30_change_btc'] =  cmc_chart_data[this_coin_id]['price_btc'][total_sample_count - 2][1] / cmc_chart_data[this_coin_id]['price_btc'][total_sample_count - 2 -day_approx30_sample_count][1]
        cmc_summary_data_df_indexed.loc[this_coin_id, 'days_approx30_ago_price_usd'] = cmc_chart_data[this_coin_id]['price_usd'][total_sample_count - 2 - day_approx30_sample_count][1]
        cmc_summary_data_df_indexed.loc[this_coin_id, 'days_approx30_change_usd'] =  cmc_chart_data[this_coin_id]['price_usd'][total_sample_count - 2][1] / cmc_chart_data[this_coin_id]['price_usd'][total_sample_count - 2 - day_approx30_sample_count][1]
    except:
        print('Error- There is no days_approx30_ago cmc_chart_data for coin: ' + this_coin_id)
        cmc_summary_data_df_indexed.loc[this_coin_id, 'days_approx30_ago_price_btc'] = np.NaN
        cmc_summary_data_df_indexed.loc[this_coin_id, 'days_approx30_change_btc'] = np.NaN
        cmc_summary_data_df_indexed.loc[this_coin_id, 'days_approx30_ago_price_usd'] = np.NaN
        cmc_summary_data_df_indexed.loc[this_coin_id, 'days_approx30_change_usd'] = np.NaN

#==================

# Note how the indexing is tricky for setting values. The simplest ways generate the python error
#    SettingWithCopyWarning:
#    A value is trying to be set on a copy of a slice from a DataFrame
#    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy

# Uncomment this code to try it:
'''
this_coin_id = 'bitcoin'
cmc_ranking_data_indexed.loc[this_coin_id]['name']
cmc_ranking_data_indexed.loc[this_coin_id, 'name']
type(cmc_ranking_data_indexed.loc[this_coin_id]['name'])
cmc_summary_data_df_indexed.loc[this_coin_id]['name']
cmc_summary_data_df_indexed.loc[this_coin_id]['name'] = cmc_ranking_data_indexed.loc[this_coin_id]['name']
cmc_summary_data_df_indexed.loc[this_coin_id]['name']
cmc_summary_data_df_indexed.loc[this_coin_id]['name'] = 'WTF'
cmc_summary_data_df_indexed.loc[this_coin_id]['name']
cmc_summary_data_df_indexed.loc[this_coin_id, 'name'] = 'WTF'
cmc_summary_data_df_indexed.loc[this_coin_id]['name']
'''

#==================

#cmc_summary_data_df_indexed

#==================

def create_cmc_coins_sorted_by_price_change_truncated(change_variable, number_of_coins, coin_symbol_filter_list):
    # change_variable may be one of: 'hour_change_btc', 'day_change_btc', 'days_7_change_btc', 'days_approx30_change_btc'
    # number_of_coins is the length of the list we want. They list may be shorter if only fewer coins had data fetched.
    # coin_symbol_filter_list is a list of coin symbols to be included in the ranking
    coin_id_list = list(cmc_summary_data_df_indexed.index)
    coin_id_list.remove('bitcoin')
    coins_with_change_ratios = {}
    for this_coin_id in coin_id_list:
        if cmc_summary_data_df_indexed.loc[this_coin_id, 'symbol'] in coin_symbol_filter_list:
            coins_with_change_ratios[this_coin_id] = cmc_summary_data_df_indexed.loc[this_coin_id, change_variable]

    coins_with_change_ratios
    cmc_coins_sorted_by_price_change = sorted(coins_with_change_ratios, key=coins_with_change_ratios.__getitem__, reverse=True)

    max_number_of_coins = min(len(cmc_coins_sorted_by_price_change), number_of_coins)
    cmc_coins_sorted_by_price_change_truncated = cmc_coins_sorted_by_price_change[0:max_number_of_coins]
    
    return cmc_coins_sorted_by_price_change_truncated
    
#==================
    
def print_cmc_coins_sorted_by_price_change(change_variable, cmc_coins_sorted_by_price_change_truncated):
    # change_variable may be one of: 'hour_change_btc', 'day_change_btc', 'days_7_change_btc', 'days_approx30_change_btc'
    print('Symbol  %_Up     24h_USD$        Name          Price_in_BTC Price_in_USD$   1h_rise 1d_rise 7d_rise 30d_rise')
    file.write('Symbol  %_Up     24h_USD$        Name          Price_in_BTC Price_in_USD$   1h_rise 1d_rise 7d_rise 30d_rise\n')
    for this_coin_id in cmc_coins_sorted_by_price_change_truncated:
        symbol = cmc_summary_data_df_indexed.loc[this_coin_id, 'symbol']
        name = cmc_summary_data_df_indexed.loc[this_coin_id, 'name']
        price_btc = cmc_summary_data_df_indexed.loc[this_coin_id, 'price_btc']
        price_usd = cmc_summary_data_df_indexed.loc[this_coin_id, 'price_usd']
        h24_volume_usd = cmc_summary_data_df_indexed.loc[this_coin_id, '24h_volume_usd']
        #
        dot_place = h24_volume_usd.find(".")
        if dot_place == -1:
            h24_volume_usd_int = int(h24_volume_usd)
        else:
            h24_volume_usd_int = int(h24_volume_usd[0:dot_place])
        #
        hour_change_btc = cmc_summary_data_df_indexed.loc[this_coin_id, 'hour_change_btc']
        day_change_btc = cmc_summary_data_df_indexed.loc[this_coin_id, 'day_change_btc']
        days_7_change_btc = cmc_summary_data_df_indexed.loc[this_coin_id, 'days_7_change_btc']
        days_approx30_change_btc = cmc_summary_data_df_indexed.loc[this_coin_id, 'days_approx30_change_btc']
        if change_variable == 'hour_change_btc':
            this_percentage = (hour_change_btc - 1) * 100
        elif change_variable == 'day_change_btc':
            this_percentage = (day_change_btc - 1) * 100
        elif change_variable == 'days_7_change_btc':
            this_percentage = (days_7_change_btc - 1) * 100
        elif change_variable == 'days_approx30_change_btc':
            this_percentage = (days_approx30_change_btc - 1) * 100

    #    print('Symbol   Name             Price_in_BTC      Price in USD$      1_hour_rise  1_day_rise  7_day_rise   30_day_rise')
    #    print('{:5s}    {:15s}    {:8.6f}        {:12.6f}          {:5.2f}       {:5.2f}       {:5.2f}       {:5.2f}'.format(symbol, name, price_btc, price_usd, hour_change_btc, day_change_btc, days_7_change_btc, days_approx30_change_btc))
    #    print('###############') 
    #    print('Symbol  %_Up     24h_USD$        Name          Price_in_BTC Price_in_USD$   1h_rise 1d_rise 7d_rise 30d_rise')
        print('{:5s} {:7.2f}%   ${:11d}   {:15s} {:8.6f}    ${:12.6f}   {:5.2f}   {:5.2f}    {:5.2f}   {:5.2f}'.format(symbol, this_percentage, h24_volume_usd_int, name[0:14], price_btc, price_usd, hour_change_btc, day_change_btc, days_7_change_btc, days_approx30_change_btc))
        file.write('{:5s} {:7.2f}%   ${:11d}   {:15s} {:8.6f}    ${:12.6f}   {:5.2f}   {:5.2f}    {:5.2f}   {:5.2f}\n'.format(symbol, this_percentage, h24_volume_usd_int, name[0:14], price_btc, price_usd, hour_change_btc, day_change_btc, days_7_change_btc, days_approx30_change_btc))

#==================

def print_hour_day_7d_30d_change_coin_rank_lists(number_of_coins, coin_symbol_filter_list): 
    # number_of_coins is the length of the list we want. They list may be shorter if only fewer coins had data fetched.
    # coin_symbol_filter_list is a list of coin symbols to be included in the ranking
    #
    print('List of coins ranked by % change in price-per-BTC in the last Hour:')
    file.write('List of coins ranked by % change in price-per-BTC in the last Hour:\n')
    change_variable = 'hour_change_btc'
    cmc_coins_sorted_by_price_change_truncated = create_cmc_coins_sorted_by_price_change_truncated(change_variable, number_of_coins, coin_symbol_filter_list)
    print_cmc_coins_sorted_by_price_change(change_variable, cmc_coins_sorted_by_price_change_truncated)
    #
    print('\nList of coins ranked by % change in price-per-BTC in the last Day:')
    file.write('\nList of coins ranked by % change in price-per-BTC in the last Day:\n')
    change_variable = 'day_change_btc'
    cmc_coins_sorted_by_price_change_truncated = create_cmc_coins_sorted_by_price_change_truncated(change_variable, number_of_coins, coin_symbol_filter_list)
    print_cmc_coins_sorted_by_price_change(change_variable, cmc_coins_sorted_by_price_change_truncated)
    #
    print('\nList of coins ranked by % change in price-per-BTC in the last 7 Days:')
    file.write('\nList of coins ranked by % change in price-per-BTC in the last 7 Days:\n')
    change_variable = 'days_7_change_btc'
    cmc_coins_sorted_by_price_change_truncated = create_cmc_coins_sorted_by_price_change_truncated(change_variable, number_of_coins, coin_symbol_filter_list)
    print_cmc_coins_sorted_by_price_change(change_variable, cmc_coins_sorted_by_price_change_truncated)
    #
    print('\nList of coins ranked by % change in price-per-BTC in the last ~ 30 Days:')
    file.write('\nList of coins ranked by % change in price-per-BTC in the last ~ 30 Days:\n')
    change_variable = 'days_approx30_change_btc'
    cmc_coins_sorted_by_price_change_truncated = create_cmc_coins_sorted_by_price_change_truncated(change_variable, number_of_coins, coin_symbol_filter_list)
    print_cmc_coins_sorted_by_price_change(change_variable, cmc_coins_sorted_by_price_change_truncated)

#==================

#Print the desired reports

file = open("altcoin_trend_rank.txt", "w")

# number_of_coins is the length of the list we want. They list may be shorter if only fewer coins had data fetched.
number_of_coins = 30

print('\n'+time.ctime()+'\n')
file.write('\n'+time.ctime()+'\n\n')

# Print the ranking list including only the coins which are traded on Poloniex:
coin_symbol_filter_list = poloniex_coin_list
print('List of POLONIEX traded coins ranked by % change in the price-per-BTC over time=>\n')
file.write('List of POLONIEX traded coins ranked by % change in the price-per-BTC over time=>\n\n')
print_hour_day_7d_30d_change_coin_rank_lists(number_of_coins, coin_symbol_filter_list)

# Print the ranking list including only the coins which are traded on Bittrex:
coin_symbol_filter_list = bittrex_coin_list
print('\n\n==========================================================\n')
print('List of BITTREX traded coins ranked by % change in the price-per-BTC over time=>\n')
file.write('\n\n==========================================================\n\n')
file.write('List of BITTREX traded coins ranked by % change in the price-per-BTC over time=>\n\n')
print_hour_day_7d_30d_change_coin_rank_lists(number_of_coins, coin_symbol_filter_list)

# Print the ranking list including all the coins from coinmarketcap.com:
coin_symbol_filter_list = list(cmc_summary_data_df_indexed['symbol'])
print('\n\n==========================================================\n')
print('List of the top '+str(number_of_coins)+ ' coins from "coinmarketcap.com" ranked by % change in the price-per-BTC over time=>\n')
file.write('\n\n==========================================================\n\n')
file.write('List of the top '+str(number_of_coins)+ ' coins from "coinmarketcap.com" ranked by % change in the price-per-BTC over time=>\n\n')
print_hour_day_7d_30d_change_coin_rank_lists(number_of_coins, coin_symbol_filter_list)

'''
# Print the ranking list including only certain selected coins
print('\n\n==========================================================\n')
print('List of certain selected coins ranked by their 30-day % change in price-per-BTC=>\n')
file.write('\n\n==========================================================\n\n')
file.write('List of certain selected coins ranked by their 30-day % change in price-per-BTC=>\n\n')
change_variable = 'days_approx30_change_btc'
coin_symbol_filter_list = ['ETC','MAID','PIVX','GNO','VRT']
cmc_coins_sorted_by_price_change_truncated = create_cmc_coins_sorted_by_price_change_truncated(change_variable, number_of_coins, coin_symbol_filter_list)
print_cmc_coins_sorted_by_price_change(change_variable, cmc_coins_sorted_by_price_change_truncated)
'''

print('\n\n')
file.write('\n\n\n')

file.close()
