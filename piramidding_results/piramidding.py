import pandas as pd
import json
import os
import random

####
# only run new calculations
# run all possible levels concurrently
# upgrade fee levels during run if trade volume is high enough
####

modelfile = "model_small_10"
dir = f"/home/joren/Coding/cryptodata/predictions/{modelfile}/"
column_names = ['open','close','high','low','volume','adosc','atr','macd','macd_signal','macd_hist','mfi','upper_band','middle_band','lower_band','rsi','difference_low_high','difference_open_close','target']

if not os.path.exists(f"piramidding_results/{modelfile}"):
    os.makedirs(f"piramidding_results/{modelfile}/")

def filenames():
    filenames = []
    for file in os.listdir(dir):
        filename = os.fsdecode(file)
        filenames.append(filename)
        # random.shuffle(filenames)
    return filenames
files = filenames()

def random_offset(prize):
    randomnr = random.randint(-2, 2)
    prize = prize*(1+(randomnr/1000))
    return prize

levels = {
    1: [0.004, 0.004, 0],
    2: [0.0035, 0.0035, 25000],
    3: [0.0015, 0.0025, 50000],
    4: [0.001, 0.0016, 100000],
    5: [0.0009, 0.0015, 250000],
    6: [0.0008, 0.0014, 1000000],
    7: [0.0007, 0.0013, 20000000],
    8: [0.0006, 0.0012, 100000000],
    9: [0.0004, 0.001, 200000000]
}
original_level = 1
discount = 0

def set_level(levelnr):
    global level, discount, maker, taker
    print(f'set level to {levelnr}')
    level = levelnr
    maker, taker, minimum_volume = levels[level]

    #BINANCE#
    maker, taker = 0.001, 0.001
    minimum_volume = 9999999999999999999999
    #BINANCE#

    if discount > 0:
        maker *= (1-discount)
        taker *= (1-discount)

set_level(original_level)

processed = []
if os.path.exists(f"piramidding_results/{modelfile}/level{original_level}_discount{discount}_result.json"):
    with open(f"piramidding_results/{modelfile}/level{original_level}_discount{discount}_result.json", 'r+') as file:
        results = json.load(file)
        for item in results['results']:
            processed.append(item['file'])

for file in files:
    # print(file)
    # if file in processed:
    #     print(file)
    #     continue
    
    df = pd.read_csv(f"/home/joren/Coding/cryptodata/predictions/model_small_10/{file}", index_col=0)
    df.drop(columns=column_names, inplace=True)

    df2 = pd.read_csv(f"/home/joren/Coding/cryptodata/1MIN/{file}")
    df2 = df2.rename(columns={"event_time": "timestamp"})
    df2 = df2.set_index('timestamp')

    df = df.join(df2)
    # df = df.iloc[-10080:]

    buy_hold = ((df.iloc[-1]['close'] / df.iloc[0]['close']) - 1) * 100
    # print(f"buy & hold strat: {round((buy_hold-1)*100, 2)}% result: {round(10*buy_hold, 2)}")

#################### with piramidding ####################
    set_level(original_level)
    candle = 0
    average_buy_price = 0
    last_buy_price = 0

    position = 0
    buy_count = 0

    balance = 20
    original_balance = balance
    inzet = 1
    trade_volume = 0

    for index, row in df.iterrows():
        candle += 1
        if position > 0:
            if row['predictions'] == 1 and row['close'] < last_buy_price and balance >= inzet:
                #piramidding
                buy_count += 1
                position += inzet - ( taker * inzet )
                trade_volume += inzet - ( taker * inzet )

                # buy_price = random_offset(row['close'])
                buy_price = row['close']
                last_buy_price = buy_price
                average_buy_price = ( buy_price + (average_buy_price*(buy_count-1)) ) / buy_count

                balance -= inzet
                
        if row['predictions'] == 1 and position == 0 and balance >= inzet:
            #initial buy
            # print('first buy')
            buy_count += 1
            position += inzet - ( taker * inzet )
            trade_volume += inzet - ( taker * inzet )
            # buy_price = random_offset(row['close'])
            buy_price = row['close']
            last_buy_price = buy_price
            average_buy_price = buy_price
            balance -= inzet

        if row['predictions'] == 2 and position > 0:
            # sell
            # buy_price = random_offset(row['close'])
            buy_price = row['close']
            percentage = average_buy_price / buy_price

            balance += (position-(maker*position)) * percentage
            trade_volume += position-(maker*position)

            position = 0
            buy_count = 0
        

        if level < 9 and (trade_volume/(candle/43200) > levels[level+1][2]) and trade_volume > levels[level+1][2]:
            # print(f"trade volume: {trade_volume}")
            # print(f"monthly volume: {(trade_volume/(candle/43200))} ")
            set_level(level+1)
    # print(f"Piramidding: {balance}")

################### without piramidding ###################
    set_level(original_level)
    candle = 0
    position = 0
    buy_price = 0
    normal_balance = 20
    normal_original_balance = normal_balance
    normal_inzet = 1
    normal_trade_volume = 0

    for index, row in df.iterrows():
        candle += 1
        if row['predictions'] == 1 and position == 0 and normal_balance > normal_inzet:
            position = normal_inzet - ( taker * normal_inzet )
            normal_trade_volume += normal_inzet - ( taker * normal_inzet )
            buy_price = row['close']
            normal_balance -= normal_inzet

        if row['predictions'] == 2 and position > 0:
            percentage = buy_price / row['close']
            normal_balance += (position-maker) * percentage
            normal_trade_volume += position-(maker*position)
            position = 0
        
        if level < 9 and (normal_trade_volume/(candle/43200) > levels[level+1][2]) and normal_trade_volume > levels[level+1][2]:
            set_level(level+1)
    
    # print(f"Normal: {normal_balance}")

################### Write to JSON ###################

    print({
        "file": file,
        "buy & hold": round(buy_hold, 2),
        "piramidding_start_balance": original_balance,
        "piramidding": round( balance, 2),
        "piramidding%": str( round( ( ( balance / original_balance ) - 1 ) * 100, 2) ) + "%",
        "piramidding_inzet": inzet,
        "piramidding_volume": str( round( trade_volume / 1000, 2) )+"K",
        "normal_start_balance": normal_original_balance,
        "normal": round( normal_balance, 2),
        "normal%": str( round( ( ( normal_balance / normal_original_balance ) - 1 ) * 100, 2) ) + "%",
        "normal_inzet": normal_inzet,
        "normal_volume": str( round( normal_trade_volume / 1000, 2) )+"K",
        "period_days": round(len(df)/60/24, 2)
    })

    dictionary ={
        "file": file,
        "buy & hold": round(buy_hold, 2),
        "piramidding_start_balance": original_balance,
        "piramidding": round( balance, 2),
        "piramidding%": str( round( ( ( balance / original_balance ) - 1 ) * 100, 2 ) ) + "%",
        "piramidding_inzet": inzet,
        "piramidding_volume": str( round( trade_volume / 1000, 2))+"K",
        "normal_start_balance": normal_original_balance,
        "normal": round( normal_balance, 2),
        "normal%": str( round( ( ( normal_balance / normal_original_balance ) - 1 ) * 100, 2) ) + "%",
        "normal_inzet": normal_inzet,
        "normal_volume": str( round( normal_trade_volume / 1000, 2) )+"K",
        "period_days": round(len(df)/60/24, 2)
    }

    
    if os.path.exists(f"piramidding_results/{modelfile}/level{original_level}_discount{discount}_result.json"):
        test = ''
        with open(f"piramidding_results/{modelfile}/level{original_level}_discount{discount}_result.json", 'r+') as file:
            test = json.load(file)
            test["results"].append(dictionary)

        with open(f"piramidding_results/{modelfile}/level{original_level}_discount{discount}_result.json", "w") as outfile:
            outfile.write(json.dumps(test, indent = 4))
        
    else:
        with open(f"piramidding_results/{modelfile}/level{original_level}_discount{discount}_result.json", "w") as outfile:
            results = {
                "results":[]
            }
            results['results'].append(dictionary)
            outfile.write(json.dumps(results, indent = 4))
