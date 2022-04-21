import pandas as pd
import json
import os

####
# only run new calculations
# run all possible levels concurrently
# upgrade fee levels during run if trade volume is high enough
####


modelfile = "model_small_10"
dir = f"/home/joren/Coding/cryptodata/predictions/{modelfile}/"
column_names = ['open','close','high','low','volume','adosc','atr','macd','macd_signal','macd_hist','mfi','upper_band','middle_band','lower_band','rsi','difference_low_high','difference_open_close','target']

def filenames():
    filenames = []
    for file in os.listdir(dir):
        filename = os.fsdecode(file)
        filenames.append(filename)
        # random.shuffle(filenames)
    return filenames
files = filenames()

levels = {
    1: [0.004, 0.004],
    2: [0.0035, 0.0035],
    3: [0.0015, 0.0025],
    4: [0.001, 0.0016],
    5: [0.0009, 0.0015],
    6: [0.0008, 0.0014],
    7: [0.0007, 0.0013],
    8: [0.0006, 0.0012],
    9: [0.0004, 0.001]
}
level = 1
# maker, taker = 0.001, 0.001
maker, taker = levels[level]
# print(f"fees: {maker} & {taker}")

discount = 0
if discount > 0:
    maker *= (1-discount)
    taker *= (1-discount)

for file in files:
    # print(file)
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

    average_buy_price = 0
    last_buy_price = 0

    position = 0
    buy_count = 0

    balance = 10000
    original_balance = balance
    inzet = 50
    trade_volume = 0

    for index, row in df.iterrows():
        if position > 0:
            if row['predictions'] == 1 and row['close'] < last_buy_price and balance >= 1:
                #piramidding
                buy_count += 1
                position += inzet - ( taker * inzet )
                trade_volume += inzet - ( taker * inzet )

                last_buy_price = row['close']
                average_buy_price = ( row['close'] + (average_buy_price*(buy_count-1)) ) / buy_count

                balance -= inzet
                
        if row['predictions'] == 1 and position == 0:
            #initial buy
            # print('first buy')
            buy_count += 1
            position += inzet - ( taker * inzet )
            trade_volume += inzet - ( taker * inzet )
            last_buy_price = row['close']
            average_buy_price = row['close']
            balance -= inzet


        if row['predictions'] == 2 and position > 0:
            # sell
            percentage = average_buy_price / row['close']

            balance += (position-(maker*position)) * percentage
            trade_volume += position-(maker*position)

            position = 0
            buy_count = 0
        
    
    # print(f"Piramidding: {balance}")

################### without piramidding ###################

    position = 0
    buy_price = 0
    normal_balance = 10000
    normal_original_balance = normal_balance
    normal_inzet = 50
    normal_trade_volume = 0

    for index, row in df.iterrows():

        if row['predictions'] == 1 and position == 0 and normal_balance > 0:
            position = normal_inzet - ( taker * normal_inzet )
            normal_trade_volume += normal_inzet - ( taker * normal_inzet )
            buy_price = row['close']
            normal_balance -= normal_inzet

        if row['predictions'] == 2 and position > 0:
            percentage = buy_price / row['close']
            normal_balance += (position-maker) * percentage
            normal_trade_volume += position-(maker*position)
            position = 0
    
    # print(f"Normal: {normal_balance}")

################### Write to JSON ###################

    print({
        "file": file,
        "buy & hold": round(buy_hold, 2),
        "piramidding_start_balance": original_balance,
        "piramidding": str( round( ( ( balance / original_balance ) - 1 ) * 100, 2) ) + "%",
        "piramidding_inzet": inzet,
        "piramidding_volume": str( round(trade_volume/1000, 2) )+"K",
        "normal_start_balance": normal_original_balance,
        "normal": str( round( ( ( normal_balance / normal_original_balance ) - 1 ) * 100, 2) ) + "%",
        "normal_inzet": normal_inzet,
        "normal_volume": str( round(normal_trade_volume/1000, 2) )+"K",
        "period_days": round(len(df)/60/24, 2)
    })

    dictionary ={
        "file": file,
        "buy & hold": round(buy_hold, 2),
        "piramidding_start_balance": original_balance,
        "piramidding": str( round( ( ( balance / original_balance ) - 1 ) * 100, 2 ) ) + "%",
        "piramidding_inzet": inzet,
        "piramidding_volume": str( round(trade_volume/1000, 2))+"K",
        "normal_start_balance": normal_original_balance,
        "normal": str( round( ( ( normal_balance / normal_original_balance ) - 1 ) * 100, 2) ) + "%",
        "normal_inzet": normal_inzet,
        "normal_volume": str( round( normal_trade_volume/1000, 2) )+"K",
        "period_days": round(len(df)/60/24, 2)
    }

    if not os.path.exists(f"piramidding_results/{modelfile}"):
        os.makedirs(f"piramidding_results/{modelfile}/")

    if os.path.exists(f"piramidding_results/{modelfile}/level{level}_discount{discount}_result.json"):
        test = ''
        with open(f"piramidding_results/{modelfile}/level{level}_discount{discount}_result.json", 'r+') as file:
            test = json.load(file)
            test["results"].append(dictionary)

        with open(f"piramidding_results/{modelfile}/level{level}_discount{discount}_result.json", "w") as outfile:
            outfile.write(json.dumps(test, indent = 4))
        
    else:
        with open(f"piramidding_results/{modelfile}/level{level}_discount{discount}_result.json", "w") as outfile:
            results = {
                "results":[]
            }
            results['results'].append(dictionary)
            outfile.write(json.dumps(results, indent = 4))
