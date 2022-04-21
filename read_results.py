import json


percentages, negatives, length_in_hours, trades = 0,0,0,0

with open(f"test_results/model_small_10_result.json", 'r+') as file:
    test = json.load(file)
    for item in test['results']:
        # print(item)
        percentages += item['percentage']
        negatives += 1 if item['negative'] else 0
        length_in_hours += item['length_in_hours']
        trades += item['trades']

print(f"""
Average % per hour:         {(percentages/length_in_hours)*100}%
Average % per trade:        {percentages/trades}
Average # trades per hour:  {trades/length_in_hours}
negative results:           {negatives}
""")

