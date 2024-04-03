from utils import *
from sklearn.model_selection import train_test_split

def add_history(row):
	if 'history' not in row:
		row['history'] = []
	return row

alpaca_gpt4_en = read_json('ft-data/alpaca_gpt4_data_en.json')
alpaca_gpt4_zh = read_json('ft-data/alpaca_gpt4_data_zh.json')
lima = read_json('ft-data/lima.json')
# oaast_en = read_json('ft-data/oaast_sft.json')
oaast_zh = read_json('ft-data/oaast_sft_zh.json')
self_cognition = read_json('ft-data/self_cognition.json')

all_data = alpaca_gpt4_en + alpaca_gpt4_zh + lima

seen = []
cnt = 0
for row in oaast_zh:
	flat_history = ''
	for pair in row['history']:
		flat_history += pair[0] + pair[1]
	if flat_history + row['instruction'] + row['input'] in seen:
		cnt += 1
		continue
	else:
		seen.append(flat_history + row['instruction'] + row['input'])
		all_data.append(row)
print(cnt, len(oaast_zh))

all_data += [{'instruction': row['instruction'], 'input': row['input'], 'output': row['output'].replace('<NAME>', 'A Happy Bot').replace('<AUTHOR>', 'George Play AI')} for row in self_cognition]
all_data = [add_history(row) for row in all_data]
train, test = train_test_split(all_data, test_size=0.1, random_state=42)
print(len(train), len(test))
to_json(train, 'ft-data/train.json')
to_json(test, 'ft-data/test.json')

