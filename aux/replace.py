import argparse
import random

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('path', type=str)
	parser.add_argument('rate', type=float)
	parser.add_argument('id', type=str)
	args = parser.parse_args()

	path = args.path
	rate = args.rate
	id = args.id


	with open(path, 'r') as f:
		text = f.readlines()
		text = [t.split(' ') for t in text]
		word_cnt = sum([len(t) for t in text])
		num_replaced = word_cnt // 100 * rate

	for _ in range(num_replaced):
		i = random.randint(0, len(text)-1)
		j = random.randint(0, len(text[i])-1)
		while text[i][j] == '[UNK]':
			i = random.randint(0, len(text)-1)
			j = random.randint(0, len(text[i])-1)
		text[i][j] = '[UNK]'
		
	text = [' '.join(t) for t in text]
	with open(f'/Users/takuto/git/subword-polysemy/data/{id}-replaced-{rate}.txt', 'w') as g:
		g.write('\n'.join(text))