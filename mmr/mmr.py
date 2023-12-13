# %%
import re
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import operator

# %%
# create stemmer
stemmer = PorterStemmer()

def cleanData(sentence):
	#sentence = re.sub('[^A-Za-z0-9 ]+', '', sentence)
	#sentence filter(None, re.split("[.!?", setence))
	ret = []
	sentence = stemmer.stem(sentence)	
	for word in sentence.split():
		ret.append(word)
	return " ".join(ret)

# %%
def getVectorSpace(cleanSet):
	vocab = {}
	for data in cleanSet:
		for word in data.split():
			vocab[word] = 0
	return vocab.key

# %%
def calculateSimilarity(sentence, doc):
	if doc == []:
		return 0
	vocab = {}
	for word in sentence:
		vocab[word] = 0
	
	docInOneSentence = '';
	for t in doc:
		docInOneSentence += (t + ' ')
		for word in t.split():
			vocab[word]=0	
	
	cv = CountVectorizer(vocabulary=vocab.keys())

	docVector = cv.fit_transform([docInOneSentence])
	sentenceVector = cv.fit_transform([sentence])
	return cosine_similarity(docVector, sentenceVector)[0][0]

# %%
def concat(x):
    # print(len(x), len(x[:-2]))
    x = ' '.join(x)
    x = x.split('\n')
    x = list(filter(lambda s: not s == ' ', x))
    x = list(map(lambda s: s.strip(), x))
    return x

# %%
def get_sentences(texts, sentences, clean, originalSentenceOf):
    # for line in texts:
    #     parts = line.split('.')
    #     for part in parts:
    #         cl = cleanData(part)
    #         sentences.append(part)
    #         clean.append(cl)
    #         originalSentenceOf[cl] = part		
    # parts = texts.split('.')
    # print('texts.split')
    for part in texts:
        cl = cleanData(part)
        sentences.append(part)
        clean.append(cl)
        originalSentenceOf[cl] = part		
    setClean = set(clean)
    return setClean

# %%
import signal

# Define the handler function to raise an exception
def handler(signum, frame):
    raise Exception("Function execution took too long")

# Set the signal handler
signal.signal(signal.SIGALRM, handler)

# %%
# from termcolor import colored
from icecream import ic
def get_mmr(doc, alpha):
	try:
		# set an alarm for 60 seconds
		signal.alarm(60)
		sentences = []
		clean = []
		originalSentenceOf = {}

		cleanSet = get_sentences(doc, sentences, clean, originalSentenceOf)

		scores = {}
		for data in clean:
			temp_doc = cleanSet - set([data])
			score = calculateSimilarity(data, list(temp_doc))
			scores[data] = score

		n = 20 * len(sentences) / 100
		# ic(n)
		# ic(scores.keys())
		summarySet = []
		while n > 0:
			mmr = {}
			for sentence in scores.keys():
				if not sentence in summarySet:
					mmr[sentence] = alpha * scores[sentence] - (1-alpha) * calculateSimilarity(sentence, summarySet)	
			if mmr == {}:
				break
			selected = max(mmr.items(), key=operator.itemgetter(1))[0]	
			summarySet.append(selected)
			n -= 1

		original = [originalSentenceOf[sentence].strip() for sentence in summarySet]
		# print ('\nSummary:\n')
		# for sentence in summarySet:
		# 	print (originalSentenceOf [sentence].lstrip(' '))
		# print()

		# print ('=============================================================')
		# print ('\nOriginal Passages:\n')

		# for sentence in clean:
		# 	if sentence in summarySet:
		# 		print (colored(originalSentenceOf[sentence].lstrip(' '), 'red'))
		# 	else:
		# 		print (originalSentenceOf[sentence].lstrip(' '))
		
		return original
	except Exception as e:
		print('Exception')
		print(e)
		return []

# %%
import wandb

api = wandb.Api()
artifact = api.artifact('ire-shshsh/mdes/multi_news:v0', type='dataset')
path_to_file = artifact.download()

# %%
# path_to_file = './our_dataset - Sheet1.csv'
# path_to_file = '../our_dataset.csv'
path_to_file

# %%
import pandas as pd
import os

df = pd.read_csv(os.path.join(path_to_file, 'test.csv'))
df['documents'] = df['documents'].apply(lambda x: eval(x))
df['concat_doc'] = df['documents'].apply(lambda x: concat(x))

# %%
# get_mmr(df.at[0, 'concat_doc'], alpha=0.8)

# %%
# import pandas as pd
# from icecream import ic

# df = pd.read_csv(path_to_file)
# df['concat_doc'] = df['doc1'] + ' ' + df['doc2'] + ' ' + df['doc3']
# df.drop(['doc1', 'doc2', 'doc3'], axis=1, inplace=True)
# df

# %%
import os
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

# for all files in path_to_file
for file in os.listdir(path_to_file):
    # if file is a csv file
    if file.endswith('.csv'):
        # read the csv file
        df = pd.read_csv(os.path.join(path_to_file, file))

        df['documents'] = df['documents'].apply(lambda x: eval(x))
        df['concat_doc'] = df['documents'].apply(lambda x: concat(x))

        # Loop over different alpha values
        for alpha in [0.2, 0.5, 0.8]:
            # Initialize an empty 'mmr' column
            df['mmr'] = ''

            file_name = file.split('.')[0]

            # Write the header to the file and remove the dropped columns
            df.drop(columns=['concat_doc']).iloc[0:0].to_csv(f'{file_name}_{alpha}.csv', index=False)

            for i, row in tqdm(df.iterrows()):
                df.at[i, 'mmr'] = get_mmr(df.at[i, 'concat_doc'], alpha)

                # If the MMR is an empty list, skip this row
                if df.at[i, 'mmr'] == []:
                    continue

                row = df.iloc[i].drop(['concat_doc'])

                # Save the current row to the file
                row.to_frame().T.to_csv(f'{file_name}_{alpha}.csv', mode='a', header=False, index=False)

# %%
import wandb
run = wandb.init(entity='ire-shshsh', project='mmr', job_type='mmr')

for alpha in [0.2, 0.5, 0.8]:
    artifact = wandb.Artifact(name=f'multi_news_{alpha}', type='dataset')
    for file in ['train', 'validation', 'test']:
        artifact.add_file(f'{file}_{alpha}.csv')
    run.log_artifact(artifact)

run.finish()

# %%

#print str(time.time() - start)
	
# print ('\nSummary:\n')
# for sentence in summarySet:
# 	print (originalSentenceOf [sentence].lstrip(' '))
# print()

# print '============================================================='
# print '\nOriginal Passages:\n'
# from termcolor import colored

# for sentence in clean:
# 	if sentence in summarySet:
# 		print colored(originalSentenceOf[sentence].lstrip(' '), 'red')
# 	else:
# 		print originalSentenceOf[sentence].lstrip(' ')
