# %%
# Import the PorterStemmer from nltk
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

def cleanData(sentence):
    return stemmer.stem(sentence)

# %%
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculateSimilarity(sentence, doc):
    if doc == []:
        return 0

    vocab = {}

    # For each word in the sentence, add it to the vocabulary dictionary
    for word in sentence:
        vocab[word] = 0

    # Initialize an empty string to hold the document in one sentence
    docInOneSentence = ''

    # For each term in the document, add it to the docInOneSentence string
    # and add each word in the term to the vocabulary dictionary
    for t in doc:
        docInOneSentence += (t + ' ')
        for word in t.split():
            vocab[word]=0

    # Initialize a CountVectorizer with the vocabulary dictionary as the vocabulary
    cv = CountVectorizer(vocabulary=vocab.keys())

    # Fit transform the document into a vector
    docVector = cv.fit_transform([docInOneSentence])
    sentenceVector = cv.fit_transform([sentence])

    return cosine_similarity(docVector, sentenceVector)[0][0]

# %%
def concat(x):
    x = ' '.join(x)
    x = x.split('\n')
    
    # Filter out any strings in the list that are just a space
    x = list(filter(lambda s: not s == ' ', x))
    
    # Remove leading and trailing whitespace from each string in the list
    x = list(map(lambda s: s.strip(), x))
    
    return x

# %%
def get_sentences(texts, sentences, clean, originalSentenceOf):
    # Split the text into sentences
    parts = texts.split('.')
    
    for part in parts:
        cl = cleanData(part)
        
        sentences.append(part)
        clean.append(cl)
        
        # Map the cleaned part to the original part in the originalSentenceOf dictionary
        originalSentenceOf[cl] = part
    
    # Remove duplicates from the clean list by converting it to a set
    setClean = set(clean)

    return setClean

# %%
import signal

# Define a handler function that raises an exception when called
def handler(signum, frame):
    raise Exception("Function execution took too long")

# Set the alarm signal handler to the handler function
# When the alarm signal is received, the handler function will be called
signal.signal(signal.SIGALRM, handler)

# %%
from icecream import ic
import operator

def get_mmr(doc, alpha):
    try:
        # Set an alarm for 60 seconds
        signal.alarm(60)
        
        sentences = []
        clean = []
        originalSentenceOf = {}

        # Get the set of cleaned sentences from the document
        cleanSet = get_sentences(doc, sentences, clean, originalSentenceOf)

        scores = {}
        
        # For each cleaned sentence, calculate its score and add it to the scores dictionary
        for data in clean:
            temp_doc = cleanSet - set([data])
            score = calculateSimilarity(data, list(temp_doc))
            scores[data] = score

        # Calculate the number of sentences to include in the summary
        n = 20 * len(sentences) / 100

        summarySet = []
        
        while n > 0:
            mmr = {}
            
            # For each sentence, calculate its MMR and add it to the mmr dictionary
            for sentence in scores.keys():
                if not sentence in summarySet:
                    mmr[sentence] = alpha * scores[sentence] - (1-alpha) * calculateSimilarity(sentence, summarySet)	
            
            if mmr == {}:
                break
            
            selected = max(mmr.items(), key=operator.itemgetter(1))[0]	
            summarySet.append(selected)
            
            n -= 1

        # Get the original form of the sentences in the summary set
        original = [originalSentenceOf[sentence].strip() for sentence in summarySet]
        
        # Return the original sentences
        return original
    except Exception as e:
        # If an exception occurs, return an empty list
        return []

# %%
import wandb

# Create an API object to interact with the Weights & Biases service
api = wandb.Api()

artifact = api.artifact('ire-shshsh/mdes/multi_news:v0', type='dataset')

path_to_file = artifact.download()

# %%
# path_to_file = './our_dataset - Sheet1.csv'

# %%
# import os
# for file in os.listdir(path_to_file):
#     # if file is a csv file
#     if file.endswith('.csv'):
#         # get the file wihout the extension
#         file = file.split('.')[0]
#         print(file)

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
# Initialize a Weights & Biases run
run = wandb.init(entity='ire-shshsh', project='mmr', job_type='mmr')

for alpha in [0.2, 0.5, 0.8]:
    artifact = wandb.Artifact(name=f'multi_news_{alpha}', type='dataset')
    for file in ['train', 'validation', 'test']:
        artifact.add_file(f'{file}_{alpha}.csv')
    run.log_artifact(artifact)

run.finish()

# %%
# import wandb
# wandb.finish()

# %%
# import pandas as pd
# df = pd.read_csv(path_to_file)
# df['documents'] = df['documents'].apply(lambda x: eval(x))
# df['concat_doc'] = df['documents'].apply(lambda x: concat(x))
# df.head()

# %%
# import pandas as pd
# from icecream import ic

# df = pd.read_csv(path_to_file)
# df['concat_doc'] = df['doc1'] + ' ' + df['doc2'] + ' ' + df['doc3']
# df.drop(['doc1', 'doc2', 'doc3'], axis=1, inplace=True)
# df

# %%
# # Import necessary libraries
# import os
# import pandas as pd
# from tqdm import tqdm
# tqdm.pandas()

# # Initialize a Weights & Biases run
# # run = wandb.init(entity='ire-shshsh', project='mmr', job_type='mmr')

# # Loop over different alpha values
# for alpha in [0.2, 0.5, 0.8]:
#     # Load the data from the CSV file
#     # df = pd.read_csv(path_to_file)

#     # df['abstracts'] = df['abstracts'].progress_apply(lambda x: eval(x))

#     # df['concat_doc'] = df['abstracts'].progress_apply(lambda x: concat(x))

#     # Concatenate the documents in each row
#     # df['concat_doc'] = df['doc1'] + df['doc2'] + df['doc3']

#     # Initialize an empty 'mmr' column
#     df['mmr'] = ''

#     # Write the header to the file and remove the dropped columns
#     df.drop(columns=['concat_docs']).iloc[0:0].to_csv(
#         f'train_{alpha}.csv', index=False)

#     for i, row in tqdm(df.iterrows()):
#         df.at[i, 'mmr'] = get_mmr(df.at[i, 'concat_doc'], alpha)

#         # If the MMR is an empty list, skip this row
#         if df.at[i, 'mmr'] == []:
#             continue

#         row = df.iloc[i].drop(['concat_doc'])

#         # Save the current row to the file
#         row.to_frame().T.to_csv(
#             f'test_{alpha}.csv', mode='a', header=False, index=False)

#     # Drop the 'concat_doc' and 'name' columns from the DataFrame
#     # df.drop(['concat_doc', 'name'], axis=1, inplace=True)

#     # Save the DataFrame to a CSV file
#     # df.to_csv(f'test_{alpha}.csv', index=False)

#     artifact = wandb.Artifact(name=f'multi_news_{alpha}', type='dataset')
#     artifact.add_file(f'train_{alpha}.csv')

#     # run.log_artifact(artifact)

# # wandb.finish()

# %%
# Uncomment to print the time taken by the process
# print str(time.time() - start)

# Uncomment to print the summary
# print ('\nSummary:\n')
# for sentence in summarySet:
# 	print (originalSentenceOf [sentence].lstrip(' '))
# print()

# Print a separator
# print '============================================================='
# print '\nOriginal Passages:\n'

# Import the termcolor module for colored output
# from termcolor import colored

# For each sentence in the cleaned data
# for sentence in clean:
# 	# If the sentence is in the summary set, print it in red
# 	if sentence in summarySet:
# 		print colored(originalSentenceOf[sentence].lstrip(' '), 'red')
# 	# Otherwise, print it in the default color
# 	else:
# 		print originalSentenceOf[sentence].lstrip(' ')


