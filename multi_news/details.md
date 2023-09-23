## Data

To access the full data, download from: https://drive.google.com/drive/folders/1_Rkr3CczybVh6YfzSrrY7YjF-4f4Kw4E?usp=sharing

## Data Handling

We take a sample of the complete data and built smaller versions. \
The `sample_{train,val,test}.csv` files contain the following columns: \
- `documents`: A list of documents for each sample.
- `num_documents`: The number of documents for each sample.
- `summary`: The summary of the sample.

The `sample_{train,val,test}.csv` contains 2 documents for each sample which aren't part of the sample.

**Current Counts** \
Train: 500 \
Validation: 250 \
Test: 250

## Clustering Documents

### TF-IDF

#### Method 1

We take all the unique documents present in the corpus and create `one` tf-idf matrix. \
We then take the `tf-idf` matrix and then fit each sample to the matrix and get the most dominant cluster. \
We then assign the cluster to the sample and output the `tf_idf_method_1_sample_{train,validation,test}.csv` file.

> We use `TfidfVectorizer` from sklearn to create the `tf-idf` matrix. \
> We use a cluster size of 3 for the `KMeans` clustering algorithm.

The `tf_idf_method_1_sample_{train,validation,test}.csv` files contain the following columns:
- Index of the sample.
- `documents`: A list of documents for each sample.
- `num_documents`: The number of documents for each sample.
- `summary`: The summary of the sample.

#### Method 2

We only take the documents present in one sample aka data point and create a `tf-idf` matrix for it. \
We then try to cluster the sample based on this matrix and take the dominant cluster. \
We then assign the cluster to the sample and output the `tf_idf_method_2_sample_{train,validation,test}.csv` file.

> We use `TfidfVectorizer` from sklearn to create the `tf-idf` matrix. \
> We use a cluster size of 3 for the `KMeans` clustering algorithm.

The `tf_idf_method_2_sample_{train,validation,test}.csv` files contain the following columns:
- Index of the sample.
- `documents`: A list of documents for each sample.
- `num_documents`: The number of documents for each sample.
- `summary`: The summary of the sample.