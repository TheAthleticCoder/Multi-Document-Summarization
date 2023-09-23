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