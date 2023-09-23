## Clustering Documents

### TF-IDF

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