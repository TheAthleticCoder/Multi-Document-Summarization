To access the full data, download from: https://drive.google.com/drive/folders/1_Rkr3CczybVh6YfzSrrY7YjF-4f4Kw4E?usp=sharing

The `sample_{train,val,test}.csv` are much smaller versions to build the pipeline on. \
**Current Counts** \
Train: 1000 \
Validation: 250 \
Test: 250

-----

The `tf_idf_train.csv` is built on the `sample_train.csv`.

The `sample_train.csv` contains 2 documents for each sample which aren't part of the sample. \

In the `tf_idf.ipynb`, we create a `tf_idf` matrix from all the documents and then with a cluster size of `3`, we cluster each sample to get the documents grouped together. \
We then take the most dominant cluster and assign it to the sample and output the `tf_idf_train.csv` file.