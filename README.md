# Sentence-Sentiment-Analysis

## DataSet
There are variuos type of getting dataset for this task. One of them is to use Twitter API https://developer.twitter.com/en to stream live data and store it. The other way is to use competition dataset. In this project, dataset used from https://www.kaggle.com/kazanova/sentiment140. Since Twitter didn't approve my developer account the first approach is TODO 

## Preprocessing 
1) Run preprocess.py <raw-csv-path> on both train and test data. This will generate a preprocessed version of the dataset.
2) Run stats.py <preprocessed-csv-path> where <preprocessed-csv-path> is the path of csv generated from preprocess.py. This gives general statistical information about the dataset and will two pickle files which are the frequency distribution of unigrams and bigrams in the training dataset.
3) Run model.py
  
## Files
1) dataset_manual_raw.csv - raw dataset from link https://www.kaggle.com/kazanova/sentiment140
2) freqdist - frequency list 
3) freqdist-bi - frequency of bigrams  
4) glove-seeds - Glove seeds from https://github.com/stanfordnlp/GloVe
