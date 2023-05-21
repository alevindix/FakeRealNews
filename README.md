# PySpark-FakeRealNews
A system that estimates the truthfulness of a news using Apache Hadoop, PySpark and Flask.
## Dataflow
### Model Training
- Import on HDFS the [fake real news dataset](https://www.kaggle.com/datasets/bjoernjostein/fake-news-data-set)
- Develope an input transformation pipeline that execute the following tasks:
  - Tokenization
  - Stop words
  - Bag of words
  - Vectorization (?)
  - IDF
 - Train a Naive Bayes classifier
 - Accuracy of the model is 0.92
 - Save the model the and transformation pipeline on HDFS
