# Fake/Real News Prediction
A system that estimates the truthfulness of a news using Apache Hadoop, PySpark and Flask.
## Dataflow
### Model Training
- Import on HDFS the [fake real news dataset](https://www.kaggle.com/datasets/bjoernjostein/fake-news-data-set)
- Develop an input transformation pipeline that execute the following tasks:
  - Tokenizer
  - Stop word
  - Bag of words
  - Count Vectorizer
  - IDF
 - Train a Naive Bayes classifier
 - Accuracy of the model is 0.92
 - Save the model and the transformation pipeline on HDFS
 ### Flask Web Application
 Created to enter the news, process it and show the output.
 - Load model from HDFS
 - Inser news on user interface
 - Get input with an AJAX request
 - Process input running transformation pipeline
 - Compute prediction and send it with a AJAX response
 - Show the result on user interface
## Usage
1. In ```PySpark/script/runSpark.sh``` set ```MY_PTH="your/path/to/FakeRealNews/PySpark" ``` 
2. Execute ```sh runModel.sh```   
3. Execute ```sh runApp.sh```
4. Open http://localhost:5000 
5. Insert the news and get the prediction
