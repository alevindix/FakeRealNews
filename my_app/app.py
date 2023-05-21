#!/usr/bin/env python

from pyspark import SparkContext, SparkConf
import json
from pyspark.ml.classification import NaiveBayesModel

from pyspark.sql import SparkSession

from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizerModel, IDFModel

from pyspark.sql.types import *
from pyspark.sql import Row
from flask import Flask, request, jsonify, render_template

# create a Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
    
# create a SparkSession with a custom name
sparkConf = SparkConf()

sparkConf.setAppName('FakeRealNewsTest')

spark = SparkSession.builder.config(conf=sparkConf).getOrCreate()

sc = spark.sparkContext


tokenizer = Tokenizer(inputCol="text", outputCol='words')
stopWord = StopWordsRemover.load("hdfs://localhost:9000/user/bigdata2022/output/stopWord");
cvModel = CountVectorizerModel.load("hdfs://localhost:9000/user/bigdata2022/output/cvModel");
idfModel = IDFModel.load("hdfs://localhost:9000/user/bigdata2022/output/idfModel");
nbModel = NaiveBayesModel.load("hdfs://localhost:9000/user/bigdata2022/output/nbModel");

def process_input(input_data):

	
	schema = StructType([StructField('text', StringType())])
	rows = [Row(text=input_data)]

	testSet = spark.createDataFrame(rows, schema);

	tokenizedTest = tokenizer.transform(testSet);
	filteredTestSet = stopWord.transform(tokenizedTest);
	vectorizedTest = cvModel.transform(filteredTestSet);
	weightedTest = idfModel.transform(vectorizedTest);
	predictionTest = nbModel.transform(weightedTest);
	
	prediction = predictionTest.first()['prediction']
	
	if float(prediction) == 1.0:
		output="' is a real news"
	else:
		output="' is a fake news"


	# read input_data into a dataframe using spark.read
	# process the data using PySpark functions
	# return the processed data as a pandas dataframe
	return output
    	
# define a route to accept POST requests with input data
@app.route("/process_input", methods=["POST"])
def process_input_route():
    # get the input data from the request
    input_data = request.json["input_data"]
    
    # process the input data using PySpark
    processed_data = process_input(input_data)

    # convert the processed data to JSON and return it to the user
    return processed_data

# start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
