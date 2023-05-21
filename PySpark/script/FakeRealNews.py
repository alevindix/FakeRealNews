#!/usr/bin/env python

# coding: utf-8



# In[1]:





from pyspark import SparkContext, SparkConf

from pyspark.sql.functions import when, lit

from pyspark.ml.classification import NaiveBayes, NaiveBayesModel

from pyspark.sql import SparkSession

from pyspark.ml.evaluation import BinaryClassificationEvaluator

from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF, CountVectorizerModel, IDFModel


# In[2]:





# SparkSession is now the entry point of Spark

# SparkSession can also be construed as gateway to spark libraries

# create instance of spark class

sparkConf = SparkConf()

sparkConf.setAppName('FakeRealNews')

spark = SparkSession.builder.config(conf=sparkConf).getOrCreate()

sc = spark.sparkContext



# create spark dataframe of input csv file

path = 'hdfs://localhost:9000/user/bigdata2022/input'

df = spark.read.csv(path + '/df.csv', inferSchema=True, header=True, escape='"', multiLine=True)



df = df.withColumn('label', when(df.label == 'fake', lit(0)).otherwise(1))





# In[3]:





# TOKENIZER

tokenizer = Tokenizer(inputCol="text", outputCol='words')

tokenized = tokenizer.transform(df)





# In[15]:





# STOP WORD

stopWord = StopWordsRemover(stopWords=StopWordsRemover.loadDefaultStopWords("english"), inputCol=tokenizer.getOutputCol(), outputCol="filteredWords")

filtered = stopWord.transform(tokenized)



stopWord.write().overwrite().save("hdfs://localhost:9000/user/bigdata2022/output/stopWord")





# In[5]:





# COUNT VECTORIZER

cv = CountVectorizer(inputCol=stopWord.getOutputCol(), outputCol="vectors")

cvModel = cv.fit(filtered)

vectorized = cvModel.transform(filtered)



cvModel.write().overwrite().save("hdfs://localhost:9000/user/bigdata2022/output/cvModel")





# In[6]:





# IDF

idf = IDF(inputCol=cv.getOutputCol(), outputCol="features")

idfModel = idf.fit(vectorized)

weighted = idfModel.transform(vectorized)



weighted.show(5)

idfModel.write().overwrite().save("hdfs://localhost:9000/user/bigdata2022/output/idfModel");





# In[7]:





train, test = weighted.randomSplit([0.8, 0.2])



train.show(5)





# In[8]:





# NAIVE BAYES

nb = NaiveBayes(modelType="multinomial")

nbModel = nb.fit(train)

nbModel.write().overwrite().save("hdfs://localhost:9000/user/bigdata2022/output/nbModel");






predictionTrain = nbModel.transform(train)

predictionTest = nbModel.transform(test)







# BINARY CLASSIFICATION EVALUATOR

evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol = "label")



accuracyTrain = evaluator.evaluate(predictionTrain)

accuracyTest = evaluator.evaluate(predictionTest)



print(accuracyTrain)

print(accuracyTest)





# In[ ]:





sc.stop()


