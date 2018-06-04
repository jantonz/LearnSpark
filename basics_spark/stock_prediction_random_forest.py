import numpy
import pandas as pd
import matplotlib.pyplot as plt
from pylab import *
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, concat, col, lit
from pyspark.sql.types import IntegerType, ArrayType, StringType, DoubleType
import string
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer, CountVectorizer, Tokenizer, StopWordsRemover, NGram
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = SparkSession.builder\
    .master("local")\
    .appName("Stock Prediction")\
    .getOrCreate()

#the data file is read from HDFS
#the file stockMarketAndNewsData.csv a modified version of the file combined_News_DJIA.csv, whereby the newlines that are wrongly introduced in the data are removed

data = spark.read.load("data/Combined_News_DJIA.csv",
                          delimiter=',',
                          format='com.databricks.spark.csv',
                          header='true',
                          inferSchema='true')

#replace null values with empty string
data = data.na.fill(' ')

# Only the columns that represent the news
newsColumns = [x for x in data.columns if x not in ['Date', 'Label']]

#merge news from different news sources per day

data = data.withColumn("allNews", data.Top1)
for i in range(2, len(newsColumns)+1):
    colName = 'Top' + str(i)
    data = data.withColumn('allNews', concat(col("allNews"), lit(" "), col(colName)))

#remove puntuation marks from the news

removePunctuation = udf(lambda x: ''.join([' ' if ch in string.punctuation else ch for ch in x]))
data = data.withColumn('allNews', removePunctuation(data.allNews))

#split the news into words

splitNews = udf(lambda s: [x for x in s.split(' ') if (x != u'' and len(x) >= 2)], ArrayType(StringType(), True))
data = data.withColumn('words', splitNews(data.allNews)).select('Date', 'label', 'words')

#remove the stop words

myStopwordRemover = StopWordsRemover(inputCol="words", outputCol="stopRemoved")
data = myStopwordRemover.transform(data)

# Create ngrams of size 2

myngram = NGram(inputCol="stopRemoved", outputCol="ngrams", n=2)
data = myngram.transform(data)
data = data.withColumn('ngrams', data.ngrams.cast(ArrayType(StringType(), True)))

# Apply count vectorizer to convert to vector of counts of the ngrams

myCountVectorizer = CountVectorizer(inputCol="ngrams", outputCol="countVect", minDF=1.0)
data = myCountVectorizer.fit(data).transform(data)

# Transform the label using StringINdexer

si_label = StringIndexer(inputCol="label", outputCol="label2", handleInvalid="skip")
data = si_label.fit(data).transform(data)
data.drop('label')
data = data.withColumn('label', data.label2)

# Divide into training and test data

trainData = data[data['Date'] < '20150101']
testData = data[data['Date'] >= '20141231']

# define the random forest classifier model

rf = RandomForestClassifier(labelCol="label", featuresCol="countVect", numTrees=3, maxDepth=4, maxBins=200)
# perform a grid search on a set of parameter values

grid = ParamGridBuilder().addGrid(rf.numTrees, [2, 5])\
                         .addGrid(rf.maxDepth, [2, 5])\
                         .build()
evaluator = BinaryClassificationEvaluator()
cv = CrossValidator(estimator=rf, estimatorParamMaps=grid, evaluator=evaluator)
cvModel = cv.fit(trainData)
evaluator.evaluate(cvModel.transform(testData))
