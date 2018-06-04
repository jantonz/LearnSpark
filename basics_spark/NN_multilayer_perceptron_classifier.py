from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession

spark = SparkSession.builder\
    .master("local")\
    .appName("Neural Network")\
    .getOrCreate()

# Load trainin data
data = spark.read.format("libsvm")\
    .load("data/sample_multiclass_classification_data.txt")

data.show()

# Split the data intro train and test
splits = data.randomSplit([0.6, 0.4], 1234)
train = splits[0]
test = splits[1]

# specify layers for the neural network:
# input layer of size 4 (features), two intermediate of size 5 and 4
# and output of size 3 (classes)
layers = [4, 5, 4, 3]

# create the trainer and set its parameters
trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layers,
                                         blockSize=128, seed=1234)

# train model
model = trainer.fit(train)
# compute accuracy on the test set
result = model.transform(test)
predictionsAndLabel = result.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("test set accuracy = " + str(evaluator.evaluate(predictionsAndLabel)))

spark.stop()
