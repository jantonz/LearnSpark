import pandas as pd
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator

spark = SparkSession.builder\
    .master("local")\
    .appName("Churn Prediction")\
    .getOrCreate()

train_data = spark.read.load("data/churn-bigml-80.csv",
                             format="com.databricks.spark.csv",
                             header="true",
                             inferSchema="true")

test_data = spark.read.load("data/churn-bigml-20.csv",
                            format="com.databricks.spark.csv",
                            header="true",
                            inferSchema="true")

# train_data.cache()  # cachegem train_data en memòria (MEMORY_ONLY)

# train_data.printSchema()  # Noms de columnes i tipus
# train_data.show(5)
# train_data.describe().toPandas().transpose()

# Correlation analysis
numeric_features = [t[0] for t in
                    train_data.dtypes if t[1] == "int" or t[1] == "double"]

sampled_data = train_data.select(numeric_features) \
    .sample(False, 0.10).toPandas()

axs = pd.plotting.scatter_matrix(sampled_data, figsize=(12, 12))

# Rotate axis labels and remove axis ticks
n = len(sampled_data.columns)
for i in range(n):
    v = axs[i, 0]
    v.yaxis.label.set_rotation(0)
    v.yaxis.label.set_ha('right')
    v.set_yticks(())
    h = axs[n-1, i]
    h.xaxis.label.set_rotation(90)
    h.set_xticks(())

# Change "yes", "no", "true", "false" to 1 and 0.
# Drop Area Code and too-closely-related variables (the "total XX charge"
# family)

train_data = train_data.drop("State").drop("Area code") \
                       .drop("Total day charge").drop("Total eve charge") \
                       .drop("Total night charge").drop("Total intl charge") \
                       .withColumn("Churn",
                                   train_data["Churn"].cast(DoubleType())) \
                       .withColumn("International plan",
                                   train_data["International plan"]
                                   .cast("boolean").cast(DoubleType())) \
                       .withColumn("Voice mail plan",
                                   train_data["Voice mail plan"]
                                   .cast("boolean").cast(DoubleType())) \

test_data = test_data.drop("State").drop("Area code") \
                       .drop("Total day charge").drop("Total eve charge") \
                       .drop("Total night charge").drop("Total intl charge") \
                       .withColumn("Churn",
                                   test_data["Churn"].cast(DoubleType())) \
                       .withColumn("International plan",
                                   test_data["International plan"]
                                   .cast("boolean").cast(DoubleType())) \
                       .withColumn("Voice mail plan",
                                   test_data["Voice mail plan"]
                                   .cast("boolean").cast(DoubleType())) \


# df = train_data
# df.dtypes
# df.show()
# df.head()
# df.first()
# df.take(2)
# df.schema
# df.printSchema()
# df.describe().show()
# df.columns
# df.count()
# df.distinct().count()
# df.dropDuplicates().count()
# df.explain()
# df.select(df["Customer service calls"] > 0).show()
# df.filter(df["Customer service calls"] > 0).show()
# df.groupBy("Churn").count().show()
# df.sort("Churn", ascending=False).show()
# df.orderBy(["Churn", "Customer service calls"], ascending=[1, 0]).show()
# df.na.drop()
# df.na.fill(50)
# df.na.replace(0, 20).show()  # Res a veure amb na, això canvia els 0 per 20

# Així queda:
pd.DataFrame(train_data.take(5), columns=train_data.columns).transpose()

#########################
# VectorAssembler!!!
#########################
train_data.columns
vecAssembler = VectorAssembler(inputCols=['Account length',
                                          'International plan',
                                          'Voice mail plan',
                                          'Number vmail messages',
                                          'Total day minutes',
                                          'Total day calls',
                                          'Total eve minutes',
                                          'Total eve calls',
                                          'Total night minutes',
                                          'Total night calls',
                                          'Total intl minutes',
                                          'Total intl calls',
                                          'Customer service calls'],
                               outputCol="features")

df_train = vecAssembler.transform(train_data)
df_test = vecAssembler.transform(test_data)

dt = DecisionTreeClassifier(labelCol="Churn", featuresCol="features")

pipeline = Pipeline(stages=[dt])

model = pipeline.fit(df_train)

predictions = model.transform(df_test)

predictions.select("prediction", "Churn", "features").show()

evaluator = BinaryClassificationEvaluator(
    labelCol="Churn", rawPredictionCol="prediction")

evaluator.evaluate(predictions)
