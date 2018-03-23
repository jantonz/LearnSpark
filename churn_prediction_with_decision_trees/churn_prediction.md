
Please refer to Ben Sadeghi's [post](https://mapr.com/blog/churn-prediction-pyspark-using-mllib-and-ml-packages/) for the complete tutorial. This is a mere update, as some lines of code, functions, dependencies, etc. did not work as of March 2018... Spark environment is moving fast!
So, for everyone's sake, read the main post first, then come back when you find yourself in trouble. 

Let's first import some of the needed libraries. We will, however, be importing libraries as we need them.


```python
import pandas as pd
from pyspark.sql import SparkSession
```

### Starting a Spark session
Let's first start fresh with a brand-new SparkSession.


```python
spark = SparkSession.builder\
    .master("local")\
    .appName("Churn Prediction")\
    .getOrCreate()
```

### Reading the data

Let's read the data, which is already split into training and test set.


```python
train_data = spark.read.load("data/churn-bigml-80.csv",
                             format="com.databricks.spark.csv",
                             header="true",
                             inferSchema="true")

test_data = spark.read.load("data/churn-bigml-20.csv",
                            format="com.databricks.spark.csv",
                            header="true",
                            inferSchema="true")
```

### Data exploration and data wrangling

This train_data.printSchema() will give us some very important info.


```python
train_data.printSchema()
```

    root
     |-- State: string (nullable = true)
     |-- Account length: integer (nullable = true)
     |-- Area code: integer (nullable = true)
     |-- International plan: string (nullable = true)
     |-- Voice mail plan: string (nullable = true)
     |-- Number vmail messages: integer (nullable = true)
     |-- Total day minutes: double (nullable = true)
     |-- Total day calls: integer (nullable = true)
     |-- Total day charge: double (nullable = true)
     |-- Total eve minutes: double (nullable = true)
     |-- Total eve calls: integer (nullable = true)
     |-- Total eve charge: double (nullable = true)
     |-- Total night minutes: double (nullable = true)
     |-- Total night calls: integer (nullable = true)
     |-- Total night charge: double (nullable = true)
     |-- Total intl minutes: double (nullable = true)
     |-- Total intl calls: integer (nullable = true)
     |-- Total intl charge: double (nullable = true)
     |-- Customer service calls: integer (nullable = true)
     |-- Churn: boolean (nullable = true)
    


Some column types are good news, some are bad.

We move the spark DataFrame out to a Pandas dataframe just because it looks prettier when printed out. So we can get a feel of how our data is.


```python
pd.DataFrame(train_data.take(5), columns=train_data.columns).transpose()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>State</th>
      <td>KS</td>
      <td>OH</td>
      <td>NJ</td>
      <td>OH</td>
      <td>OK</td>
    </tr>
    <tr>
      <th>Account length</th>
      <td>128</td>
      <td>107</td>
      <td>137</td>
      <td>84</td>
      <td>75</td>
    </tr>
    <tr>
      <th>Area code</th>
      <td>415</td>
      <td>415</td>
      <td>415</td>
      <td>408</td>
      <td>415</td>
    </tr>
    <tr>
      <th>International plan</th>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>Voice mail plan</th>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
    </tr>
    <tr>
      <th>Number vmail messages</th>
      <td>25</td>
      <td>26</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Total day minutes</th>
      <td>265.1</td>
      <td>161.6</td>
      <td>243.4</td>
      <td>299.4</td>
      <td>166.7</td>
    </tr>
    <tr>
      <th>Total day calls</th>
      <td>110</td>
      <td>123</td>
      <td>114</td>
      <td>71</td>
      <td>113</td>
    </tr>
    <tr>
      <th>Total day charge</th>
      <td>45.07</td>
      <td>27.47</td>
      <td>41.38</td>
      <td>50.9</td>
      <td>28.34</td>
    </tr>
    <tr>
      <th>Total eve minutes</th>
      <td>197.4</td>
      <td>195.5</td>
      <td>121.2</td>
      <td>61.9</td>
      <td>148.3</td>
    </tr>
    <tr>
      <th>Total eve calls</th>
      <td>99</td>
      <td>103</td>
      <td>110</td>
      <td>88</td>
      <td>122</td>
    </tr>
    <tr>
      <th>Total eve charge</th>
      <td>16.78</td>
      <td>16.62</td>
      <td>10.3</td>
      <td>5.26</td>
      <td>12.61</td>
    </tr>
    <tr>
      <th>Total night minutes</th>
      <td>244.7</td>
      <td>254.4</td>
      <td>162.6</td>
      <td>196.9</td>
      <td>186.9</td>
    </tr>
    <tr>
      <th>Total night calls</th>
      <td>91</td>
      <td>103</td>
      <td>104</td>
      <td>89</td>
      <td>121</td>
    </tr>
    <tr>
      <th>Total night charge</th>
      <td>11.01</td>
      <td>11.45</td>
      <td>7.32</td>
      <td>8.86</td>
      <td>8.41</td>
    </tr>
    <tr>
      <th>Total intl minutes</th>
      <td>10</td>
      <td>13.7</td>
      <td>12.2</td>
      <td>6.6</td>
      <td>10.1</td>
    </tr>
    <tr>
      <th>Total intl calls</th>
      <td>3</td>
      <td>3</td>
      <td>5</td>
      <td>7</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Total intl charge</th>
      <td>2.7</td>
      <td>3.7</td>
      <td>3.29</td>
      <td>1.78</td>
      <td>2.73</td>
    </tr>
    <tr>
      <th>Customer service calls</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Churn</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



This are some summary statistics on our data.


```python
train_data.describe().toPandas().transpose()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>summary</th>
      <td>count</td>
      <td>mean</td>
      <td>stddev</td>
      <td>min</td>
      <td>max</td>
    </tr>
    <tr>
      <th>State</th>
      <td>2666</td>
      <td>None</td>
      <td>None</td>
      <td>AK</td>
      <td>WY</td>
    </tr>
    <tr>
      <th>Account length</th>
      <td>2666</td>
      <td>100.62040510127532</td>
      <td>39.56397365334985</td>
      <td>1</td>
      <td>243</td>
    </tr>
    <tr>
      <th>Area code</th>
      <td>2666</td>
      <td>437.43885971492875</td>
      <td>42.521018019427174</td>
      <td>408</td>
      <td>510</td>
    </tr>
    <tr>
      <th>International plan</th>
      <td>2666</td>
      <td>None</td>
      <td>None</td>
      <td>No</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>Voice mail plan</th>
      <td>2666</td>
      <td>None</td>
      <td>None</td>
      <td>No</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>Number vmail messages</th>
      <td>2666</td>
      <td>8.021755438859715</td>
      <td>13.61227701829193</td>
      <td>0</td>
      <td>50</td>
    </tr>
    <tr>
      <th>Total day minutes</th>
      <td>2666</td>
      <td>179.48162040510135</td>
      <td>54.21035022086982</td>
      <td>0.0</td>
      <td>350.8</td>
    </tr>
    <tr>
      <th>Total day calls</th>
      <td>2666</td>
      <td>100.31020255063765</td>
      <td>19.988162186059512</td>
      <td>0</td>
      <td>160</td>
    </tr>
    <tr>
      <th>Total day charge</th>
      <td>2666</td>
      <td>30.512404351087813</td>
      <td>9.215732907163497</td>
      <td>0.0</td>
      <td>59.64</td>
    </tr>
    <tr>
      <th>Total eve minutes</th>
      <td>2666</td>
      <td>200.38615903976006</td>
      <td>50.95151511764598</td>
      <td>0.0</td>
      <td>363.7</td>
    </tr>
    <tr>
      <th>Total eve calls</th>
      <td>2666</td>
      <td>100.02363090772693</td>
      <td>20.16144511531889</td>
      <td>0</td>
      <td>170</td>
    </tr>
    <tr>
      <th>Total eve charge</th>
      <td>2666</td>
      <td>17.033072018004518</td>
      <td>4.330864176799864</td>
      <td>0.0</td>
      <td>30.91</td>
    </tr>
    <tr>
      <th>Total night minutes</th>
      <td>2666</td>
      <td>201.16894223555968</td>
      <td>50.780323368725206</td>
      <td>43.7</td>
      <td>395.0</td>
    </tr>
    <tr>
      <th>Total night calls</th>
      <td>2666</td>
      <td>100.10615153788447</td>
      <td>19.418458551101697</td>
      <td>33</td>
      <td>166</td>
    </tr>
    <tr>
      <th>Total night charge</th>
      <td>2666</td>
      <td>9.052689422355604</td>
      <td>2.2851195129157564</td>
      <td>1.97</td>
      <td>17.77</td>
    </tr>
    <tr>
      <th>Total intl minutes</th>
      <td>2666</td>
      <td>10.23702175543886</td>
      <td>2.7883485770512566</td>
      <td>0.0</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>Total intl calls</th>
      <td>2666</td>
      <td>4.467366841710428</td>
      <td>2.4561949030129466</td>
      <td>0</td>
      <td>20</td>
    </tr>
    <tr>
      <th>Total intl charge</th>
      <td>2666</td>
      <td>2.764489872468112</td>
      <td>0.7528120531228477</td>
      <td>0.0</td>
      <td>5.4</td>
    </tr>
    <tr>
      <th>Customer service calls</th>
      <td>2666</td>
      <td>1.5626406601650413</td>
      <td>1.3112357589949093</td>
      <td>0</td>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>



We now plot a bit! Please refer to the [original post](https://mapr.com/blog/churn-prediction-pyspark-using-mllib-and-ml-packages/) for explanation. It is OK if this piece of code takes a while to run!


```python
import matplotlib.pyplot as plt
```


```python
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
```


![png](output_16_0.png)


So pretty! But we want to skip the obviously correlated features, and also change the "yes", "no", "true" and "false" to ones and zeros. We'll do that by droping some columns and changing some others.


```python
from pyspark.sql.types import DoubleType
```


```python
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
                                   .cast("boolean").cast(DoubleType()))
```

And let's not forget the test data.


```python
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
                                   .cast("boolean").cast(DoubleType()))
```

This looks like this now:


```python
pd.DataFrame(train_data.take(5), columns=train_data.columns).transpose()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Account length</th>
      <td>128.0</td>
      <td>107.0</td>
      <td>137.0</td>
      <td>84.0</td>
      <td>75.0</td>
    </tr>
    <tr>
      <th>International plan</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Voice mail plan</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Number vmail messages</th>
      <td>25.0</td>
      <td>26.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>Total day minutes</th>
      <td>265.1</td>
      <td>161.6</td>
      <td>243.4</td>
      <td>299.4</td>
      <td>166.7</td>
    </tr>
    <tr>
      <th>Total day calls</th>
      <td>110.0</td>
      <td>123.0</td>
      <td>114.0</td>
      <td>71.0</td>
      <td>113.0</td>
    </tr>
    <tr>
      <th>Total eve minutes</th>
      <td>197.4</td>
      <td>195.5</td>
      <td>121.2</td>
      <td>61.9</td>
      <td>148.3</td>
    </tr>
    <tr>
      <th>Total eve calls</th>
      <td>99.0</td>
      <td>103.0</td>
      <td>110.0</td>
      <td>88.0</td>
      <td>122.0</td>
    </tr>
    <tr>
      <th>Total night minutes</th>
      <td>244.7</td>
      <td>254.4</td>
      <td>162.6</td>
      <td>196.9</td>
      <td>186.9</td>
    </tr>
    <tr>
      <th>Total night calls</th>
      <td>91.0</td>
      <td>103.0</td>
      <td>104.0</td>
      <td>89.0</td>
      <td>121.0</td>
    </tr>
    <tr>
      <th>Total intl minutes</th>
      <td>10.0</td>
      <td>13.7</td>
      <td>12.2</td>
      <td>6.6</td>
      <td>10.1</td>
    </tr>
    <tr>
      <th>Total intl calls</th>
      <td>3.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>7.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>Customer service calls</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>Churn</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



Please note that we are only explorating, getting summary statistics, plotting,... the training set.

### Machine Learning

Let's do some **machine learning** now. We will use the ML library from Spark (not the MLlib from spark).


```python
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
```

From the original post: "The ML package needs data be put in a (label: Double, features: Vector) DataFrame format with correspondingly named fields. The vectorizeData() function below performs this formatting." In this case we'll be using the VectorAssembler function from pyspark.ml.feature.


```python
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
```


```python
df_train = vecAssembler.transform(train_data)
```

Note the "features" column that we just added, which includes all other features in one single array. We execute the former code just to see what vecAssembler will be doing; the actual work will be carried out with a pipeline wrapper.


```python
pd.DataFrame(df_train.take(5), columns=df_train.columns).transpose()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Account length</th>
      <td>128</td>
      <td>107</td>
      <td>137</td>
      <td>84</td>
      <td>75</td>
    </tr>
    <tr>
      <th>International plan</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Voice mail plan</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Number vmail messages</th>
      <td>25</td>
      <td>26</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>Total day minutes</th>
      <td>265.1</td>
      <td>161.6</td>
      <td>243.4</td>
      <td>299.4</td>
      <td>166.7</td>
    </tr>
    <tr>
      <th>Total day calls</th>
      <td>110</td>
      <td>123</td>
      <td>114</td>
      <td>71</td>
      <td>113</td>
    </tr>
    <tr>
      <th>Total eve minutes</th>
      <td>197.4</td>
      <td>195.5</td>
      <td>121.2</td>
      <td>61.9</td>
      <td>148.3</td>
    </tr>
    <tr>
      <th>Total eve calls</th>
      <td>99</td>
      <td>103</td>
      <td>110</td>
      <td>88</td>
      <td>122</td>
    </tr>
    <tr>
      <th>Total night minutes</th>
      <td>244.7</td>
      <td>254.4</td>
      <td>162.6</td>
      <td>196.9</td>
      <td>186.9</td>
    </tr>
    <tr>
      <th>Total night calls</th>
      <td>91</td>
      <td>103</td>
      <td>104</td>
      <td>89</td>
      <td>121</td>
    </tr>
    <tr>
      <th>Total intl minutes</th>
      <td>10</td>
      <td>13.7</td>
      <td>12.2</td>
      <td>6.6</td>
      <td>10.1</td>
    </tr>
    <tr>
      <th>Total intl calls</th>
      <td>3</td>
      <td>3</td>
      <td>5</td>
      <td>7</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Customer service calls</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Churn</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>features</th>
      <td>[128.0, 0.0, 1.0, 25.0, 265.1, 110.0, 197.4, 9...</td>
      <td>[107.0, 0.0, 1.0, 26.0, 161.6, 123.0, 195.5, 1...</td>
      <td>[137.0, 0.0, 0.0, 0.0, 243.4, 114.0, 121.2, 11...</td>
      <td>[84.0, 1.0, 0.0, 0.0, 299.4, 71.0, 61.9, 88.0,...</td>
      <td>[75.0, 1.0, 0.0, 0.0, 166.7, 113.0, 148.3, 122...</td>
    </tr>
  </tbody>
</table>
</div>



### Training the model

First, we create the decision tree with the columns "Churn" as labels and "features" as features.


```python
dt = DecisionTreeClassifier(labelCol="Churn", featuresCol="features")
```

This Pipeline chains vecAssembler (VectorAssembler) and dt (DecisionTreeClassifier).


```python
pipeline = Pipeline(stages=[vecAssembler, dt])
```

We fit the data.


```python
model = pipeline.fit(train_data)
```

### Making predictions


```python
predictions = model.transform(test_data)
```


```python
predictions.select("prediction", "Churn", "features").toPandas().head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>prediction</th>
      <th>Churn</th>
      <th>features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>[117.0, 0.0, 0.0, 0.0, 184.5, 97.0, 351.6, 80....</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>[65.0, 0.0, 0.0, 0.0, 129.1, 137.0, 228.5, 83....</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>[161.0, 0.0, 0.0, 0.0, 332.9, 67.0, 317.8, 97....</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>[111.0, 0.0, 0.0, 0.0, 110.4, 103.0, 137.3, 10...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>[49.0, 0.0, 0.0, 0.0, 119.3, 117.0, 215.1, 109...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>[36.0, 0.0, 1.0, 30.0, 146.3, 128.0, 162.5, 80...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>[65.0, 0.0, 0.0, 0.0, 211.3, 120.0, 162.6, 122...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>[119.0, 0.0, 0.0, 0.0, 159.1, 114.0, 231.3, 11...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>[10.0, 0.0, 0.0, 0.0, 186.1, 112.0, 190.2, 66....</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>[68.0, 0.0, 0.0, 0.0, 148.8, 70.0, 246.5, 164....</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>[74.0, 0.0, 1.0, 33.0, 193.7, 91.0, 246.1, 96....</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>[85.0, 0.0, 0.0, 0.0, 235.8, 109.0, 157.2, 94....</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>[46.0, 0.0, 0.0, 0.0, 214.1, 72.0, 164.4, 104....</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>[128.0, 0.0, 1.0, 29.0, 179.3, 104.0, 225.9, 8...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>[155.0, 0.0, 0.0, 0.0, 203.4, 100.0, 190.9, 10...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>[73.0, 0.0, 0.0, 0.0, 160.1, 110.0, 213.3, 72....</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>[77.0, 0.0, 0.0, 0.0, 251.8, 72.0, 205.7, 126....</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>[108.0, 0.0, 0.0, 0.0, 178.3, 137.0, 189.0, 76...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>[95.0, 0.0, 0.0, 0.0, 135.0, 99.0, 183.6, 106....</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>[36.0, 0.0, 1.0, 29.0, 281.4, 102.0, 202.2, 76...</td>
    </tr>
  </tbody>
</table>
</div>



We use Pandas only because I like the styling better.

### Evaluating the model

Unlike in the original post, we will be evaluating the model with a BinaryClassifierEvaluator rather than with a MulticlassClassifierEvaluator.


```python
evaluator = BinaryClassificationEvaluator(
    labelCol="Churn", rawPredictionCol="prediction")
```


```python
evaluator.evaluate(predictions)
```




    0.8184302539565699



By default BinaryClassificationEvaluator gives the ROC Area Under the Curve.

### Improving the model

We can try to improve our model with a bit of parameter tuning. This will take a while.


```python
paramGrid = ParamGridBuilder().addGrid(dt.maxDepth, [2,3,4,5,6,7]).build()

# Set up 3-fold cross validation
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator, 
                          numFolds=3)

CV_model = crossval.fit(train_data)
```

We fetch the best model and print it:


```python
tree_model = CV_model.bestModel.stages[1]
print(tree_model)
```

    DecisionTreeClassificationModel (uid=DecisionTreeClassifier_422596b245d603704296) of depth 7 with 121 nodes



```python
predictions_improved = CV_model.bestModel.transform(test_data)
```


```python
predictions_improved.select("prediction", "Churn", "features").toPandas().head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>prediction</th>
      <th>Churn</th>
      <th>features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>[117.0, 0.0, 0.0, 0.0, 184.5, 97.0, 351.6, 80....</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>[65.0, 0.0, 0.0, 0.0, 129.1, 137.0, 228.5, 83....</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>[161.0, 0.0, 0.0, 0.0, 332.9, 67.0, 317.8, 97....</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>[111.0, 0.0, 0.0, 0.0, 110.4, 103.0, 137.3, 10...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>[49.0, 0.0, 0.0, 0.0, 119.3, 117.0, 215.1, 109...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>[36.0, 0.0, 1.0, 30.0, 146.3, 128.0, 162.5, 80...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>[65.0, 0.0, 0.0, 0.0, 211.3, 120.0, 162.6, 122...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>[119.0, 0.0, 0.0, 0.0, 159.1, 114.0, 231.3, 11...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>[10.0, 0.0, 0.0, 0.0, 186.1, 112.0, 190.2, 66....</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>[68.0, 0.0, 0.0, 0.0, 148.8, 70.0, 246.5, 164....</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>[74.0, 0.0, 1.0, 33.0, 193.7, 91.0, 246.1, 96....</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>[85.0, 0.0, 0.0, 0.0, 235.8, 109.0, 157.2, 94....</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>[46.0, 0.0, 0.0, 0.0, 214.1, 72.0, 164.4, 104....</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>[128.0, 0.0, 1.0, 29.0, 179.3, 104.0, 225.9, 8...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>[155.0, 0.0, 0.0, 0.0, 203.4, 100.0, 190.9, 10...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>[73.0, 0.0, 0.0, 0.0, 160.1, 110.0, 213.3, 72....</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>[77.0, 0.0, 0.0, 0.0, 251.8, 72.0, 205.7, 126....</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>[108.0, 0.0, 0.0, 0.0, 178.3, 137.0, 189.0, 76...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>[95.0, 0.0, 0.0, 0.0, 135.0, 99.0, 183.6, 106....</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>[36.0, 0.0, 1.0, 29.0, 281.4, 102.0, 202.2, 76...</td>
    </tr>
  </tbody>
</table>
</div>




```python
evaluator.evaluate(predictions_improved)
```




    0.8526683842473316



We did indeed improve our AUC ROC!

##### March 2018, Josep Anton Mir Tutusaus
