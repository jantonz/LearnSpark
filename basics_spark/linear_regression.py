
# coding: utf-8

# In[ ]:


# TUTORIAL from https://www.datacamp.com/community/tutorials/apache-spark-tutorial-machine-learning#basics


# In[29]:


from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql import Row
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.linalg import DenseVector
from pyspark.ml.feature import StandardScaler
from pyspark.ml.regression import LinearRegression


# In[2]:


# Create the spark context
sc = SparkContext("local", "Linear Regression Model")
spark = SparkSession(sc)


# In[3]:


# Load the data
rdd = sc.textFile("/home/jmir/ai/learning/spark/cal_housing.data")

# Load the header
header = sc.textFile("/home/jmir/ai/learning/spark/cal_housing.domain")


# In[4]:


header.collect()


# In[5]:


rdd.take(5)


# In[6]:


# Split lines on commas (because it is a .csv)
rdd = rdd.map(lambda line: line.split(","))


# In[7]:


# Inspect the same first 5 lines
rdd.take(5)  # or rdd.first() or rdd.top(5)


# In[8]:


# Let's construct a SchemaRDD to convert the RDD to a DataFrame


# In[9]:


df = rdd.map(lambda line: Row(longitude=line[0],
                              latitude=line[1],
                              housingMedianAge=line[2],
                              totalRooms=line[3],
                              totalBedRooms=line[4],
                              population=line[5],
                              households=line[6],
                              medianIncome=line[7],
                              medianHouseValue=line[8])).toDF()  # .toDF() requires a SparkSession (or SQLContext)!


# In[10]:


# Let's inspect the df first.
# df.first()  or
# df.head(20)  or
# df.take(20)  or
df.show(20)  # this one preserves the crow/column format


# In[11]:


df.columns  # gives you the columns of the DataFrame


# In[12]:


df.dtypes


# In[13]:


df.printSchema()


# In[14]:


# We should assign the proper data types to the columns (we don't really want them to be strings!)
df = df.withColumn("longitude", df["longitude"].cast(FloatType()))    .withColumn("latitude", df["latitude"].cast(FloatType()))    .withColumn("housingMedianAge",df["housingMedianAge"].cast(FloatType()))    .withColumn("totalRooms", df["totalRooms"].cast(FloatType()))    .withColumn("totalBedRooms", df["totalBedRooms"].cast(FloatType()))    .withColumn("population", df["population"].cast(FloatType()))    .withColumn("households", df["households"].cast(FloatType()))    .withColumn("medianIncome", df["medianIncome"].cast(FloatType()))    .withColumn("medianHouseValue", df["medianHouseValue"].cast(FloatType()))


# In[15]:


# But this method, using a User-Defined Function (UDF) looks prettier
def convertColumns(df, names, newType):
    for name in names:
        df = df.withColumn(name, df[name].cast(newType))
    return df


# In[16]:


# Assign column names to columns:
columns = df.columns

# Use the function!
df = convertColumns(df, columns, FloatType())


# In[17]:


df.printSchema()


# In[18]:


# All good now!
# Let's launch sql-like queries now
df.select("population" ,"totalBedrooms").show(10)


# In[19]:


df.groupBy("housingMedianAge").count().sort("housingMedianAge", ascending=False).show()


# In[20]:


df.describe().show()


# In[21]:


# Adjust the values of "medianHouseValue"
df = df.withColumn("medianHouseValue", col("medianHouseValue")/100000)
df.show(5)


# In[22]:


df = df.withColumn("roomsPerHousehold", col("totalRooms")/col("households"))        .withColumn("populationPerHousehold", col("population")/col("households"))        .withColumn("bedroomsPerRoom", col("totalBedRooms")/col("totalRooms"))


# In[23]:


df.first()


# In[24]:


df = df.select("medianHouseValue",
            "totalBedRooms",
            "population",
            "medianIncome",
            "roomsPerHousehold",
            "populationPerHousehold",
            "bedroomsPerRoom")


# In[25]:


# Define the `input_data`
input_data = df.rdd.map(lambda x: (x[0], DenseVector(x[1:])))

# Replace `df` with the new DataFrame
df = spark.createDataFrame(input_data, ["label", "features"])


# In[26]:


df.show(5)


# In[27]:


# Initialize the `standardScaler`
standardScaler = StandardScaler(inputCol="features", outputCol="features_scaled")

# Fit the DataFrame to the scaler
scaler = standardScaler.fit(df)

# Transform the data in "df" with the scaler
scaled_df = scaler.transform(df)

# Inspect the result
scaled_df.show(5)


# In[55]:


# Split the data into train and test sets
train, test = scaled_df.randomSplit([.8, .2], seed=1234)


# In[66]:


# Note that the argument elasticNetParam corresponds to α or the vertical
# intercept and that the regParam or the regularization paramater corresponds to λ.

# Initialize `lr`
lr = LinearRegression(labelCol="label", maxIter=100, regParam=0.3, elasticNetParam=0.8)
# Fit the data to the model
linearModel = lr.fit(train)


# In[67]:


# Generate predictions
pred = linearModel.transform(test)

# Extract predictions and labels
predictions = pred.select("prediction").rdd.map(lambda x: x[0])
labels = pred.select("label").rdd.map(lambda x: x[0])


# In[68]:


# Zip `predictions` and `labels` into a list
predictionsAndLabel = predictions.zip(labels).collect()

predictionsAndLabel[:5]


# In[69]:


# Coefficients for the model
linearModel.coefficients


# In[70]:


# Intercept for the model
linearModel.intercept


# In[71]:


# RMSE
linearModel.summary.rootMeanSquaredError


# In[72]:


# R²
linearModel.summary.r2


# In[73]:


spark.stop()
