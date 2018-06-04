from pyspark import SparkContext

sc = SparkContext("local", "Collect app")  # Master i appName

words = sc.parallelize(
    ["scala",
     "java",
     "hadoop",
     "spark",
     "akka",
     "spark vs hadoop",
     "pyspark",
     "pyspark and spark"])

counts = words.count()
coll = words.collect()

print("Number of elements in RDD -> ", counts)
print("Elements in RDD -> ", coll)
