from pyspark import SparkContext

sc = SparkContext("local", "Cache app")

words = sc.parallelize(
    ["scala",
     "java",
     "hadoop",
     "spark",
     "akka",
     "spark vs hadoop",
     "pyspark",
     "pyspark and spark"])

words.cache()

caching = words.persist().is_cached

print("Words got cached -> ", caching)
