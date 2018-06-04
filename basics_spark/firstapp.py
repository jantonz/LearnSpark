from pyspark import SparkContext

logFile = "/usr/local/spark/README.md"

sc = SparkContext("local", "First app")  # Master i appName

logData = sc.textFile(logFile).cache()

numAs = logData.filter(lambda s: "a" in s).count()
numBs = logData.filter(lambda s: "b" in s).count()

print("Lines with a: ", numAs, " lines with b: ", numBs)
