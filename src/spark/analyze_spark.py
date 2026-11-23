from pyspark.sql import SparkSession, Window, functions as F

spark = (SparkSession.builder
         .appName("lmia-analysis")
         .master("local[*]")
         .config("spark.driver.host", "127.0.0.1")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

df = spark.read.parquet("data/lmia_all_labeled.parquet")
approved = df.filter(F.col("Decision") == "Approved")

# 1. Approved positions per quarter + quarter-over-quarter change (lag window)
w_time = Window.orderBy("Quarter")
quarterly = (approved.groupBy("Quarter")
             .agg(F.sum("Positions").alias("approved_positions"))
             .withColumn("prev", F.lag("approved_positions").over(w_time))
             .withColumn("qoq_pct",
                         F.round((F.col("approved_positions") - F.col("prev"))
                                 / F.col("prev") * 100, 1))
             .drop("prev")
             .orderBy("Quarter"))
print("=== Approved positions by quarter ===")
quarterly.show(truncate=False)

first = quarterly.orderBy("Quarter").first()["approved_positions"]
last = quarterly.orderBy(F.desc("Quarter")).first()["approved_positions"]
print(f"Change, first to last quarter: {(last - first) / first * 100:.1f}%")

# 2. Top 5 occupations per quarter (row_number window partitioned by quarter)
occ = (approved.groupBy("Quarter", "Occupation_Title")
       .agg(F.sum("Positions").alias("positions")))
w_rank = Window.partitionBy("Quarter").orderBy(F.desc("positions"))
top5 = (occ.withColumn("rank", F.row_number().over(w_rank))
        .filter(F.col("rank") <= 5)
        .orderBy("Quarter", "rank"))
print("=== Top 5 occupations per quarter ===")
top5.show(60, truncate=False)

# 3. Denial rate by quarter (share of LMIAs denied)
denial = (df.groupBy("Quarter")
          .pivot("Decision", ["Approved", "Denied"])
          .agg(F.sum("LMIAs"))
          .withColumn("denial_rate_pct",
                      F.round(F.col("Denied")
                              / (F.col("Approved") + F.col("Denied")) * 100, 1))
          .orderBy("Quarter"))
print("=== Denial rate by quarter ===")
denial.show(truncate=False)

spark.stop()
