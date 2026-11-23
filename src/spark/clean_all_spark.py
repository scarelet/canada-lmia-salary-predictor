import re
from pyspark.sql import SparkSession, functions as F

spark = (SparkSession.builder
         .appName("lmia-clean")
         .master("local[*]")
         .getOrCreate())


def snake(name):
    # Parquet rejects spaces and some symbols in column names
    return re.sub(r"[^0-9a-zA-Z]+", "_", name.strip()).strip("_")


def load_and_label(pattern, decision, count_prefix):
    df = (spark.read
          .option("header", True)
          .option("multiLine", True)
          .option("escape", '"')
          .csv(pattern)
          .withColumn("source_file", F.input_file_name()))
    for c in df.columns:
        df = df.withColumnRenamed(c, snake(c))
    # positive files say "Approved", negative say "Requested"
    df = (df.withColumnRenamed(f"{count_prefix}_LMIAs", "LMIAs")
            .withColumnRenamed(f"{count_prefix}_Positions", "Positions")
            .withColumn("Decision", F.lit(decision)))
    return df


positive = load_and_label("data/raw_csv/tfwp_*_pos_en.csv", "Approved", "Approved")
negative = load_and_label("data/raw_csv/tfwp_*_neg_en.csv", "Denied", "Requested")

print("Positive rows:", positive.count())
print("Negative rows:", negative.count())

df = positive.unionByName(negative, allowMissingColumns=True)

# Drop footer/note rows (no Employer value)
df = df.filter(F.col("Employer").isNotNull())

# NOC code + clean occupation title
df = (df
      .withColumn("NOC_Code", F.regexp_extract("Occupation", r"^(\d{4,5})-", 1))
      .withColumn("Occupation_Title", F.regexp_replace("Occupation", r"^\d{4,5}-", ""))
      .filter(F.col("NOC_Code") != ""))

# Quarter from the file name
df = df.withColumn(
    "Quarter",
    F.regexp_extract("source_file", r"tfwp_(\d{4}q\d)_(?:pos|neg)_en", 1))

# Trim whitespace
for c in ["Employer", "Address", "Occupation_Title", "Program_Stream", "Province_Territory"]:
    df = df.withColumn(c, F.trim(F.col(c)))

# Numeric casts (bad values become null, like pd.to_numeric errors="coerce")
df = (df.withColumn("LMIAs", F.col("LMIAs").cast("double"))
        .withColumn("Positions", F.col("Positions").cast("double")))

total = df.count()
print("Total labeled rows:", total)

(df.groupBy("Decision").count()
   .withColumn("pct", F.round(F.col("count") / total * 100, 2))
   .show())

(df.write.mode("overwrite")
   .partitionBy("Quarter")
   .parquet("data/lmia_all_labeled.parquet"))
print("Saved to data/lmia_all_labeled.parquet")

spark.stop()