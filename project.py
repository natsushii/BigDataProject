import argparse
from pyspark.sql import SparkSession
from spark_utils import SparkUtils


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--rows", type=int, default=20)
    args = p.parse_args()

    spark = (SparkSession.builder.appName("project")
             .config("spark.sql.parquet.enableVectorizedReader", "false")
             .getOrCreate())
    su = SparkUtils(spark)

    df = su.read_input(args.input)
    df = su.clean(df)
    df = su.add_columns(df)
    df = su.join_lookup(df, su.source_lookup())
    result = su.filter_sort(su.aggregate(df))

    # Persistencia (Seccion 3)
    result.write.mode("overwrite").partitionBy("region").parquet(args.output)

    spark.stop()


if __name__ == "__main__":
    main()
