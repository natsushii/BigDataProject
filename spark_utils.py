from pyspark.sql import functions as F


class SparkUtils:
    def __init__(self, spark):
        self.spark = spark

    def read_input(self, path):
        # Leer solo las columnas necesarias (column pruning en Parquet)
        df = self.spark.read.parquet(path)
        return df.select(
            "id",
            F.col("sources.dataset").alias("datasets"),
            "height", "num_floors", "bbox",
        )

    # 1. Limpieza: nulos y valores fuera de rango
    def clean(self, df):
        return (df.filter(F.col("id").isNotNull() & F.col("bbox").isNotNull())
                  .filter(F.col("height").isNull() | F.col("height").between(0.5, 1000))
                  .filter(F.col("num_floors").isNull() | F.col("num_floors").between(1, 200)))

    # 2. Columnas derivadas
    def add_columns(self, df):
        lon = (F.col("bbox.xmin") + F.col("bbox.xmax")) / 2
        lat = (F.col("bbox.ymin") + F.col("bbox.ymax")) / 2
        region = (F.when(lon.between(-170, -30) & (lat >= 12), "North America")
                   .when(lon.between(-95, -30) & (lat < 12), "South America")
                   .when(lon.between(-30, 60) & (lat >= 35), "Europe")
                   .when(lon.between(-30, 55) & (lat < 35), "Africa")
                   .when(lon.between(60, 180) & (lat >= 0), "Asia")
                   .when(lon.between(60, 180) & (lat < 0), "Oceania")
                   .otherwise("Other"))
        return (df
            .withColumn("source_dataset",
                        F.coalesce(F.try_element_at("datasets", F.lit(1)), F.lit("unknown")))
            .withColumn("lon", lon)
            .withColumn("lat", lat)
            .withColumn("region", region)
            .withColumn("has_height", F.col("height").isNotNull().cast("int"))
            .withColumn("has_floors", F.col("num_floors").isNotNull().cast("int"))
            .drop("datasets"))

    # 3. Join con catalogo pequeno
    def source_lookup(self):
        rows = [("OpenStreetMap", "Crowdsourced"),
                ("Microsoft ML Buildings", "Machine learning"),
                ("Google Open Buildings", "Machine learning"),
                ("Esri Community Maps", "Community/Government")]
        return self.spark.createDataFrame(rows, ["source_dataset", "provider_type"])

    def join_lookup(self, df, lookup):
        return (df.join(F.broadcast(lookup), on="source_dataset", how="left")
                  .fillna({"provider_type": "Other"}))

    # 4. Agregacion
    def aggregate(self, df):
        return (df.groupBy("region", "source_dataset", "provider_type")
                  .agg(F.count("*").alias("total_buildings"),
                       F.round(F.avg("has_height") * 100, 2).alias("pct_with_height"),
                       F.round(F.avg("has_floors") * 100, 2).alias("pct_with_floors"),
                       F.round(F.avg("height"), 2).alias("avg_height_m"),
                       F.max("height").alias("max_height_m")))

    # 5. Filtrado y orden
    def filter_sort(self, df, min_total=1000):
        return (df.filter(F.col("total_buildings") >= min_total)
                  .orderBy(F.desc("total_buildings")))
