import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.feature._
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.types.StructType
object shoping {
  case class Rating(CustomerID: Int, StockCodeIndex: Int, rating: Float)
  def main(args: Array[String]) {
    val spark = SparkSession.builder.appName("shoping").master("local")getOrCreate
    import spark.implicits._
    val data1 = spark.read.format("csv").option("header", "true").option("inferSchema", true).load(("hdfs://172.16.24.100:9000/user/hadoop/test.csv"))
    var data=data1.sqlContext.createDataFrame(data1.rdd, StructType(data1.schema.map(_.copy(nullable = false))))
    data= new StringIndexer().setInputCol("StockCode").setOutputCol("StockCodeIndex").fit(data).transform(data)
//    data=new StringIndexer().setInputCol("CustomerID").setOutputCol("CustomerIDIndex").fit(data).transform(data)

    val assembler = new VectorAssembler()
      .setInputCols(Array("StockCodeIndex", "CustomerID"))
      .setOutputCol("features")
    val output = assembler.transform(data).dropDuplicates("features")

    val text = output.groupBy("features").sum("Quantity").dropDuplicates("features")
    //    val lsy=text.select("InvoiceNo","features")
    val df = output.join(text, Seq("features"))
    //        df.show()
    val dataframe = df.select("StockCodeIndex", "CustomerID", "sum(Quantity)")

    var dataframe1=dataframe.select(dataframe("StockCodeIndex").cast("Int"),dataframe("CustomerID").cast("Int"),dataframe("sum(Quantity)").cast("Float"))

    val daf=dataframe1.rdd.map{r=>Rating((r(0).toString.toInt),r(1).toString.toInt,r(2).toString.toFloat)}.filter(x => !x.rating.isNaN && !x.rating.isInfinity).toDF()

    val Array(training, test) = daf.randomSplit(Array(0.8, 0.2))

    val alsExplicit = new ALS().setMaxIter(5).setRegParam(0.01).setUserCol("CustomerID"). setItemCol("StockCodeIndex").setRatingCol("rating")

    val modelExplicit = alsExplicit.fit(training)

    val predictionsExplicit = modelExplicit.transform(test)
    predictionsExplicit.show()
    val evaluator = new RegressionEvaluator().setMetricName("rmse").setLabelCol("rating"). setPredictionCol("prediction")
    val rmseExplicit = evaluator.evaluate(predictionsExplicit)
    println(s"Explicit:Root-mean-square error = $rmseExplicit")
//    var converter = new IndexToString().setInputCol("InvoiceNoIndex").setOutputCol("InvoiceNo").transform(predictionsExplicit)
//    converter = new IndexToString().setInputCol("CustomerIDIndex").setOutputCol("CustomerID").transform(predictionsExplicit)
//    converter.show()



  }
}
