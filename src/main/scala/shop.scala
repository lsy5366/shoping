import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.SparkSession
object shop {
  case class Rating(userId: Int, movieId: Int, rating: Float, timestamp: Long)
  def main(args: Array[String]) {
    def parseRating(str: String): Rating = {
             val fields = str.split("::")
             assert(fields.size == 4)
             Rating(fields(0).toInt, fields(1).toInt, fields(2).toFloat, fields(3).toLong)
          }

    val spark = SparkSession.builder.getOrCreate
    import spark.implicits._
    val ratings = spark.sparkContext.textFile("file:///usr/local/spark/data/mllib/als/sample_movielens_ratings.txt").map(parseRating).toDF()
    ratings.show()
    val Array(training, test) = ratings.randomSplit(Array(0.8, 0.2))
    val alsExplicit = new ALS().setMaxIter(5).setRegParam(0.01).setUserCol("userId"). setItemCol("movieId").setRatingCol("rating")
    val alsImplicit = new ALS().setMaxIter(5).setRegParam(0.01).setImplicitPrefs(true). setUserCol("userId").setItemCol("movieId").setRatingCol("rating")
    val modelExplicit = alsExplicit.fit(training)
    val modelImplicit = alsImplicit.fit(training)
    val predictionsExplicit = modelExplicit.transform(test).show()
    val predictionsImplicit = modelImplicit.transform(test).show()
  }
}