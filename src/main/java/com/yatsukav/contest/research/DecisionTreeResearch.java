package com.yatsukav.contest.research;

import com.yatsukav.contest.data.InitialData;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.PCA;
import org.apache.spark.ml.feature.PCAModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SaveMode;

import java.util.Arrays;

public interface DecisionTreeResearch {

    /**
     * Current top accuracy: 0.5697907709945543
     * PCA_K: 79
     */
    static void run(int pcaK, boolean printTestDataResult) {
        // Get train dataset
        Dataset<Row> train = InitialData.getTrainDF();

        // Features columns names
        String[] features = Arrays.stream(train.columns())
                .filter(c -> !c.equals("label"))
                .toArray(String[]::new);
//            System.out.println("Features count: " + features.length);

        // All columns with features to single multidimensional vector
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(features)
                .setOutputCol("features");
        train = assembler.transform(train)
                .select("label", "features");

        // Project vectors to a low-dimensional space using PCA
        PCAModel pca = new PCA()
                .setInputCol("features")
                .setOutputCol("pcaFeatures")
                .setK(pcaK) // todo important parameter
                .fit(train);
        train = pca.transform(train).drop("features");
        train = train.withColumnRenamed("pcaFeatures", "features");

        // Decision tree classifier
        DecisionTreeClassifier dt = new DecisionTreeClassifier()
                .setLabelCol("label")
                .setFeaturesCol("features");

        DecisionTreeClassificationModel model = dt.fit(train)
                .setPredictionCol("prediction")
                .setProbabilityCol("probability")
                .setRawPredictionCol("raw_prediction");
        train = model.transform(train).drop(features);

//            System.out.println(model.toDebugString());
//            train.show();

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");
        double accuracy = evaluator.evaluate(train);
        System.out.println("pca: " + pcaK + "; accuracy: " + accuracy);

        double currentTop = 0.5697907709945543;
        System.out.println("Accuracy is better of current top: " + (accuracy > currentTop));
        System.out.println("Accuracy is equal of current top: " + (accuracy == currentTop));
        System.out.println("Accuracy is worse of current top: " + (accuracy < currentTop));

        System.out.println();
        ResearchUtil.printPercentageConfusionMatrix(train, "label", "prediction");

        if (printTestDataResult) {
            //// TEST DATA PREDICTION
            Dataset<Row> test = InitialData.getTestDF();
            test = assembler.transform(test);
            test = pca.transform(test).drop("features").withColumnRenamed("pcaFeatures", "features");
            test = model.transform(test);
            test.select("prediction").coalesce(1).write().mode(SaveMode.Overwrite).option("separator", ";").csv("dt.csv");
        }
    }
}
