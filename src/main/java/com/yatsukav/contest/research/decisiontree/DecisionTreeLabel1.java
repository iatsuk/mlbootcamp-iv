package com.yatsukav.contest.research.decisiontree;

import com.yatsukav.contest.data.InitialData;
import com.yatsukav.contest.research.ResearchUtil;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.feature.PCA;
import org.apache.spark.ml.feature.PCAModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.functions;

import java.util.Arrays;

public interface DecisionTreeLabel1 {

    /**
     * Current top accuracy: 0.6408934707903781; negative result: 0.2340764331210191
     * PCA_K: 140
     */
    static void run(int pcaK) {
        // Get train dataset
        Dataset<Row> train = InitialData.getTrainDF();
        train = train.withColumn("label", functions.expr("cast(if(label=1, 0, 1) as double)"));

        // Features columns names
        String[] features = Arrays.stream(train.columns())
                .filter(c -> !c.equals("label"))
                .toArray(String[]::new);

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

        long countLabel = train.select("label").filter("label=0").count();
        long countPositivePredicted = train.select("label", "prediction").filter("label=0 and prediction=0").count();
        long countForNegativePredicted = train.select("label", "prediction").filter("label=1 and prediction=1").count();
        long countNegativePredicted = train.select("label", "prediction").filter("label=1 and prediction=0").count();

        double accuracy = (double) countPositivePredicted / countLabel;
        System.out.println("pca: " + pcaK + "; accuracy: " + accuracy + "; negative result: " + ((double) countNegativePredicted / countForNegativePredicted));

        ResearchUtil.printPercentageConfusionMatrix(train, "label", "prediction");
    }
}
