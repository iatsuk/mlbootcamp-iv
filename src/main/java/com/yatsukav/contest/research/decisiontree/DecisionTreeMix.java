package com.yatsukav.contest.research.decisiontree;

import com.yatsukav.contest.data.InitialData;
import com.yatsukav.contest.research.ResearchUtil;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.PCA;
import org.apache.spark.ml.feature.PCAModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.functions;

import java.util.Arrays;

public interface DecisionTreeMix {

    /**
     * accuracy: 0.591000286615076
     * 0	1	2	3	4
     * 0:	43	25	29	1	0
     * 1:	0	59	36	3	0
     * 2:	0	20	68	10	0
     * 3:	0	8	48	42	0
     * 4:	0	3	33	28	34
     * Total count of 0.0 is: 113
     * Total count of 1.0 is: 1164
     * Total count of 2.0 is: 1521
     * Total count of 3.0 is: 581
     * Total count of 4.0 is: 110
     * <p>
     * Test Data Accuracy: 0,5017668
     */
    static void run(boolean printTestDataResult) {
        // Get train dataset
        Dataset<Row> train = InitialData.getTrainDF();
        train = train.withColumn("label4", functions.expr("cast(if(label=4, 0, 1) as double)"));
        train = train.withColumn("label0", functions.expr("cast(if(label=0, 0, 1) as double)"));
        train = train.withColumn("label3", functions.expr("cast(if(label=3, 0, 1) as double)"));

        // Features columns names
        String[] features = Arrays.stream(train.columns())
                .filter(c -> !c.contains("label"))
                .toArray(String[]::new);

        // All columns with features to single multidimensional vector
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(features)
                .setOutputCol("features");
        train = assembler.transform(train)
                .select("label", "label0", "label3", "label4", "features");

        // Project vectors to a low-dimensional space using PCA
        PCAModel pca = new PCA().setInputCol("features").setOutputCol("pcaFeatures").setK(79).fit(train);
        PCAModel pca4 = new PCA().setInputCol("features").setOutputCol("pcaFeatures4").setK(165).fit(train);
        PCAModel pca0 = new PCA().setInputCol("features").setOutputCol("pcaFeatures0").setK(100).fit(train);
        PCAModel pca3 = new PCA().setInputCol("features").setOutputCol("pcaFeatures3").setK(140).fit(train);

        train = pca.transform(train);
        train = pca4.transform(train);
        train = pca0.transform(train);
        train = pca3.transform(train);
        train = train.drop("features");

        // Decision tree classifier
        DecisionTreeClassifier dt = new DecisionTreeClassifier().setLabelCol("label").setFeaturesCol("pcaFeatures");
        DecisionTreeClassifier dt4 = new DecisionTreeClassifier().setLabelCol("label4").setFeaturesCol("pcaFeatures4");
        DecisionTreeClassifier dt0 = new DecisionTreeClassifier().setLabelCol("label0").setFeaturesCol("pcaFeatures0");
        DecisionTreeClassifier dt3 = new DecisionTreeClassifier().setLabelCol("label3").setFeaturesCol("pcaFeatures3");

        DecisionTreeClassificationModel model = dt.fit(train).setPredictionCol("prediction").setProbabilityCol("pc").setRawPredictionCol("rpc");
        DecisionTreeClassificationModel model4 = dt4.fit(train).setPredictionCol("prediction4").setProbabilityCol("pc4").setRawPredictionCol("rpc4");
        DecisionTreeClassificationModel model0 = dt0.fit(train).setPredictionCol("prediction0").setProbabilityCol("pc0").setRawPredictionCol("rpc0");
        DecisionTreeClassificationModel model3 = dt3.fit(train).setPredictionCol("prediction3").setProbabilityCol("pc3").setRawPredictionCol("rpc3");

        train = model.transform(train);
        train = model4.transform(train);
        train = model0.transform(train);
        train = model3.transform(train);

        train = train.withColumn("result", functions.expr("cast(if(prediction4=0, 4, if(prediction0=0, 0, if(prediction3=0, 3, prediction))) as double)"));

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("result")
                .setMetricName("accuracy");
        double accuracy = evaluator.evaluate(train);
        System.out.println("accuracy: " + accuracy);

        ResearchUtil.printPercentageConfusionMatrix(train, "label", "result");

        if (printTestDataResult) {
            //// TEST DATA PREDICTION
            Dataset<Row> test = InitialData.getTestDF();
            test = assembler.transform(test);

            test = pca.transform(test);
            test = pca4.transform(test);
            test = pca0.transform(test);
            test = pca3.transform(test);

            test = model.transform(test);
            test = model4.transform(test);
            test = model0.transform(test);
            test = model3.transform(test);

            test = test.withColumn("result", functions.expr("cast(if(prediction4=0, 4, if(prediction0=0, 0, if(prediction3=0, 3, prediction))) as double)"));

            test.select("result").coalesce(1).write().mode(SaveMode.Overwrite).option("separator", ";").csv("dt.csv");
        }
    }
}
