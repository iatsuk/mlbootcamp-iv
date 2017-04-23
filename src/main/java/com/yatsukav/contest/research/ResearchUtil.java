package com.yatsukav.contest.research;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import scala.Tuple2;

import java.util.*;
import java.util.stream.Collectors;

public interface ResearchUtil {

    static void printConfusionMatrix(Dataset<Row> df, String label, String prediction) {
        // prepare dataset to old mllib format
        JavaRDD<Tuple2<Object, Object>> predictionAndLabels = df.select(prediction, label)
                .toJavaRDD().map(v1 -> new Tuple2<>(v1.get(0), v1.get(1)));

        // Get evaluation metrics.
        MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());

        // Confusion matrix
        Matrix confusion = metrics.confusionMatrix();
        System.out.println("Confusion matrix: \n" + confusion + "\n");

        // Overall statistics
        System.out.println("Accuracy = " + metrics.accuracy() + "\n");

        // Stats by labels
        for (int i = 0; i < metrics.labels().length; i++) {
            System.out.format("Class %f precision = %f\n", metrics.labels()[i], metrics.precision(
                    metrics.labels()[i]));
            System.out.format("Class %f recall = %f\n", metrics.labels()[i], metrics.recall(
                    metrics.labels()[i]));
            System.out.format("Class %f F1 score = %f\n", metrics.labels()[i], metrics.fMeasure(
                    metrics.labels()[i]));
            System.out.println();
        }

        //Weighted stats
        System.out.format("Weighted precision = %f\n", metrics.weightedPrecision());
        System.out.format("Weighted recall = %f\n", metrics.weightedRecall());
        System.out.format("Weighted F1 score = %f\n", metrics.weightedFMeasure());
        System.out.format("Weighted false positive rate = %f\n", metrics.weightedFalsePositiveRate());
    }

    static void printPercentageConfusionMatrix(Dataset<Row> df, String label, String prediction) {
        df = df.select(label, prediction);
        List<Double> labels = df.select(label)
                .sort(label)
                .distinct()
                .map((MapFunction<Row, Double>) row -> row.getDouble(0), Encoders.DOUBLE())
                .collectAsList();

        List<Tuple2<Double, Double>> labelAndPredictions = df.select(label, prediction)
                .collectAsList()
                .stream()
                .map(row -> new Tuple2<>(row.getDouble(0), row.getDouble(1)))
                .collect(Collectors.toList());

        /*
         * Key is label
         * Value is list where:
         * 0 element - count of current label in dataset
         * 1..n element - count of (label+1) predicted as current label
         */
        Map<Double, List<Integer>> preMatrix = new HashMap<>();
        labels.forEach(d -> preMatrix.put(d, new ArrayList<>(Collections.nCopies(labels.size() + 1, 0))));

        for (Tuple2<Double, Double> labelAndPrediction : labelAndPredictions) {
            List<Integer> preMatrixElem = preMatrix.get(labelAndPrediction._1());
            preMatrixElem.set(0, preMatrixElem.get(0) + 1); // update count of current label
            preMatrixElem.set((int) (labelAndPrediction._2() + 1),
                    preMatrixElem.get((int) (labelAndPrediction._2() + 1)) + 1);
        }

        // print matrix
        for (Double currentLabel : labels) {
            System.out.print("\t" + currentLabel.intValue());
        }
        System.out.println();

        for (Double currentLabel : labels) {
            System.out.print(currentLabel.intValue() + ":\t");

            List<Integer> preMatrixElem = preMatrix.get(currentLabel);
            for (int i = 1; i < preMatrixElem.size(); i++) {
                System.out.print((int) (100d * preMatrixElem.get(i) / preMatrixElem.get(0)));
                System.out.print("\t");
            }
            System.out.println();
        }

        System.out.println();
        for (Double currentLabel : labels) {
            System.out.println("Total count of " + currentLabel + " is: " + preMatrix.get(currentLabel).get(0));
        }
    }
}
