package com.yatsukav.contest.data;

import com.yatsukav.contest.GlobalVars;
import com.yatsukav.contest.spark.Spark;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SaveMode;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class InitialData {
    private static final String INITIAL_DATA_DIR = "initial_data";

    private static final Path TRAIN_CSV_PATH = Paths.get(GlobalVars.ROOT_DIR, GlobalVars.WAREHOUSE_DIR, INITIAL_DATA_DIR, "train.csv");
    private static final String XTRAIN_CSV_PATH = InitialData.class.getClassLoader().getResource("x_train.csv").getFile();
    private static final String YTRAIN_CSV_PATH = InitialData.class.getClassLoader().getResource("y_train.csv").getFile();
    private static final String TEST_CSV_PATH = InitialData.class.getClassLoader().getResource("x_test.csv").getFile();

    private static final Path TRAIN_DF_PATH = Paths.get(GlobalVars.ROOT_DIR, GlobalVars.WAREHOUSE_DIR, INITIAL_DATA_DIR, "train.parquet");
    private static final Path XTRAIN_DF_PATH = Paths.get(GlobalVars.ROOT_DIR, GlobalVars.WAREHOUSE_DIR, INITIAL_DATA_DIR, "x_train.parquet");
    private static final Path YTRAIN_DF_PATH = Paths.get(GlobalVars.ROOT_DIR, GlobalVars.WAREHOUSE_DIR, INITIAL_DATA_DIR, "y_train.parquet");
    private static final Path TEST_DF_PATH = Paths.get(GlobalVars.ROOT_DIR, GlobalVars.WAREHOUSE_DIR, INITIAL_DATA_DIR, "test.parquet");

    public static void init() throws IOException {
        if (!Files.exists(TRAIN_CSV_PATH)) addTrainCsv();
        if (!Files.exists(XTRAIN_DF_PATH)) addXTrainDF();
        if (!Files.exists(YTRAIN_DF_PATH)) addYTrainDF();
        if (!Files.exists(TRAIN_DF_PATH)) addTrainDF();
        if (!Files.exists(TEST_DF_PATH)) addTestDF();
    }

    public static Dataset<Row> getXTrainDF() {
        return Spark.session().read().parquet(XTRAIN_DF_PATH.toString());
    }

    public static Dataset<Row> getYTrainDF() {
        return Spark.session().read().parquet(YTRAIN_DF_PATH.toString());
    }

    public static Dataset<Row> getTrainDF() {
        return Spark.session().read().parquet(TRAIN_DF_PATH.toString());
    }

    public static Dataset<Row> getTestDF() {
        return Spark.session().read().parquet(TEST_DF_PATH.toString());
    }


    private static void addTrainCsv() throws IOException {
        File xtrain = new File(XTRAIN_CSV_PATH);
        BufferedReader readerX = new BufferedReader(new FileReader(xtrain));

        File ytrain = new File(YTRAIN_CSV_PATH);
        BufferedReader readerY = new BufferedReader(new FileReader(ytrain));

        Files.createDirectories(TRAIN_CSV_PATH.getParent());
        FileWriter out = new FileWriter(TRAIN_CSV_PATH.toFile());

        final char SEPARATOR = ';';
        final char LINE_END = '\n';
        while (true) {
            String label = readerY.readLine();
            String features = readerX.readLine();
            if (Objects.isNull(label) && Objects.isNull(features)) break;
            out.write(label + SEPARATOR + features + LINE_END);
        }

        out.close();
        readerY.close();
        readerX.close();
    }


    private static void addXTrainDF() {
        parseCsv(XTRAIN_CSV_PATH, false).write().mode(SaveMode.Overwrite).parquet(XTRAIN_DF_PATH.toString());
    }

    private static void addYTrainDF() {
        parseCsv(YTRAIN_CSV_PATH, true).write().mode(SaveMode.Overwrite).parquet(YTRAIN_DF_PATH.toString());
    }

    private static void addTrainDF() {
        parseCsv(TRAIN_CSV_PATH.toString(), true).write().mode(SaveMode.Overwrite).parquet(TRAIN_DF_PATH.toString());
    }

    private static void addTestDF() {
        parseCsv(TEST_CSV_PATH, false).write().mode(SaveMode.Overwrite).parquet(TEST_DF_PATH.toString());
    }

    private static Dataset<Row> parseCsv(String csvPath, boolean hasLabel) {
        JavaRDD<Row> rdd = Spark.jsc().textFile(csvPath)
                .map(line -> Stream.of(line.split(";")).map(Double::valueOf).toArray())
                .map(RowFactory::create);

        int offset = hasLabel ? -1 : 0;
        List<StructField> schema = IntStream.range(offset, rdd.first().size() + offset)
                .mapToObj(i -> "c" + i)
                .map(col -> DataTypes.createStructField(col, DataTypes.DoubleType, false))
                .collect(Collectors.toList());

        Dataset<Row> df = Spark.session().createDataFrame(rdd, DataTypes.createStructType(schema));
        if (hasLabel) df = df.withColumnRenamed("c-1", "label");

        return df;
    }

}
