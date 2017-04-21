package com.yatsukav.contest;

import org.apache.spark.sql.SparkSession;

public class App {

    public static void main(String[] args) {
        SparkSession spark = new SparkSession.Builder()
                .appName("MLBootCampIV_Yatsuk")
                .master("local[*]")
                .getOrCreate();
        spark.sparkContext().setLogLevel("WARN");


        spark.stop();
    }

}
