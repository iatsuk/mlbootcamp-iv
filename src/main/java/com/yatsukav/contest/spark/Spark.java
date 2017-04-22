package com.yatsukav.contest.spark;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;

public class Spark {
    private static SparkSession session = null;

    public static void init() {
        if (session != null) return;
        session = new SparkSession.Builder()
                .appName("MLBootCampIV_Yatsuk")
                .master("local[*]")
                .getOrCreate();
        session.sparkContext().setLogLevel("WARN");
    }

    public static void stop() {
        if (session == null) return;
        session.stop();
    }

    public static SparkSession session() {
        init();
        return session;
    }

    public static JavaSparkContext jsc() {
        init();
        return new JavaSparkContext(session().sparkContext());
    }
}
