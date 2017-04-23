package com.yatsukav.contest;

import com.yatsukav.contest.data.InitialData;
import com.yatsukav.contest.research.DecisionTreeResearch;
import com.yatsukav.contest.spark.Spark;

import java.io.IOException;

public class App {

    public static void main(String[] args) throws IOException {
        Spark.init();
        InitialData.init();

        DecisionTreeResearch.run(79, false);

        Spark.stop();
    }

}
