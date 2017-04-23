package com.yatsukav.contest;

import com.yatsukav.contest.data.InitialData;
import com.yatsukav.contest.research.DecisionTreeLabel4;
import com.yatsukav.contest.spark.Spark;

import java.io.IOException;

public class App {

    public static void main(String[] args) throws IOException {
        Spark.init();
        InitialData.init();

        //DecisionTreeResearch.run(79, false);
        DecisionTreeLabel4.run(165);
        for (int i = 1; i < 223; i++) {
        }

        Spark.stop();
    }

}
