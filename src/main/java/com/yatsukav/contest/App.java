package com.yatsukav.contest;

import com.yatsukav.contest.data.InitialData;
import com.yatsukav.contest.research.DecisionTreeLabel0;
import com.yatsukav.contest.research.DecisionTreeLabel4;
import com.yatsukav.contest.research.DecisionTreeResearch;
import com.yatsukav.contest.spark.Spark;

import java.io.IOException;

public class App {

    public static void main(String[] args) throws IOException {
        Spark.init();
        InitialData.init();

        System.out.println("\n\n===== Multiclass decision tree =====");
        DecisionTreeResearch.run(79, false);

        System.out.println("\n\n===== Decision tree by 4 label =====");
        DecisionTreeLabel4.run(165);

        System.out.println("\n\n===== Decision tree by 0 label =====");
        DecisionTreeLabel0.run(100);

        for (int i = 1; i < 223; i++) {
        }

        Spark.stop();
    }

}
