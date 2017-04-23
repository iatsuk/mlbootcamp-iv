package com.yatsukav.contest;

import com.yatsukav.contest.data.InitialData;
import com.yatsukav.contest.research.decisiontree.*;
import com.yatsukav.contest.spark.Spark;

import java.io.IOException;

public class App {

    public static void main(String[] args) throws IOException {
        Spark.init();
        InitialData.init();

        System.out.println("\n\n===== Multiclass decision tree =====");
        DecisionTreeResearch.run(79, false);

        System.out.println("\n\n===== Decision tree by 0 label+ =====");
        DecisionTreeLabel0.run(100);

        System.out.println("\n\n===== Decision tree by 1 label =====");
        DecisionTreeLabel1.run(140);

        System.out.println("\n\n===== Decision tree by 2 label =====");
        DecisionTreeLabel2.run(140);

        System.out.println("\n\n===== Decision tree by 3 label =====");
        DecisionTreeLabel3.run(140);

        System.out.println("\n\n===== Decision tree by 4 label+ =====");
        DecisionTreeLabel4.run(165);

        System.out.println("\n\n===== Decision tree by 1 label!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! =====");
        for (int i = 1; i < 223; i++) {
            DecisionTreeLabel1.run(i);
        }


        System.out.println("\n\n===== Decision tree by 2 label!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! =====");
        for (int i = 1; i < 223; i++) {
            DecisionTreeLabel2.run(i);
        }


        System.out.println("\n\n===== Decision tree by 3 label!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! =====");
        for (int i = 1; i < 223; i++) {
            DecisionTreeLabel3.run(i);
        }

        Spark.stop();
    }

}
