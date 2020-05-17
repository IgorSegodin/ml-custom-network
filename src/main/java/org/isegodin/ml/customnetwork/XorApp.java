package org.isegodin.ml.customnetwork;

import lombok.Data;
import lombok.RequiredArgsConstructor;
import lombok.SneakyThrows;
import org.isegodin.ml.customnetwork.calc.NeuralNetworkResultCalculator;
import org.isegodin.ml.customnetwork.data.FeedforwardResultData;
import org.isegodin.ml.customnetwork.data.NetworkBuilder;
import org.isegodin.ml.customnetwork.data.NeuralNetworkData;
import org.isegodin.ml.customnetwork.train.NeuralNetworkBackpropagationAlgorithm;

import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

/**
 * @author isegodin
 */
public class XorApp {

    @SneakyThrows
    public static void main(String[] args) {
        Supplier<NeuralNetworkData> networkInitializer = () -> NetworkBuilder.builder(2, 1)
                .addLayer(2)
                .build();

        NeuralNetworkData neuralNetworkData = networkInitializer.get();

        List<TrainData> trainData = Arrays.asList(
                new TrainData(new double[]{0, 0}, new double[]{0}),
                new TrainData(new double[]{0, 1}, new double[]{1}),
                new TrainData(new double[]{1, 1}, new double[]{0}),
                new TrainData(new double[]{1, 0}, new double[]{1})
        );

        double alpha = 0.5;

        double error = 1;

        int count = 0;
        int resetCount = 0;

        while (error > 0.01) {
            double avgError = 0;
            for (TrainData d : trainData) {
                FeedforwardResultData calcResult = NeuralNetworkResultCalculator.calcResult(d.getInput(), neuralNetworkData);

                avgError += NeuralNetworkBackpropagationAlgorithm.train(d.getTarget(), calcResult, neuralNetworkData, alpha);
            }
            error = avgError / trainData.size();
            count++;
            if (count > 100000) {
                count = 0;
                error = 1;
                resetCount++;
                neuralNetworkData = networkInitializer.get();
            }
        }

        System.out.println("Result after iterations:" + count + ", reset count: " + resetCount + ", error: " + error);

        for (TrainData d : trainData) {
            System.out.println(Arrays.toString(NeuralNetworkResultCalculator.calcResult(d.getInput(), neuralNetworkData).getFinalOut()));
        }
    }

    @Data
    @RequiredArgsConstructor
    private static class TrainData {
        private final double[] input;
        private final double[] target;
    }
}
