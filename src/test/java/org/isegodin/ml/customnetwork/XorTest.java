package org.isegodin.ml.customnetwork;

import lombok.Data;
import lombok.RequiredArgsConstructor;
import org.isegodin.ml.customnetwork.network.calc.NeuralNetworkResultCalculator;
import org.isegodin.ml.customnetwork.network.data.ActivationFunctions;
import org.isegodin.ml.customnetwork.network.data.FeedforwardResultData;
import org.isegodin.ml.customnetwork.network.data.NetworkBuilder;
import org.isegodin.ml.customnetwork.network.data.NeuralNetworkData;
import org.isegodin.ml.customnetwork.network.train.NeuralNetworkBackpropagationAlgorithm;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * @author isegodin
 */
class XorTest {

    @Test
    void testSuccess() {

        Supplier<NeuralNetworkData> networkInitializer = () -> NetworkBuilder.builder(2)
                .addLayer(2, ActivationFunctions.SIGMOID)
                .output(1, ActivationFunctions.SIGMOID)
                .build();

        NeuralNetworkData neuralNetworkData = networkInitializer.get();

        List<TrainData> trainData = Arrays.asList(
                new TrainData(new double[]{0, 0}, new double[]{0}),
                new TrainData(new double[]{0, 1}, new double[]{1}),
                new TrainData(new double[]{1, 1}, new double[]{0}),
                new TrainData(new double[]{1, 0}, new double[]{1})
        );

        double maxAllowedError = 0.05;
        double maxIterationsBeforeResetWeights = 100000;
        double maxResetCount = 2;

        double alpha = 0.5;

        double error = maxAllowedError + 1;

        int count = 0;
        int resetCount = 0;

        boolean first = true;

        while (error > maxAllowedError) {
            double avgError = 0;
            for (TrainData d : trainData) {
                FeedforwardResultData calcResult = NeuralNetworkResultCalculator.calcResult(d.getInput(), neuralNetworkData);

                avgError += NeuralNetworkBackpropagationAlgorithm.train(d.getTarget(), calcResult, neuralNetworkData, alpha);
            }

            double newError = avgError / trainData.size();

            if (!first) {

                alpha += alpha * (error - newError) / error * -1;

                alpha = Math.min(0.5, alpha);
            }
//            System.out.println(alpha);

            error = newError;

            first = false;

            count++;
            if (count > maxIterationsBeforeResetWeights) {
                alpha = 0.5;
                count = 0;
                error = maxAllowedError + 1;
                resetCount++;
                neuralNetworkData = networkInitializer.get();
            }
            if (resetCount > maxResetCount) {
                throw new IllegalStateException("To many weight resets " + resetCount);
            }
        }

        System.out.println("Result after iterations:" + count + ", reset count: " + resetCount + ", error: " + error + ", alpha: " + alpha);

        assertEquals(
                0,
                calcResult(0, 0, neuralNetworkData)
        );

        assertEquals(
                1,
                calcResult(0, 1, neuralNetworkData)
        );

        assertEquals(
                1,
                calcResult(1, 0, neuralNetworkData)
        );

        assertEquals(
                0,
                calcResult(1, 1, neuralNetworkData)
        );

    }

    private int calcResult(int a, int b, NeuralNetworkData neuralNetworkData) {
        double[] out = NeuralNetworkResultCalculator.calcResult(new double[]{a, b}, neuralNetworkData).getFinalOut();

        return Long.valueOf(Math.round(out[0])).intValue();
    }

    @Data
    @RequiredArgsConstructor
    private static class TrainData {
        private final double[] input;
        private final double[] target;
    }

}
