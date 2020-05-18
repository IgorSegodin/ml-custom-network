package org.isegodin.ml.customnetwork;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.SneakyThrows;
import org.isegodin.ml.customnetwork.calc.NeuralNetworkResultCalculator;
import org.isegodin.ml.customnetwork.data.NeuralNetworkData;
import org.isegodin.ml.customnetwork.data.FeedforwardResultData;
import org.isegodin.ml.customnetwork.train.NeuralNetworkBackpropagationAlgorithm;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.util.Arrays;

/**
 * @author isegodin
 */
public class App {

    private static ObjectMapper objectMapper = new ObjectMapper();

    @SneakyThrows
    public static void main(String[] args) {
//        NeuralNetworkData neuralNetworkData;
//
//        try (InputStream is = new FileInputStream(new File("/Users/isegodin/GitHub/machine-learning-custom-network/nn-data.json"))) {
//            neuralNetworkData = objectMapper.readValue(is, NeuralNetworkData.class);
//        }
//
//        double[] input = {0.05, 0.1};
//
//        FeedforwardResultData calcResult = NeuralNetworkResultCalculator.calcResult(input, neuralNetworkData);
//
//        double[] expectedOut = {0.7513650695523157, 0.7729284653214625};
//
//        System.out.println(neuralNetworkData);
//        System.out.println(calcResult);
//        System.out.println("Expected: " + Arrays.toString(expectedOut));
//        System.out.println("Actual:   " + Arrays.toString(calcResult.getFinalOut()));
//
//
//        NeuralNetworkBackpropagationAlgorithm.train(new double[]{0.01, 0.99}, calcResult, neuralNetworkData, 0.5);

    }
}
