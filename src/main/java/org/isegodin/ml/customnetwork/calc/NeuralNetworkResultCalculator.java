package org.isegodin.ml.customnetwork.calc;

import lombok.Data;
import org.isegodin.ml.customnetwork.data.ActivationFunctions;
import org.isegodin.ml.customnetwork.data.NeuralNetworkData;
import org.isegodin.ml.customnetwork.data.FeedforwardResultData;

/**
 * @author isegodin
 */
@Data
public class NeuralNetworkResultCalculator {

    public static FeedforwardResultData calcResult(double[] input, NeuralNetworkData networkData) {
        NeuralNetworkData.Layer firstLayer = networkData.getLayers()[0];

        int requiredInputSize = firstLayer.getNodes()[0].getWeights().length;
        if (input.length != requiredInputSize) {
            throw new IllegalArgumentException("Wrong input size " + input.length + ", required " + requiredInputSize);
        }

        double[] currentInput = input;

        FeedforwardResultData.LayerResultData[] layerResult = new FeedforwardResultData.LayerResultData[networkData.getLayers().length];

        for (int l = 0; l < networkData.getLayers().length; l++) {
            NeuralNetworkData.Layer layer = networkData.getLayers()[l];

            double[] layerNetResult = new double[layer.getNodes().length];
            double[] layerOutResult = new double[layerNetResult.length];

            for (int n = 0; n < layer.getNodes().length; n++) {
                NeuralNetworkData.Node node = layer.getNodes()[n];

                layerNetResult[n] = calcNodeNet(currentInput, layer.getWeightedBias(), node);
                layerOutResult[n] = ActivationFunctions.SIGMOID.calcOut(layerNetResult[n]);
            }

            currentInput = layerOutResult;

            layerResult[l] = new FeedforwardResultData.LayerResultData(layerNetResult, layerOutResult);
        }

        return new FeedforwardResultData(input, layerResult);
    }

    private static double calcNodeNet(double[] input, double weightedBias, NeuralNetworkData.Node node) {
        if (input.length != node.getWeights().length) {
            throw new IllegalStateException("Wrong node input size " + input.length + ", required " + node.getWeights().length);
        }

        double result = 0.0;

        for (int i = 0; i < input.length; i++) {
            result += node.getWeights()[i] * input[i];
        }

        return result + weightedBias;
    }

}
