package org.isegodin.ml.customnetwork.network.calc;

import lombok.Data;
import org.isegodin.ml.customnetwork.network.data.FeedforwardResultData;
import org.isegodin.ml.customnetwork.network.data.NeuralNetworkData;

/**
 * @author isegodin
 */
@Data
public class NeuralNetworkResultCalculator {

    public static FeedforwardResultData calcResult(double[] input, NeuralNetworkData networkData) {
        NeuralNetworkData.Layer firstLayer = networkData.getLayers()[0];

        int requiredInputSize = firstLayer.getNodes()[0].getWeights().length - 1;
        if (input.length != requiredInputSize) {
            throw new IllegalArgumentException("Wrong input size " + input.length + ", required " + requiredInputSize);
        }

        input = appendBias(input);

        double[] currentInput = input;

        FeedforwardResultData.LayerResultData[] layerResult = new FeedforwardResultData.LayerResultData[networkData.getLayers().length];

        for (int l = 0; l < networkData.getLayers().length; l++) {
            NeuralNetworkData.Layer layer = networkData.getLayers()[l];

            double[] layerOutResult = new double[layer.getNodes().length];

            for (int n = 0; n < layer.getNodes().length; n++) {
                NeuralNetworkData.Node node = layer.getNodes()[n];

                double nodeNetResult = calcNodeNet(currentInput, node);
                layerOutResult[n] = layer.getFunction().calcOut(nodeNetResult);
            }

            if (l < networkData.getLayers().length - 1) {
                layerOutResult = appendBias(layerOutResult);
            }

            currentInput = layerOutResult;

            layerResult[l] = new FeedforwardResultData.LayerResultData(layerOutResult);
        }

        return new FeedforwardResultData(input, layerResult);
    }

    private static double[] appendBias(double[] input) {
        double[] out = new double[input.length + 1];
        System.arraycopy(input, 0, out, 0, input.length);
        out[out.length - 1] = 1;
        return out;
    }

    private static double calcNodeNet(double[] input, NeuralNetworkData.Node node) {
        if (input.length != node.getWeights().length) {
            throw new IllegalStateException("Wrong node input size " + input.length + ", required " + node.getWeights().length);
        }

        double result = 0.0;

        for (int i = 0; i < input.length; i++) {
            result += node.getWeights()[i] * input[i];
        }

        return result;
    }

}
