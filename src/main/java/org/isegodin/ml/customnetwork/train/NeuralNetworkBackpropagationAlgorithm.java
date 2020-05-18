package org.isegodin.ml.customnetwork.train;

import org.isegodin.ml.customnetwork.data.ActivationFunctions;
import org.isegodin.ml.customnetwork.data.FeedforwardResultData;
import org.isegodin.ml.customnetwork.data.NeuralNetworkData;

/**
 * @author isegodin
 */
public class NeuralNetworkBackpropagationAlgorithm {

    /**
     * @return total error
     */
    public static double train(double[] target, FeedforwardResultData resultData, NeuralNetworkData networkData, double learningRate) {

        // (layer, input)
        double[][] layerWeightedNodeErrorSum = new double[networkData.getLayers().length][];

        for (int l = networkData.getLayers().length - 1; l >= 0; l--) {
            NeuralNetworkData.Layer layer = networkData.getLayers()[l];

            // (node, input)
            double[][] weightedNodeErrors = new double[layer.getNodes().length][];

            for (int n = 0; n < layer.getNodes().length; n++) {
                NeuralNetworkData.Node node = layer.getNodes()[n];

                // (input)
                double[] weightedErrors = new double[node.getWeights().length];

                double nodeError;

                double nodeOut = resultData.getLayerData()[l].getOut()[n];
                double derivativeNodeOut = ActivationFunctions.SIGMOID.calcDerivative(nodeOut);

                if (l == networkData.getLayers().length - 1) {
                    double nodeTarget = target[n];

                    nodeError = (nodeTarget - nodeOut) * derivativeNodeOut;
                } else {

                    nodeError = layerWeightedNodeErrorSum[l + 1][n] * derivativeNodeOut;
                }

                for (int i = 0; i < node.getWeights().length; i++) {

                    double input = l == 0 ? resultData.getInput()[i] : resultData.getLayerData()[l - 1].getOut()[i];

                    double deltaWeight = nodeError * input;

                    double weight = node.getWeights()[i];

                    weightedErrors[i] = nodeError * weight;

                    node.getWeights()[i] = weight + learningRate * deltaWeight;
                }

                weightedNodeErrors[n] = weightedErrors;
            }

            // Calc weighted sum

            int inputNumber = weightedNodeErrors[0].length;

            double[] weightedSum = new double[inputNumber];

            for (int i = 0; i < inputNumber; i++) {
                double sum = 0;

                for (int n = 0; n < weightedNodeErrors.length; n++) {
                    sum += weightedNodeErrors[n][i];
                }
                weightedSum[i] = sum;
            }

            layerWeightedNodeErrorSum[l] = weightedSum;
        }

        return calcTotalError(target, resultData.getFinalOut());
    }

    public static double calcTotalError(double[] target, double[] out) {
        double totalError = 0;

        for (int i = 0; i < out.length; i++) {
            double t = target[i];
            double o = out[i];

            totalError += Math.pow(t - o, 2) / 2;
        }

        return totalError;
    }

}
