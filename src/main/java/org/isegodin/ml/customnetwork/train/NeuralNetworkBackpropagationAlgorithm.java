package org.isegodin.ml.customnetwork.train;

import org.isegodin.ml.customnetwork.data.ActivationFunctions;
import org.isegodin.ml.customnetwork.data.NeuralNetworkData;
import org.isegodin.ml.customnetwork.data.FeedforwardResultData;

/**
 * @author isegodin
 */
public class NeuralNetworkBackpropagationAlgorithm {

    public static void train(double[] target, FeedforwardResultData resultData, NeuralNetworkData networkData, double learningRate) {

        double[] currentTarget = target;

        for (int l = networkData.getLayers().length - 1; l >= 0; l--) {
            NeuralNetworkData.Layer layer = networkData.getLayers()[l];

            for (int n = 0; n < layer.getNodes().length; n++) {
                NeuralNetworkData.Node node = layer.getNodes()[n];

                double nodeTarget = currentTarget[n];
                double nodeOut = resultData.getLayerData()[l].getOut()[n];

                double nodeError = (nodeTarget - nodeOut) * ActivationFunctions.SIGMOID.calcDerivative(nodeOut);

                for (int i = 0; i < node.getWeights().length; i++) {

                    double weightInput = l == 0 ? resultData.getInput()[i] : resultData.getLayerData()[l - 1].getOut()[i];

                    double deltaWeight = nodeError * weightInput;

                    double weight = node.getWeights()[i];

                    node.getWeights()[i] = weight + learningRate * deltaWeight;
                }
            }

            break;
            // TODO how to calc currentTarget for previous layer
//            currentTarget = ...;
        }



        /*

            t - node target
            O - node real out
            nodeError=(t - O)*deriv(O)

            // input - previous layer node output, for specified weight
            deltaWeight = learningRate * nodeError * input

         */

    }

}
