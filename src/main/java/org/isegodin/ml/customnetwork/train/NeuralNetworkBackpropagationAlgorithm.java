package org.isegodin.ml.customnetwork.train;

import org.isegodin.ml.customnetwork.data.ActivationFunctions;
import org.isegodin.ml.customnetwork.data.NeuralNetworkData;
import org.isegodin.ml.customnetwork.data.FeedforwardResultData;

/**
 * @author isegodin
 */
public class NeuralNetworkBackpropagationAlgorithm {

    public static void train(double[] target, FeedforwardResultData resultData, NeuralNetworkData networkData, double learningRate) {



        for (int l = networkData.getLayers().length - 1; l >= 0; l--) {
            NeuralNetworkData.Layer layer = networkData.getLayers()[l];

//            double[] weightedNodeErrors = new double[networkData.getLayers().length];

            for (int n = 0; n < layer.getNodes().length; n++) {
                NeuralNetworkData.Node node = layer.getNodes()[n];

                double nodeError;

                double nodeOut = resultData.getLayerData()[l].getOut()[n];
                double derivativeNodeOut = ActivationFunctions.SIGMOID.calcDerivative(nodeOut);

                if (l == networkData.getLayers().length - 1) {
                    double nodeTarget = target[n];

                    nodeError = (nodeTarget - nodeOut) * derivativeNodeOut;
                } else {

                    nodeError = sum * derivativeNodeOut;
                }

                for (int i = 0; i < node.getWeights().length; i++) {

                    double input = l == 0 ? resultData.getInput()[i] : resultData.getLayerData()[l - 1].getOut()[i];

                    double deltaWeight = nodeError * input;

                    double weight = node.getWeights()[i];

                    node.getWeights()[i] = weight + learningRate * deltaWeight;
                }
            }



            break;
            // TODO how to calc currentTarget for previous layer
//            currentTarget = ...;
        }


        /*

            k - node idx
            t - node target
            O - node real out
            nodeError=(t - O)*deriv(O)

            // input - previous layer node output, for specified weight
            deltaWeight = learningRate * nodeError * input


            // hidden
            nodeError=(t - O) * O * Sum(weigtedNodeError)
            where the Sum term adds the weighted error signal for all nodes, k,  in the output layer.



            hiddenError = deriv(h1Out) * dTotalE_dH1Out * input


                dTotalE_dH1Out (sum) = (t1 - o1Out) * deriv(O1) * w5(Out)
                                            + (t2 - o2Out) * deriv(O2) * w7(Out)

         */

    }

}
