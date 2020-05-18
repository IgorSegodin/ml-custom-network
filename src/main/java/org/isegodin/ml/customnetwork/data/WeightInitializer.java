package org.isegodin.ml.customnetwork.data;

import java.util.PrimitiveIterator;
import java.util.Random;
import java.util.stream.DoubleStream;

/**
 * @author isegodin
 */
public class WeightInitializer {

    public static void initializeWeights(NeuralNetworkData networkData) {
        for (int l = 0; l < networkData.getLayers().length; l++) {
            NeuralNetworkData.Layer layer = networkData.getLayers()[l];
            for (int n = 0; n < layer.getNodes().length; n++) {
                NeuralNetworkData.Node node = layer.getNodes()[n];

                int inputSize = node.getWeights().length - 1; // bias weight is always last

                double from = - 1 / Math.sqrt(inputSize);
                double to = 1 / Math.sqrt(inputSize);

                DoubleStream random = new Random().doubles(from, to).limit(inputSize);
                PrimitiveIterator.OfDouble iterator = random.iterator();

                for (int i = 0; i < inputSize; i++) {
                    // TODO ensure weight can not be zero (dead neuron)
                    node.getWeights()[i] = iterator.next();
                }

                node.getWeights()[node.getWeights().length - 1] = 0; // initial bias weight should be zero
            }
        }
    }
}
