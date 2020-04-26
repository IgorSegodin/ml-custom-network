package org.isegodin.ml.customnetwork.data;

import lombok.Data;

/**
 * @author isegodin
 */
@Data
public class NeuralNetworkData {

    private Layer[] layers;

    @Data
    public static class Layer {
        private Node[] nodes;

        private double weightedBias;
    }

    @Data
    public static class Node {
        private double[] weights;
    }
}
