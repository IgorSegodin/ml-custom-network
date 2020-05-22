package org.isegodin.ml.customnetwork.network.data;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * @author isegodin
 */
@Data
@AllArgsConstructor
@NoArgsConstructor
public class NeuralNetworkData {

    private Layer[] layers;

    @Data
    @AllArgsConstructor
    @NoArgsConstructor
    public static class Layer {
        private Node[] nodes;
        private ActivationFunctions function;
    }

    @Data
    @AllArgsConstructor
    @NoArgsConstructor
    public static class Node {
        private double[] weights;
    }
}
