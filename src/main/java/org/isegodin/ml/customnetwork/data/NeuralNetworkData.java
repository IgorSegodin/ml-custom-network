package org.isegodin.ml.customnetwork.data;

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
    }

    @Data
    @AllArgsConstructor
    @NoArgsConstructor
    public static class Node {
        private double[] weights;
    }
}
