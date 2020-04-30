package org.isegodin.ml.customnetwork.data;

import lombok.Data;

import java.util.LinkedList;
import java.util.List;

/**
 * @author isegodin
 */
public class NetworkBuilder {

    private final int inputSize;
    private final int outputSize;
    private final double outputWeightedBias;
    private final List<LayerInfo> layerInfoList = new LinkedList<>();

    private NetworkBuilder(int inputSize, int outputSize, double outputWeightedBias) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.outputWeightedBias = outputWeightedBias;
    }

    public static NetworkBuilder builder(int inputSize, int outputSize, double outputWeightedBias) {
        return new NetworkBuilder(inputSize, outputSize, outputWeightedBias);
    }

    public NetworkBuilder addLayer(int size, double weightedBias) {
        layerInfoList.add(new LayerInfo(size, weightedBias));
        return this;
    }

    public NeuralNetworkData build() {
        List<NeuralNetworkData.Layer> layers = new LinkedList<>();

        int layerInputSize = inputSize;

        for (LayerInfo l : layerInfoList) {
            NeuralNetworkData.Layer layer = new NeuralNetworkData.Layer(new NeuralNetworkData.Node[l.getSize()], l.getWeightedBias());
            layers.add(layer);

            for (int n = 0; n < layer.getNodes().length; n++) {
                layer.getNodes()[n] = new NeuralNetworkData.Node(new double[layerInputSize]);
            }

            layerInputSize = l.getSize();
        }

        NeuralNetworkData.Layer outLayer = new NeuralNetworkData.Layer(new NeuralNetworkData.Node[outputSize], outputWeightedBias);
        layers.add(outLayer);

        for (int n = 0; n < outLayer.getNodes().length; n++) {
            outLayer.getNodes()[n] = new NeuralNetworkData.Node(new double[layerInputSize]);
        }

        NeuralNetworkData networkData = new NeuralNetworkData(layers.toArray(new NeuralNetworkData.Layer[0]));

        WeightInitializer.initializeWeights(networkData);

        return networkData;
    }

    @Data
    private static class LayerInfo {
        private final int size;
        private final double weightedBias;
    }
}
