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
    private final List<LayerInfo> layerInfoList = new LinkedList<>();

    private NetworkBuilder(int inputSize, int outputSize) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
    }

    public static NetworkBuilder builder(int inputSize, int outputSize) {
        return new NetworkBuilder(inputSize, outputSize);
    }

    public NetworkBuilder addLayer(int size) {
        layerInfoList.add(new LayerInfo(size));
        return this;
    }

    public NeuralNetworkData build() {
        List<NeuralNetworkData.Layer> layers = new LinkedList<>();

        int layerInputSize = inputSize + 1; // 1 additional bias input

        for (LayerInfo l : layerInfoList) {
            NeuralNetworkData.Layer layer = new NeuralNetworkData.Layer(new NeuralNetworkData.Node[l.getSize()]);
            layers.add(layer);

            for (int n = 0; n < layer.getNodes().length; n++) {
                layer.getNodes()[n] = new NeuralNetworkData.Node(new double[layerInputSize]);
            }

            layerInputSize = l.getSize() + 1; // 1 additional bias input
        }

        NeuralNetworkData.Layer outLayer = new NeuralNetworkData.Layer(new NeuralNetworkData.Node[outputSize]);
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
    }
}
