package org.isegodin.ml.customnetwork.data;

import lombok.Data;

import java.util.LinkedList;
import java.util.List;

/**
 * @author isegodin
 */
public class NetworkBuilder {

    private final int inputSize;
    private int outputSize;
    private ActivationFunctions outputFunc;
    private final List<LayerInfo> layerInfoList = new LinkedList<>();

    private NetworkBuilder(int inputSize) {
        this.inputSize = inputSize;
    }

    public static NetworkBuilder builder(int inputSize) {
        return new NetworkBuilder(inputSize);
    }

    public NetworkBuilder addLayer(int size, ActivationFunctions func) {
        layerInfoList.add(new LayerInfo(size, func));
        return this;
    }

    public NetworkBuilder output(int size,  ActivationFunctions func) {
        outputSize = size;
        outputFunc = func;
        return this;
    }

    public NeuralNetworkData build() {
        if (inputSize == 0 || outputSize == 0) {
            throw new IllegalStateException("Empty network builder");
        }

        List<NeuralNetworkData.Layer> layers = new LinkedList<>();

        int layerInputSize = inputSize + 1; // 1 additional bias input

        for (LayerInfo l : layerInfoList) {
            NeuralNetworkData.Layer layer = new NeuralNetworkData.Layer(new NeuralNetworkData.Node[l.getSize()], l.getFunc());
            layers.add(layer);

            for (int n = 0; n < layer.getNodes().length; n++) {
                layer.getNodes()[n] = new NeuralNetworkData.Node(new double[layerInputSize]);
            }

            layerInputSize = l.getSize() + 1; // 1 additional bias input
        }

        NeuralNetworkData.Layer outLayer = new NeuralNetworkData.Layer(new NeuralNetworkData.Node[outputSize], outputFunc);
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
        private final ActivationFunctions func;
    }
}
