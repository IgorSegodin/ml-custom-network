package org.isegodin.ml.customnetwork.network;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.Builder;
import lombok.Data;
import lombok.RequiredArgsConstructor;
import lombok.SneakyThrows;
import org.isegodin.ml.customnetwork.network.calc.NeuralNetworkResultCalculator;
import org.isegodin.ml.customnetwork.network.data.FeedforwardResultData;
import org.isegodin.ml.customnetwork.network.data.NetworkBuilder;
import org.isegodin.ml.customnetwork.network.data.NeuralNetworkData;
import org.isegodin.ml.customnetwork.network.train.NeuralNetworkBackpropagationAlgorithm;
import org.isegodin.ml.customnetwork.util.ArrayUtil;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.function.Supplier;

/**
 * @author isegodin
 */
public class SimpleNeuralNetwork {

    private static ObjectMapper objectMapper = new ObjectMapper();

    private final NetworkBuilder networkBuilder;

    private NeuralNetworkData data;

    private int epoch;

    public SimpleNeuralNetwork(NetworkBuilder networkBuilder) {
        this.networkBuilder = networkBuilder;
        this.data = networkBuilder.build();
    }

    public SimpleNeuralNetwork(NetworkBuilder networkBuilder, NeuralNetworkData data, int epoch) {
        this.networkBuilder = networkBuilder;
        this.data = data;
        this.epoch = epoch;
    }

    public double[] feedForward(double[] input) {
        FeedforwardResultData calcResult = NeuralNetworkResultCalculator.calcResult(input, data);
        return calcResult.getFinalOut();
    }

    public double evaluate(List<TrainData> dataList) {
        int correct = 0;

        for (TrainData trainData : dataList) {
            FeedforwardResultData calcResult = NeuralNetworkResultCalculator.calcResult(trainData.getInput(), data);

            int valueIdx = findMaxValueIndex(calcResult.getFinalOut());

            if (trainData.getTarget()[valueIdx] == 1) {
                correct++;
            }
        }

        return (double) correct / dataList.size();
    }

    public double train(List<TrainData> dataList, double alpha) {
        int correct = 0;

        for (TrainData trainData : dataList) {
            FeedforwardResultData calcResult = NeuralNetworkResultCalculator.calcResult(trainData.getInput(), data);

            int valueIdx = findMaxValueIndex(calcResult.getFinalOut());

            if (trainData.getTarget()[valueIdx] == 1) {
                correct++;
            }

            NeuralNetworkBackpropagationAlgorithm.train(trainData.getTarget(), calcResult, data, alpha);
        }

        return (double) correct / dataList.size();
    }

    private int findMaxValueIndex(double[] array) {
        return ArrayUtil.findMaxValueIndex(array);
    }

    public void addEpoch() {
        epoch++;
    }

    public int getEpoch() {
        return epoch;
    }

    @SneakyThrows
    public static SimpleNeuralNetwork loadLatestOrNew(String pathToFolder, Supplier<NetworkBuilder> builderSupplier) {
        if (pathToFolder == null || pathToFolder.trim().isEmpty()) {
            throw new IllegalArgumentException("Supplied path is empty: " + pathToFolder);
        }
        if (!Files.isDirectory(Paths.get(pathToFolder))) {
            throw new IllegalArgumentException("Supplied path is not a directory: " + pathToFolder);
        }

        Optional<FileInfo> latestModel = Files.list(Paths.get(pathToFolder))
                .filter(p -> Files.isRegularFile(p))
                .map(p -> {
                    String filename = p.getFileName().toString();
                    String filenameWithoutExtension = filename.substring(0, filename.lastIndexOf("."));
                    String[] split = filenameWithoutExtension.split("-");

                    if (split.length != 2) {
                        return null;
                    }
                    return FileInfo.builder()
                            .filename(filename)
                            .epoch(Integer.valueOf(split[1]))
                            .build();
                })
                .filter(Objects::nonNull)
                .max(Comparator.comparing(FileInfo::getEpoch));

        return latestModel.map(f -> {
            return loadNetworkFromFile(pathToFolder, builderSupplier, f);
        })
                .orElseGet(() -> new SimpleNeuralNetwork(builderSupplier.get()));
    }

    @SneakyThrows
    private static SimpleNeuralNetwork loadNetworkFromFile(String pathToFolder, Supplier<NetworkBuilder> builderSupplier, FileInfo f) {
        byte[] bytes = Files.readAllBytes(Paths.get(pathToFolder, f.getFilename()));
        NeuralNetworkData networkData = objectMapper.readValue(bytes, NeuralNetworkData.class);
        return new SimpleNeuralNetwork(builderSupplier.get(), networkData, f.getEpoch());
    }

    @SneakyThrows
    public static SimpleNeuralNetwork loadFromFile(String pathToFile) {
        // TODO refactor save names, use same method for loading one and multiple files
        byte[] bytes = Files.readAllBytes(Paths.get(pathToFile));
        return loadFromBytes(bytes);
    }

    @SneakyThrows
    public static SimpleNeuralNetwork loadFromBytes(byte[] bytes) {
        NeuralNetworkData networkData = objectMapper.readValue(bytes, NeuralNetworkData.class);
        return new SimpleNeuralNetwork(null, networkData, 0);
    }

    @SneakyThrows
    public void save(String pathToFolder) {
        if (pathToFolder == null || pathToFolder.trim().isEmpty()) {
            throw new IllegalArgumentException("Supplied path is empty: " + pathToFolder);
        }
        if (!Files.isDirectory(Paths.get(pathToFolder))) {
            throw new IllegalArgumentException("Supplied path is not a directory: " + pathToFolder);
        }

        byte[] bytes = objectMapper.writeValueAsBytes(data);

        Files.write(Paths.get(pathToFolder, "epoch-" + epoch + ".json"), bytes);
    }

    @Data
    @RequiredArgsConstructor
    public static class TrainData {
        private final double[] input;
        private final double[] target;
    }

    @Data
    @Builder
    private static class FileInfo {
        private String filename;
        private int epoch;
    }

}
