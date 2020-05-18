package org.isegodin.ml.customnetwork;

import lombok.SneakyThrows;
import org.isegodin.ml.customnetwork.data.NetworkBuilder;
import org.isegodin.ml.customnetwork.network.SimpleNeuralNetwork;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

/**
 * @author isegodin
 */
public class App {

    private static final String MODEL_FOLDER = "/Users/isegodin/GitHub/machine-learning-custom-network/models";
    private static final String MNIST_FOLDER = "/Users/isegodin/GitHub/machine-learning-custom-network/mnist";

    private static Map<Integer, double[]> digitTargetMap = buildDigitTargetMap();

    private static Map<Integer, double[]> buildDigitTargetMap() {
        Map<Integer, double[]> map = new HashMap<>();

        for (int i = 0; i < 10; i++) {
            double[] target = new double[10];

            target[i] = 1;

            map.put(i, target);
        }
        return map;
    }


    @SneakyThrows
    public static void main(String[] args) {
        SimpleNeuralNetwork neuralNetwork = SimpleNeuralNetwork.loadLatestOrNew(
                MODEL_FOLDER,
                () -> NetworkBuilder.builder(28 * 28, 10)
                        .addLayer(15)
        );


        final int chunkSize = 100;
        final AtomicInteger counter = new AtomicInteger();
        final List<SimpleNeuralNetwork.TrainData> trainDataList = new ArrayList<>(chunkSize);

        Files.list(Paths.get(MNIST_FOLDER, "train"))
                .forEach(p -> {
                    if (!trainDataList.isEmpty() && counter.getAndIncrement() % chunkSize == 0) {
                        // TODO train
                        trainDataList.clear();
                    }

                    trainDataList.add(
                            extractTranDataFromImage(p)
                    );
                });


//        neuralNetwork.addEpoch();
//
//        neuralNetwork.save(MODEL_FOLDER);


//        BufferedImage read = ImageIO.read(new File("/Users/isegodin/GitHub/machine-learning-custom-network/mnist/train/000000-num5.png"));
//        int[] pixel = read.getData().getPixel(13, 13, new int[1]);

        System.out.println();


//        NeuralNetworkData neuralNetworkData;
//
//        try (InputStream is = new FileInputStream(new File("/Users/isegodin/GitHub/machine-learning-custom-network/nn-data.json"))) {
//            neuralNetworkData = objectMapper.readValue(is, NeuralNetworkData.class);
//        }
//
//        double[] input = {0.05, 0.1};
//
//        FeedforwardResultData calcResult = NeuralNetworkResultCalculator.calcResult(input, neuralNetworkData);
//
//        double[] expectedOut = {0.7513650695523157, 0.7729284653214625};
//
//        System.out.println(neuralNetworkData);
//        System.out.println(calcResult);
//        System.out.println("Expected: " + Arrays.toString(expectedOut));
//        System.out.println("Actual:   " + Arrays.toString(calcResult.getFinalOut()));
//
//
//        NeuralNetworkBackpropagationAlgorithm.train(new double[]{0.01, 0.99}, calcResult, neuralNetworkData, 0.5);

    }

    private static SimpleNeuralNetwork.TrainData extractTranDataFromImage(Path image) {
        String filename = image.getFileName().toString();
        String filenameWithoutExtension = filename.substring(0, filename.lastIndexOf("."));
        int number = Integer.valueOf(
                filenameWithoutExtension.split("-")[1].substring("num".length())
        );

        return new SimpleNeuralNetwork.TrainData();
    }
}
