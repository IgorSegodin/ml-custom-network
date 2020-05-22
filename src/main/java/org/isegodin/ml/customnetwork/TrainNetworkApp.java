package org.isegodin.ml.customnetwork;

import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.isegodin.ml.customnetwork.network.SimpleNeuralNetwork;
import org.isegodin.ml.customnetwork.network.data.ActivationFunctions;
import org.isegodin.ml.customnetwork.network.data.NetworkBuilder;
import org.isegodin.ml.customnetwork.util.ImageDataExtractor;

import javax.imageio.ImageIO;
import java.awt.image.RenderedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

/**
 * @author isegodin
 */
@Slf4j
public class TrainNetworkApp {

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

    private static long lastSaveTimeMillis = 0;

    /*
        I 784
        L 128 relu
        L 64 relu
        O 10 softmax
     */

    @SneakyThrows
    public static void main(String[] args) {
        SimpleNeuralNetwork neuralNetwork = SimpleNeuralNetwork.loadLatestOrNew(
                MODEL_FOLDER,
                () -> NetworkBuilder.builder(28 * 28)
                        .addLayer(256, ActivationFunctions.ReLU)
                        .addLayer(64, ActivationFunctions.ReLU)
                        .addLayer(32, ActivationFunctions.SIGMOID)
                        .output(10, ActivationFunctions.SIGMOID)
        );

        List<SimpleNeuralNetwork.TrainData> trainData = Files.list(Paths.get(MNIST_FOLDER, "train"))
                .map(TrainNetworkApp::extractTranDataFromImage)
                .collect(Collectors.toList());


        List<SimpleNeuralNetwork.TrainData> testData = Files.list(Paths.get(MNIST_FOLDER, "test"))
                .map(TrainNetworkApp::extractTranDataFromImage)
                .collect(Collectors.toList());

        log.info("Data loaded");

        double alpha = 0.001;

        double fit = neuralNetwork.evaluate(testData);

        while (fit < 0.99) {

            neuralNetwork.train(trainData, alpha);

            neuralNetwork.addEpoch();

            // Save model not more often than each 30 seconds
            if (System.currentTimeMillis() - lastSaveTimeMillis > TimeUnit.SECONDS.toMillis(30)) {
                neuralNetwork.save(MODEL_FOLDER);
                lastSaveTimeMillis = System.currentTimeMillis();
            }

            fit = neuralNetwork.evaluate(testData);

            log.info("Test fit = " + fit + ", epoch = " + neuralNetwork.getEpoch());
        }
        neuralNetwork.save(MODEL_FOLDER);
    }

    @SneakyThrows
    private static SimpleNeuralNetwork.TrainData extractTranDataFromImage(Path imagePath) {
        String filename = imagePath.getFileName().toString();
        String filenameWithoutExtension = filename.substring(0, filename.lastIndexOf("."));
        int number = Integer.valueOf(
                filenameWithoutExtension.split("-")[1].substring("num".length())
        );

        RenderedImage processedImage = ImageDataExtractor.preProcessImage(ImageIO.read(imagePath.toFile()), 28);

        double[] input = ImageDataExtractor.extractImageData(processedImage);

        return new SimpleNeuralNetwork.TrainData(input, digitTargetMap.get(number));
    }
}
