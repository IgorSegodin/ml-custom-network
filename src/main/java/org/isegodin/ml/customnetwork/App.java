package org.isegodin.ml.customnetwork;

import lombok.SneakyThrows;
import org.isegodin.ml.customnetwork.data.NetworkBuilder;
import org.isegodin.ml.customnetwork.network.SimpleNeuralNetwork;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
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
                () -> NetworkBuilder.builder(28 * 28, 10)
                        .addLayer(128)
                        .addLayer(64)
        );

        List<SimpleNeuralNetwork.TrainData> trainData = Files.list(Paths.get(MNIST_FOLDER, "train"))
                .map(App::extractTranDataFromImage)
                .collect(Collectors.toList());


        List<SimpleNeuralNetwork.TrainData> testData = Files.list(Paths.get(MNIST_FOLDER, "test"))
                .map(App::extractTranDataFromImage)
                .collect(Collectors.toList());

        System.out.println("Data loaded");

        double alpha = 0.001;

        double fit = neuralNetwork.evaluate(testData);

        while (fit < 0.98) {

            neuralNetwork.train(trainData, alpha);

            neuralNetwork.addEpoch();

            if (neuralNetwork.getEpoch() % 10 == 0) {
                neuralNetwork.save(MODEL_FOLDER);
            }

            fit = neuralNetwork.evaluate(testData);

            System.out.println("Test fit = " + fit + ", epoch = " + neuralNetwork.getEpoch());
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

        BufferedImage bufferedImage = ImageIO.read(imagePath.toFile());
        Raster imageData = bufferedImage.getData();

        double[] input = new double[bufferedImage.getHeight() * bufferedImage.getWidth()];

        for (int y = 0; y < bufferedImage.getHeight(); y++) {
            for (int x = 0; x < bufferedImage.getWidth(); x++) {
                int idx = x + y * bufferedImage.getWidth();

                input[idx] = imageData.getPixel(x, y, new int[1])[0];
            }
        }

        return new SimpleNeuralNetwork.TrainData(input, digitTargetMap.get(number));
    }
}
