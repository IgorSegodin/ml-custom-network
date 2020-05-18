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
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

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

        final AtomicLong alpha = new AtomicLong(Double.doubleToLongBits(0.5));

        final int chunkSize = 100;
        final AtomicInteger batchCounter = new AtomicInteger();
        final List<SimpleNeuralNetwork.TrainData> trainDataList = new ArrayList<>(chunkSize);

        double avgError = 1;
        boolean first = true;

        while (avgError > 0.3) {

            trainDataList.clear();
            batchCounter.set(0);

            Files.list(Paths.get(MNIST_FOLDER, "train"))
                    .forEach(p -> {
                        if (batchCounter.getAndIncrement() % chunkSize == 0 && !trainDataList.isEmpty()) {

                            neuralNetwork.train(trainDataList, Double.longBitsToDouble(alpha.get()));

                            trainDataList.clear();
                        }

                        trainDataList.add(
                                extractTranDataFromImage(p)
                        );
                    });

            neuralNetwork.addEpoch();

            neuralNetwork.save(MODEL_FOLDER);


            trainDataList.clear();
            batchCounter.set(0);

            AtomicLong totalError = new AtomicLong();
            AtomicInteger count = new AtomicInteger();


            Files.list(Paths.get(MNIST_FOLDER, "test"))
                    .forEach(p -> {
                        if (batchCounter.getAndIncrement() % chunkSize == 0 && !trainDataList.isEmpty()) {

                            double error = neuralNetwork.evaluate(trainDataList);

                            count.getAndIncrement();

                            totalError.set(
                                    Double.doubleToLongBits(error + Double.longBitsToDouble(totalError.get()))
                            );

                            trainDataList.clear();
                        }

                        trainDataList.add(
                                extractTranDataFromImage(p)
                        );
                    });

            double newError = Double.longBitsToDouble(totalError.get()) / count.get();

            if (!first) {

                double newAlpha = Double.longBitsToDouble(alpha.get());
                newAlpha += newAlpha * (avgError - newError) / avgError * -1;
                newAlpha = Math.min(0.5, newAlpha);

                alpha.set(Double.doubleToLongBits(newAlpha));
            }

            avgError = newError;

            first = false;

            System.out.println("Test avgError = " + avgError + ", alpha = " + Double.longBitsToDouble(alpha.get()));
        }
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
