package org.isegodin.ml.customnetwork.demo.ui;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;
import lombok.SneakyThrows;
import org.isegodin.ml.customnetwork.network.SimpleNeuralNetwork;
import org.isegodin.ml.customnetwork.util.ArrayUtil;
import org.isegodin.ml.customnetwork.util.ImageDataExtractor;

import javax.imageio.ImageIO;
import java.awt.image.RenderedImage;
import java.io.File;

/**
 * @author isegodin
 */
public class MainApplication extends Application {

    private int imageSize = 28;

//    private String modelPath = "/Users/isegodin/GitHub/machine-learning-custom-network/models/epoch-36.json"; // 0.9814
//    private String tempProcessedImagePath = "/Users/isegodin/GitHub/machine-learning-custom-network/debug/processed_image.png";

    private SimpleNeuralNetwork neuralNetwork;

    @Override
    public void start(Stage stage) throws Exception {
//        neuralNetwork = SimpleNeuralNetwork.loadFromFile(modelPath);
        neuralNetwork = SimpleNeuralNetwork.loadFromBytes(
                getClass().getResource("/model/epoch-36.json").openStream().readAllBytes()
        );

        FXMLLoader loader = new FXMLLoader(getClass().getResource("/ui/main-ui.fxml"));

        loader.setControllerFactory(clazz -> new MainUiController(this::predictNumber));

        Parent root = loader.load();

        Scene scene = new Scene(root, 270, 300);

        stage.setTitle("Number guesser");
        stage.setScene(scene);
        stage.show();
    }

    @SneakyThrows
    private Integer predictNumber(RenderedImage rawImage) {
        RenderedImage processedImage = ImageDataExtractor.preProcessImage(rawImage, imageSize);

//        ImageIO.write(processedImage, "png", new File(tempProcessedImagePath));

        double[] out = neuralNetwork.feedForward(ImageDataExtractor.extractImageData(processedImage));

        return ArrayUtil.findMaxValueIndex(out);
    }
}
