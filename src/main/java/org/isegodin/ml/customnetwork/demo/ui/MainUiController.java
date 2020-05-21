package org.isegodin.ml.customnetwork.demo.ui;

import javafx.embed.swing.SwingFXUtils;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.geometry.Point2D;
import javafx.scene.SnapshotParameters;
import javafx.scene.SnapshotResult;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Label;
import javafx.scene.image.WritableImage;
import javafx.scene.input.MouseEvent;
import javafx.scene.paint.Color;
import javafx.scene.transform.Transform;
import lombok.RequiredArgsConstructor;
import lombok.SneakyThrows;
import org.isegodin.ml.customnetwork.network.SimpleNeuralNetwork;
import org.isegodin.ml.customnetwork.util.ArrayUtil;

import javax.imageio.ImageIO;
import java.awt.image.Raster;
import java.awt.image.RenderedImage;
import java.nio.file.Paths;

/**
 * @author isegodin
 */
@RequiredArgsConstructor
public class MainUiController {

    private final MainConfig config;

    private SimpleNeuralNetwork neuralNetwork;

    @FXML
    private Label predictionLabel;

    @FXML
    private Canvas canvas;

    private Point2D cursorPoint;

    private boolean draw;

    @FXML
    public void initialize() {
        clearCanvas(null);

        neuralNetwork = SimpleNeuralNetwork.loadFromFile(config.getNeuralNetworkModelFilePath());
    }

    @FXML
    private void clearCanvas(ActionEvent actionEvent) {
        GraphicsContext context2D = canvas.getGraphicsContext2D();
        context2D.setFill(Color.BLACK);
        context2D.fillRect(0, 0, canvas.getWidth(), canvas.getHeight());
    }

    @FXML
    private void onCanvasDrawPress(MouseEvent mouseEvent) {
        draw = true;
        this.cursorPoint = new Point2D(mouseEvent.getX(), mouseEvent.getY());
    }

    @FXML
    private void onCanvasDrawRelease(MouseEvent mouseEvent) {
        draw = false;
        cursorPoint = null;

        predictionLabel.setText("...");

        predictNumber(null);
    }

    @FXML
    private void onCanvasDrawMove(MouseEvent mouseEvent) {
        Point2D previousPoint = this.cursorPoint;
        this.cursorPoint = new Point2D(mouseEvent.getX(), mouseEvent.getY());

        if (draw && previousPoint != null) {
            GraphicsContext context2D = canvas.getGraphicsContext2D();
            context2D.setStroke(Color.RED);
            context2D.setLineWidth(16);
            context2D.strokeLine(previousPoint.getX(), previousPoint.getY(), cursorPoint.getX(), cursorPoint.getY());
        }
    }

    @FXML
    private void predictNumber(ActionEvent actionEvent) {
        WritableImage image = new WritableImage(28, 28);

        SnapshotParameters params = new SnapshotParameters();

        params.setTransform(Transform.scale(
                28 / canvas.getWidth(),
                28 / canvas.getHeight()
        ));

        canvas.snapshot(
                this::processImage,
                params,
                image
        );
    }

    @SneakyThrows
    private Void processImage(SnapshotResult snapshotResult) {
        RenderedImage renderedImage = SwingFXUtils.fromFXImage(snapshotResult.getImage(), null);
        ImageIO.write(
                renderedImage,
                "png",
                Paths.get(config.getResizedImageFilePath()).toFile()
        );

        double[] input = imageToInput(renderedImage);

        double[] out = neuralNetwork.feedForward(input);

        predictionLabel.setText("Result: " + ArrayUtil.findMaxValueIndex(out));

        return null;
    }

    @SneakyThrows
    private static double[] imageToInput(RenderedImage image) {
        Raster imageData = image.getData();

        double[] input = new double[image.getHeight() * image.getWidth()];

        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                int idx = x + y * image.getWidth();

                input[idx] = imageData.getPixel(x, y, new int[4])[0];
            }
        }

        return input;
    }
}
