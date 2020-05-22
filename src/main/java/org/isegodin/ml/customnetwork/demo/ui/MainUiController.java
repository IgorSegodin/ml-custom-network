package org.isegodin.ml.customnetwork.demo.ui;

import javafx.embed.swing.SwingFXUtils;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.geometry.Point2D;
import javafx.scene.SnapshotParameters;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Label;
import javafx.scene.image.WritableImage;
import javafx.scene.input.MouseEvent;
import javafx.scene.paint.Color;
import lombok.RequiredArgsConstructor;

import java.awt.image.RenderedImage;
import java.util.function.Function;

/**
 * @author isegodin
 */
@RequiredArgsConstructor
public class MainUiController {

    private final Function<RenderedImage, Integer> predictNumberFunction;

    @FXML
    private Label predictionLabel;

    @FXML
    private Canvas canvas;

    private Point2D cursorPoint;

    private boolean draw;

    @FXML
    public void initialize() {
        clearCanvas(null);
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

        predictNumber();
    }

    @FXML
    private void onCanvasDrawMove(MouseEvent mouseEvent) {
        Point2D previousPoint = this.cursorPoint;
        this.cursorPoint = new Point2D(mouseEvent.getX(), mouseEvent.getY());

        if (draw && previousPoint != null) {
            GraphicsContext context2D = canvas.getGraphicsContext2D();
            context2D.setStroke(Color.WHITE);
            context2D.setLineWidth(16);
            context2D.strokeLine(previousPoint.getX(), previousPoint.getY(), cursorPoint.getX(), cursorPoint.getY());
        }
    }

    private void predictNumber() {
        WritableImage image = new WritableImage(
                (int)canvas.getWidth(),
                (int)canvas.getHeight()
        );

//        SnapshotParameters params = new SnapshotParameters();
//
//        params.setTransform(Transform.scale(
//                28 / canvas.getWidth(),
//                28 / canvas.getHeight()
//        ));

        canvas.snapshot(
                (snapshotResult) -> {
                    RenderedImage renderedImage = SwingFXUtils.fromFXImage(snapshotResult.getImage(), null);
                    int result = predictNumberFunction.apply(renderedImage);
                    predictionLabel.setText("Result: " + result);
                    return null;
                },
                new SnapshotParameters(),
                image
        );
    }
}
