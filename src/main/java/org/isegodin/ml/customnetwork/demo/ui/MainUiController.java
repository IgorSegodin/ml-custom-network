package org.isegodin.ml.customnetwork.demo.ui;

import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.geometry.Point2D;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Label;
import javafx.scene.input.MouseEvent;
import javafx.scene.paint.Color;

/**
 * @author isegodin
 */
public class MainUiController {

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

    public void onCanvasDrawPress(MouseEvent mouseEvent) {
        draw = true;
        this.cursorPoint = new Point2D(mouseEvent.getX(), mouseEvent.getY());
    }

    public void onCanvasDrawRelease(MouseEvent mouseEvent) {
        draw = false;
        cursorPoint = null;
    }

    public void onCanvasDrawMove(MouseEvent mouseEvent) {
        Point2D previousPoint = this.cursorPoint;
        this.cursorPoint = new Point2D(mouseEvent.getX(), mouseEvent.getY());

        if (draw && previousPoint != null) {
            GraphicsContext context2D = canvas.getGraphicsContext2D();
            context2D.setStroke(Color.RED);
            context2D.setLineWidth(3);
            context2D.strokeLine(previousPoint.getX(), previousPoint.getY(), cursorPoint.getX(), cursorPoint.getY());
        }
    }
}
