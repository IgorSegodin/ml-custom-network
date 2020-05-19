package org.isegodin.ml.customnetwork.demo.ui;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.stage.Stage;

/**
 * @author isegodin
 */
public class MainApplication extends Application {

    @Override
    public void start(Stage stage) throws Exception {
        Parent root = FXMLLoader.load(getClass().getResource("/ui/main-ui.fxml"));

        Scene scene = new Scene(root, 270, 300);

        stage.setTitle("Hello World!");
        stage.setScene(scene);
        stage.show();
    }
}
