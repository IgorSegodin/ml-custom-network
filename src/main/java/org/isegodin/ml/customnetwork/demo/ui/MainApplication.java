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
        FXMLLoader loader = new FXMLLoader(getClass().getResource("/ui/main-ui.fxml"));
        loader.setControllerFactory(clazz -> new MainUiController(
                MainConfig.builder()
                        .neuralNetworkModelFilePath("epoch-10-fit-0_9766-i784-l256_relu-l64_relu-l32_sigmoid-o10_sigmoid.json")
                        .resizedImageFilePath("/Users/isegodin/GitHub/machine-learning-custom-network/draw_image.png")
                        .build()
        ));
        Parent root = loader.load();

        Scene scene = new Scene(root, 270, 300);

        stage.setTitle("Hello World!");
        stage.setScene(scene);
        stage.show();
    }
}
