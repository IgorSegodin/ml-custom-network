package org.isegodin.ml.customnetwork.demo.ui;

import lombok.AccessLevel;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;

/**
 * @author isegodin
 */
@Builder
@Getter
@AllArgsConstructor(access = AccessLevel.PRIVATE)
public class MainConfig {

    private String neuralNetworkModelFilePath;
    private String resizedImageFilePath;
}
