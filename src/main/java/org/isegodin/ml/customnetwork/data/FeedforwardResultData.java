package org.isegodin.ml.customnetwork.data;

import lombok.Data;
import lombok.RequiredArgsConstructor;

/**
 * @author isegodin
 */
@Data
@RequiredArgsConstructor
public class FeedforwardResultData {

    private final LayerResultData[] layerData;

    public double[] getFinalOut() {
        return layerData[layerData.length - 1].getOut();
    }

    @Data
    @RequiredArgsConstructor
    public static class LayerResultData {

        private final double[] net;
        private final double[] out;

    }
}
