package org.isegodin.ml.customnetwork.util;

import org.junit.jupiter.api.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

/**
 * @author isegodin
 */
class ImageDataExtractorTest {

    @Test
    void testExtractData() throws IOException {
        BufferedImage image = ImageIO.read(getClass().getResource("/images/small-3x4.png"));
        double[] data = ImageDataExtractor.extractImageData(image);

        assertArrayEquals(
                new double[]{
                        0, 255, 0,
                        255, 0, 255,
                        255, 0, 255,
                        0, 255, 0
                },
                data
        );
    }

    @Test
    void testPreProcessImage() throws Exception {
        assertArrayEquals(
                ImageDataExtractor.extractImageData(
                        ImageIO.read(getClass().getResource("/images/small-4x4.png"))
                ),

                ImageDataExtractor.extractImageData(
                        ImageDataExtractor.preProcessImage(
                                ImageIO.read(getClass().getResource("/images/big-shift-8x10.png")),
                                4
                        )
                )
        );
    }

}