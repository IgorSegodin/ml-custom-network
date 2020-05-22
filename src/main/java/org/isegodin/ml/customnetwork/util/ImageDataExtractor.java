package org.isegodin.ml.customnetwork.util;

import lombok.Getter;
import lombok.RequiredArgsConstructor;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.awt.image.RenderedImage;
import java.awt.image.WritableRaster;

/**
 * @author isegodin
 */
public class ImageDataExtractor {

    /**
     * @return Array of size width*height with pixel values.
     */
    public static double[] extractImageData(RenderedImage image) {
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

    /**
     * Crop, scale, equalize size.
     *
     * @param originalImage unprocessed image
     * @param targetSize    image size dimension, for example if value = 28, then resulting image will be 28x28
     */
    public static RenderedImage preProcessImage(RenderedImage originalImage, int targetSize) {
        return equalizeSize(
                scale(
                        crop(
                                convertToBufferedImage(originalImage),
                                findBounds(originalImage)
                        ),
                        targetSize
                )
        );
    }

    /**
     * Make square image, preserving proportions.
     * Use biggest dimension as target size.
     */
    private static BufferedImage equalizeSize(BufferedImage input) {
        int targetSize = input.getHeight() > input.getWidth() ? input.getHeight() : input.getWidth();

        BufferedImage normalizedImage = new BufferedImage(targetSize, targetSize, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D normalizedImageGraphics = normalizedImage.createGraphics();

        int shiftX = (targetSize - input.getWidth()) / 2;
        int shiftY = (targetSize - input.getHeight()) / 2;

        normalizedImageGraphics.drawImage(
                input,
                shiftX, shiftY,
                input.getWidth() + shiftX, input.getHeight() + shiftY,

                0, 0,
                input.getWidth(), input.getHeight(),
                Color.BLACK,
                null
        );

        return normalizedImage;
    }

    private static BufferedImage scale(BufferedImage input, int targetSize) {
        double scale;

        if (input.getHeight() > input.getWidth()) {
            scale = (double) targetSize / input.getHeight();
        } else {
            scale = (double) targetSize / input.getWidth();
        }

        int scaledWidth = Double.valueOf(input.getWidth() * scale).intValue();
        int scaledHeight = Double.valueOf(input.getHeight() * scale).intValue();

        BufferedImage scaledImage = new BufferedImage(scaledWidth, scaledHeight, BufferedImage.TYPE_BYTE_GRAY);
        scaledImage.createGraphics()
                .drawImage(input.getScaledInstance(scaledWidth, scaledHeight, java.awt.Image.SCALE_SMOOTH), 0, 0, null);

        return scaledImage;
    }

    private static BufferedImage crop(BufferedImage input, Bounds bounds) {
        return input.getSubimage(
                bounds.getTopLeft().getX(), bounds.getTopLeft().getY(),
                bounds.getWidth(), bounds.getHeight()
        );
    }

    private static Bounds findBounds(RenderedImage input) {
        Raster imageData = input.getData();

        int minX = input.getWidth();
        int minY = input.getHeight();

        int maxX = 0;
        int maxY = 0;

        for (int y = 0; y < input.getHeight(); y++) {
            for (int x = 0; x < input.getWidth(); x++) {

                int pixel = imageData.getPixel(x, y, new int[4])[0];

                if (pixel == 0) {
                    continue;
                }

                minX = minX > x ? x : minX;
                minY = minY > y ? y : minY;

                maxX = maxX < x ? x : maxX;
                maxY = maxY < y ? y : maxY;
            }
        }
        return new Bounds(
                new Point(minX, minY),
                new Point(maxX, maxY)
        );
    }

    private static BufferedImage convertToBufferedImage(RenderedImage input) {
        if (input instanceof BufferedImage) {
            return (BufferedImage) input;
        }

        BufferedImage bufferedImage = new BufferedImage(input.getWidth(), input.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
        WritableRaster writableRaster = bufferedImage.getRaster();
        input.copyData(writableRaster);
        return bufferedImage;
    }

    /**
     * TODO try to crop, scale and normalize in one step
     */
    private BufferedImage __test(BufferedImage input, Rectangle rect) {
        int targetWidth = (int) rect.getWidth();
        int targetHeight = (int) rect.getHeight();


        BufferedImage target = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D graphics = target.createGraphics();

        graphics.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        graphics.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
        graphics.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        graphics.drawImage(
                input,
                0, 0,
                targetWidth, targetHeight,
                (int) rect.getX(), (int) rect.getY(),
                (int) (rect.getX() + rect.getWidth()), (int) (rect.getY() + rect.getHeight()),
                null
        );

        graphics.dispose();

        return target;
    }

    @RequiredArgsConstructor
    @Getter
    private static class Point {
        private final int x;
        private final int y;
    }

    @RequiredArgsConstructor
    @Getter
    private static class Bounds {
        private final Point topLeft;
        private final Point bottomRight;

        public int getWidth() {
            return bottomRight.getX() - topLeft.getX() + 1;
        }

        public int getHeight() {
            return bottomRight.getY() - topLeft.getY() + 1;
        }
    }


}
