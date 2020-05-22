package org.isegodin.ml.customnetwork.util;

/**
 * @author isegodin
 */
public class ArrayUtil {

    public static int findMaxValueIndex(double[] array) {
        double maxValue = array[0];
        int idx = 0;

        for (int i = 1; i < array.length; i++) {
            double val = array[i];
            if (val > maxValue) {
                maxValue = val;
                idx = i;
            }
        }

        return idx;
    }
}
