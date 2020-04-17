package org.isegodin.ml.customnetwork;

import lombok.SneakyThrows;

/**
 * @author isegodin
 */
public class StaticCalculationExample {

    /*

    (i1)  (h1)  (o1)
    (i2)  (h2)  (o2)
    (b1)  (b2)
     */

    @SneakyThrows
    public static void main(String[] args) {

        double w1;
        double w2;
        double w3;
        double w4;
        double w5;
        double w6;
        double w7;
        double w8;

        // Input
        double i1 = 0.05;
        double i2 = 0.1;
        double b1I = 1;
        double b1W = 0.35;

        // Hidden Layer 1
        double[] h1Weights = {w1 = 0.15, w2 = 0.20};
        double[] h2Weights = {w3 = 0.25, w4 = 0.3};
        double b2I = 1;
        double b2W = 0.6;

        // Output Layer
        double[] o1Weights = {w5 = 0.4, w6 = 0.45};
        double[] o2Weights = {w7 = 0.5, w8 = 0.55};

        // Target
        double t1 = 0.01;
        double t2 = 0.99;

        //------------------------------------

        //------Calc Hidden Layer 1

        double h1Net = i1 * h1Weights[0] + i2 * h1Weights[1] + b1I * b1W;
        double h1Out = activationFunc(h1Net);

        double h2Net = i1 * h2Weights[0] + i2 * h2Weights[1] + b1I * b1W;
        double h2Out = activationFunc(h2Net);

        //------Calc Output Layer

        double o1Net = h1Out * o1Weights[0] + h2Out * o1Weights[1] + b2I * b2W;
        double o1Out = activationFunc(o1Net);

        double o2Net = h1Out * o2Weights[0] + h2Out * o2Weights[1] + b2I * b2W;
        double o2Out = activationFunc(o2Net);

        //------Back propagation for Output Layer

        double alpha = 0.5; // learning rate

        double dTotalE_dO1Out = -(t1 - o1Out);
        double dO1Out_dO1Net = activationFuncDerivative(o1Out);
        double dO1Net_dW5 = h1Out;
        double dO1Net_dW6 = h2Out;
        double dTotalE_dW5 = dTotalE_dO1Out * dO1Out_dO1Net * dO1Net_dW5;
        double dTotalE_dW6 = dTotalE_dO1Out * dO1Out_dO1Net * dO1Net_dW6;

        double w5_ = w5 - alpha * dTotalE_dW5; // 0.3589
        double w6_ = w6 - alpha * dTotalE_dW6; // 0.4086

        double dTotalE_dO2Out = -(t2 - o2Out);
        double dO2Out_dO2Net = activationFuncDerivative(o2Out);
        double dO2Net_dW7 = h1Out;
        double dO2Net_dW8 = h2Out;
        double dTotalE_dW7 = dTotalE_dO2Out * dO2Out_dO2Net * dO2Net_dW7;
        double dTotalE_dW8 = dTotalE_dO2Out * dO2Out_dO2Net * dO2Net_dW8;

        double w7_ = w7 - alpha * dTotalE_dW7; // 0.5113
        double w8_ = w8 - alpha * dTotalE_dW8; // 0.5613

        //------Back propagation for Hidden Layer 1

//        double dTotalE_dW1 = dtotalE_dH1Out * dH1Out_dH1Net * dH1Net_dW1




        System.out.println();
    }

    private static double activationFunc(double netValue) {
        return 1.0 / (1.0 + Math.exp(-netValue));
    }

    /**
     * F(x) * (1 - F(x))
     *
     * F - activation function
     *
     * F(x) - previous output value
     */
    private static double activationFuncDerivative(double outValue) {
        return outValue * (1 - outValue);
    }
}
