package org.isegodin.ml.customnetwork.data;

import lombok.RequiredArgsConstructor;

import java.util.function.Function;

/**
 * @author isegodin
 */
@RequiredArgsConstructor
public enum ActivationFunctions {

    /*
        https://en.wikipedia.org/wiki/Rectifier_(neural_networks)

        logistic sigmoid
        hyperbolic tangent
        rectifier
        Softplus
     */

    SIGMOID(
            (net) -> 1.0 / (1.0 + Math.exp(-net)),
            (out) -> out * (1 - out)
    );

    private final Function<Double, Double> function;
    private final Function<Double, Double> derivative;

    public double calcOut(double net) {
        return function.apply(net);
    }

    public double calcDerivative(double out) {
        return derivative.apply(out);
    }

}
