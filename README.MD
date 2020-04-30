* [+Theoretical backpropagation](https://www.python-course.eu/neural_networks_backpropagation.php)
* [+Step by step backpropagation](https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/)
* [+Backpropagation for generalized network with Python Code example](http://www.adeveloperdiary.com/data-science/machine-learning/understand-and-implement-the-backpropagation-algorithm-from-scratch-in-python/)
* [+Generalized backpropagation algorithm](https://www.cse.unsw.edu.au/~cs9417ml/MLP2/BackPropagation.html)
* [+Another step by step example with numbers](https://www.anotsorandomwalk.com/backpropagation-example-with-numbers-step-by-step/)
* [Explanation with example 1](https://medium.com/@14prakash/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c)
* [Explanation with example 2](https://towardsdatascience.com/multi-layer-neural-networks-with-sigmoid-function-deep-learning-for-rookies-2-bf464f09eb7f)
* [Explanation with example 3](https://techburst.io/understanding-backpropagation-56f0d32a3084)
* [Explanation with example 4](https://towardsdatascience.com/everything-you-need-to-know-about-neural-networks-and-backpropagation-machine-learning-made-easy-e5285bc2be3a)

```java
/*
        Assign all network inputs and output
        Initialize all weights with small random numbers, typically between -1 and 1

        repeat

            for every pattern in the training set

                Present the pattern to the network

        //        Propagated the input forward through the network:
                    for each layer in the network
                        for every node in the layer
                            1. Calculate the weight sum of the inputs to the node
                            2. Add the threshold to the sum
                            3. Calculate the activation for the node
                        end
                    end

        //        Propagate the errors backward through the network
                     for every node in the output layer
                        calculate the error signal
                    end

                    for all hidden layers
                        for every node in the layer
                            1. Calculate the node's signal error
                            2. Update each node's weight in the network
                        end
                    end

        //        Calculate Global Error
                    Calculate the Error Function

            end

        while ((maximum  number of iterations < than specified) AND
                  (Error Function is > than specified))
     */
```