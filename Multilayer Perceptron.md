

[link to the Github Repository here](https://github.com/shankarj67/Mlblr/blob/master/Mlblr_assignment.ipynb)

MULTILAYER PERCEPTRON:

`Step 0`: Input and Output


```python
import numpy as np

X = np.array([[1,0,1,0], [1,0,1,1], [0,1,0,1]])
y = np.array([[1],[1],[0]])
```

##### input and output values

| X |   |   |   |
|---|---|---|---|
| 1 | 0 | 1 | 0 |
| 1 | 0 | 1 | 1 |
| 0 | 1 | 0 | 1 |


| y |
|---|
| 1 |
| 1 |
| 0 |


`Step 1`: Initialize weights and biases with random values 

```python
import numpy as np

wh=np.random.random((4,3))
bh=np.random.random((1,3))
wout=np.random.random((3,1))
bout=np.random.random((1,1))
```
Output:

```Python

wh
0.54435702 |  0.74385286 |  0.27654845
0.06162608 | 0.35072172 | 0.13677035
0.68342269 |  0.3646227 |  0.80326076
0.4249686 |  0.82472174 | 0.17472863

bh
0.15306184 |  0.14485056 | 0.99034259
        
wout
0.81867933
0.62656632
0.32906894

bout
0.19470554
```

`Step 2`: Calculate hidden layer input:

> hidden_layer_input = matrix_dot_product(X,wh) + bh

```Python

  hidden_layer
1.38084156 | 1.25332613 |  2.0701518 
 1.80581016 | 2.07804787 | 2.24488043
 0.63965653 |  1.32029402 | 1.30184157
```

`Step 3`: Perform non-linear transformation on hidden linear input
> hiddenlayer_activations = sigmoid(hidden_layer_input)

```python
  hidden_layer_activations
 0.79912612 | 0.7778751  | 0.88796806
 0.85885473 |  0.88875117 |  0.90420802
 0.65467581 | 0.78923062 |  0.78614475
 
 ```
 
Step 4`: Perform linear and non-linear transformation of hidden layer activation at output layer
> output_layer_input = matrix_dot_product (hiddenlayer_activations * wout ) + bout output = sigmoid(output_layer_input)

```Python
  output
  0.8359677 
  0.85223517
  0.81515735
```

`Step 5`: Calculate gradient of Error(E) at output layer

> Error = y-output

```Python
    Error
  0.1640323 
  0.14776483
 -0.81515735
```

`Step 6`: Compute slope at output and hidden layer
> Slope_output_layer= derivatives_sigmoid(output)
> Slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)

```Python
  Slope output layer
 0.01924864
 0.02684511
 0.03587376
     
  Slope hidden layer
 0.16052356 | 0.17278543 | 0.09948078
 0.12122328 | 0.09887253 | 0.08661588
 0.22607539 | 0.16634565 | 0.16812118



  ```
  
`Step 7`: Compute delta at output layer

> d_output = E * slope_output_layer*lr

```Python
   delta putput
  0.00031574
  0.00039668
 -0.00292428
```

`Step 8`: Calculate Error at hidden layer

>Error_at_hidden_layer = matrix_dot_product(d_output, wout.Transpose)


```Python
error at hidden layer
0.00025849 | 0.00019783 | 0.0001039
0.00032475 | 0.00024854 | 0.00013053
-0.00239404 | -0.00183225 | -0.00096229
```

`Step 9`: Compute delta at hidden layer

> d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer


```Python
Delta_hiddenlayer

1.00830870e-05 |  1.14665200e-05 | 1.18272461e-05
2.03813678e-05 |  2.79061964e-05  | 1.50757717e-05
.-1.20430166e-04 | -1.34137479e-04 | -2.33625457e-04
```

`Step 10`: Update weight at both output and hidden layer

> wout = wout + matrix_dot_product (hiddenlayer_activations.Transpose, d_output) * learning_rate

> wh = wh+ matrix_dot_product (X.Transpose,d_hiddenlayer) * learning_rate

```Python
wout
0.81854718
0.62639534
0.32890296

  wh
  0.54436007 | 0.7438568  | 0.27655114
  0.06161404 | 0.35070831 | 0.13674699
  0.68342574 | 0.36462664 | 0.80326345
  0.4249586 |  0.82471112 | 0.17470677
```

`Step 11`: Update biases at both output and hidden layer

> bh = bh + sum(d_hiddenlayer, axis=0) * learning_rate
> bout = bout + sum(d_output, axis=0)*learning_rate 

```Python
  bh
   0.15305284 | 0.14484108 | 0.99032192

  bout
   0.19448435   
```