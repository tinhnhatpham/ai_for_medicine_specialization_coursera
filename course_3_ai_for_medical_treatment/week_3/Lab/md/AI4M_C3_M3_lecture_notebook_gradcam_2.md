
# GradCAM: Continuation (Part 2) - Lecture Notebook

In the previous lecture notebook (GradCAM Part 1) we explored what Grad-Cam is and why it is useful. We also looked at how we can compute the activations of a particular layer using Keras API. In this notebook we will check the other element that Grad-CAM requires, the gradients of the model's output with respect to our desired layer's output. This is the "Grad" portion of Grad-CAM. 

Let's dive into it!


```python
import keras
from keras import backend as K
from util import *
```

    Using TensorFlow backend.


The `load_C3M3_model()` function has been taken care of and as last time, its internals are out of the scope of this notebook.


```python
# Load the model we used last time
model = load_C3M3_model()
```

    Got loss weights
    Loaded DenseNet
    Added layers
    Compiled Model
    Loaded Weights


Kindly recall from the previous notebook (GradCAM Part 1) that our model has 428 layers. 

We are now interested in getting the gradients when the model outputs a specific class. For this we will use Keras backend's `gradients(..)` function. This function requires two arguments: 

  - Loss (scalar tensor)
  - List of variables
  
Since we want the gradients with respect to the output, we can use our model's output tensor:


```python
# Save model's output in a variable
y = model.output

# Print model's output
y
```




    <tf.Tensor 'dense_1/Sigmoid:0' shape=(?, 14) dtype=float32>



However this is not a scalar (aka rank-0) tensor because it has axes. To transform this tensor into a scalar we can slice it like this:


```python
y = y[0]
y
```




    <tf.Tensor 'strided_slice:0' shape=(14,) dtype=float32>



It is still *not* a scalar tensor so we will have to slice it again:


```python
y = y[0]
y
```




    <tf.Tensor 'strided_slice_1:0' shape=() dtype=float32>



Now it is a scalar tensor!

The above slicing could be done in a single statement like this:

```python
y = y[0,0]
```

But the explicit version of it was shown for visibility purposes.

The first argument required by `gradients(..)` function is the loss, which we will like to get the gradient of, and the second is a list of parameters to compute the gradient with respect to. Since we are interested in getting the gradient of the output of the model with respect to the output of the last convolutional layer we need to specify the layer as we did  in the previous notebook:


```python
# Save the desired layer in a variable
layer = model.get_layer("conv5_block16_concat")

# Compute gradient of model's output with respect to last conv layer's output
gradients = K.gradients(y, layer.output)

# Print gradients list
gradients
```




    [<tf.Tensor 'gradients/AddN:0' shape=(?, ?, ?, 1024) dtype=float32>]



Notice that the gradients function returns a list of placeholder tensors. To get the actual placeholder we will get the first element of this list:


```python
# Get first (and only) element in the list
gradients = gradients[0]

# Print tensor placeholder
gradients
```




    <tf.Tensor 'gradients/AddN:0' shape=(?, ?, ?, 1024) dtype=float32>



As with the activations of the last convolutional layer in the previous notebook, we still need a function that uses this placeholder to compute the actual values for an input image. This can be done in the same manner as before. Remember this **function expects its arguments as lists or tuples**:


```python
# Instantiate the function to compute the gradients
gradients_function = K.function([model.input], [gradients])

# Print the gradients function
gradients_function
```




    <keras.backend.tensorflow_backend.Function at 0x7fae8f3c1a58>



Now that we have the function for computing the gradients, let's test it out on a particular image. Don't worry about the code to load the image, this has been taken care of for you, you should only care that an image ready to be processed will be saved in the x variable:


```python
# Load dataframe that contains information about the dataset of images
df = pd.read_csv("nih_new/train-small.csv")

# Path to the actual image
im_path = 'nih_new/images-small/00000599_000.png'

# Load the image and save it to a variable
x = load_image(im_path, df, preprocess=False)

# Display the image
plt.imshow(x, cmap = 'gray')
plt.show()
```


![png](output_19_0.png)


We should normalize this image before going forward, this has also been taken care of:


```python
# Calculate mean and standard deviation of a batch of images
mean, std = get_mean_std_per_batch(df)

# Normalize image
x = load_image_normalize(im_path, mean, std)
```

Now we have everything we need to compute the actual values of the gradients. In this case we should also **provide the input as a list or tuple**:


```python
# Run the function on the image and save it in a variable
actual_gradients = gradients_function([x])
actual_gradients[0].shape
```




    (1, 10, 10, 1024)



An important intermediary step is to trim the batch dimension which can be done like this:


```python
# Remove batch dimension
actual_gradients = actual_gradients[0][0, :]
```


```python
# Print shape of the gradients array
print(f"Gradients of model's output with respect to output of last convolutional layer have shape: {actual_gradients.shape}")

# Print gradients array
actual_gradients
```

    Gradients of model's output with respect to output of last convolutional layer have shape: (10, 10, 1024)





    array([[[-1.3653503e-09,  2.7508462e-09,  3.2236332e-07, ...,
              9.0012662e-05, -6.0246934e-05,  6.2774081e-05],
            [-1.3653503e-09,  2.7508462e-09,  3.2236332e-07, ...,
              9.0012662e-05, -6.0246934e-05,  6.2774081e-05],
            [-1.3653503e-09,  2.7508462e-09,  3.2236332e-07, ...,
              9.0012662e-05, -6.0246934e-05,  6.2774081e-05],
            ...,
            [-1.3653503e-09,  2.7508462e-09,  3.2236332e-07, ...,
              9.0012662e-05, -6.0246934e-05,  6.2774081e-05],
            [-1.3653503e-09,  2.7508462e-09,  3.2236332e-07, ...,
              9.0012662e-05, -6.0246934e-05,  6.2774081e-05],
            [-1.3653503e-09,  2.7508462e-09,  3.2236332e-07, ...,
              9.0012662e-05, -6.0246934e-05,  6.2774081e-05]],
    
           [[-1.3653503e-09,  2.7508462e-09,  3.2236332e-07, ...,
              9.0012662e-05, -6.0246934e-05,  6.2774081e-05],
            [-1.3653503e-09,  2.7508462e-09,  3.2236332e-07, ...,
              9.0012662e-05, -6.0246934e-05,  6.2774081e-05],
            [-1.3653503e-09,  2.7508462e-09,  3.2236332e-07, ...,
              9.0012662e-05, -6.0246934e-05,  6.2774081e-05],
            ...,
            [-1.3653503e-09,  2.7508462e-09,  3.2236332e-07, ...,
              9.0012662e-05, -6.0246934e-05,  6.2774081e-05],
            [-1.3653503e-09,  2.7508462e-09,  3.2236332e-07, ...,
              9.0012662e-05, -6.0246934e-05,  6.2774081e-05],
            [-1.3653503e-09,  2.7508462e-09,  3.2236332e-07, ...,
              9.0012662e-05, -6.0246934e-05,  6.2774081e-05]],
    
           [[-1.3653503e-09,  2.7508462e-09,  3.2236332e-07, ...,
              9.0012662e-05, -6.0246934e-05,  6.2774081e-05],
            [-1.3653503e-09,  2.7508462e-09,  3.2236332e-07, ...,
              9.0012662e-05, -6.0246934e-05,  6.2774081e-05],
            [-1.3653503e-09,  2.7508462e-09,  3.2236332e-07, ...,
              9.0012662e-05, -6.0246934e-05,  6.2774081e-05],
            ...,
            [-1.3653503e-09,  2.7508462e-09,  3.2236332e-07, ...,
              9.0012662e-05, -6.0246934e-05,  6.2774081e-05],
            [-1.3653503e-09,  2.7508462e-09,  3.2236332e-07, ...,
              9.0012662e-05, -6.0246934e-05,  6.2774081e-05],
            [-1.3653503e-09,  2.7508462e-09,  3.2236332e-07, ...,
              9.0012662e-05, -6.0246934e-05,  6.2774081e-05]],
    
           ...,
    
           [[-1.3653503e-09,  2.7508462e-09,  3.2236332e-07, ...,
              9.0012662e-05, -6.0246934e-05,  6.2774081e-05],
            [-1.3653503e-09,  2.7508462e-09,  3.2236332e-07, ...,
              9.0012662e-05, -6.0246934e-05,  6.2774081e-05],
            [-1.3653503e-09,  2.7508462e-09,  3.2236332e-07, ...,
              9.0012662e-05, -6.0246934e-05,  6.2774081e-05],
            ...,
            [-1.3653503e-09,  2.7508462e-09,  3.2236332e-07, ...,
              9.0012662e-05, -6.0246934e-05,  6.2774081e-05],
            [-1.3653503e-09,  2.7508462e-09,  3.2236332e-07, ...,
              9.0012662e-05, -6.0246934e-05,  6.2774081e-05],
            [-1.3653503e-09,  2.7508462e-09,  3.2236332e-07, ...,
              9.0012662e-05, -6.0246934e-05,  6.2774081e-05]],
    
           [[-1.3653503e-09,  2.7508462e-09,  3.2236332e-07, ...,
              9.0012662e-05, -6.0246934e-05,  6.2774081e-05],
            [-1.3653503e-09,  2.7508462e-09,  3.2236332e-07, ...,
              9.0012662e-05, -6.0246934e-05,  6.2774081e-05],
            [-1.3653503e-09,  2.7508462e-09,  3.2236332e-07, ...,
              9.0012662e-05, -6.0246934e-05,  6.2774081e-05],
            ...,
            [-1.3653503e-09,  2.7508462e-09,  3.2236332e-07, ...,
              9.0012662e-05, -6.0246934e-05,  6.2774081e-05],
            [-1.3653503e-09,  2.7508462e-09,  3.2236332e-07, ...,
              9.0012662e-05, -6.0246934e-05,  6.2774081e-05],
            [-1.3653503e-09,  2.7508462e-09,  3.2236332e-07, ...,
              9.0012662e-05, -6.0246934e-05,  6.2774081e-05]],
    
           [[-1.3653503e-09,  2.7508462e-09,  3.2236332e-07, ...,
              9.0012662e-05, -6.0246934e-05,  6.2774081e-05],
            [-1.3653503e-09,  2.7508462e-09,  3.2236332e-07, ...,
              9.0012662e-05, -6.0246934e-05,  6.2774081e-05],
            [-1.3653503e-09,  2.7508462e-09,  3.2236332e-07, ...,
              9.0012662e-05, -6.0246934e-05,  6.2774081e-05],
            ...,
            [-1.3653503e-09,  2.7508462e-09,  3.2236332e-07, ...,
              9.0012662e-05, -6.0246934e-05,  6.2774081e-05],
            [-1.3653503e-09,  2.7508462e-09,  3.2236332e-07, ...,
              9.0012662e-05, -6.0246934e-05,  6.2774081e-05],
            [-1.3653503e-09,  2.7508462e-09,  3.2236332e-07, ...,
              9.0012662e-05, -6.0246934e-05,  6.2774081e-05]]], dtype=float32)



Looks like everything worked out nicely! You will still have to wait for the assignment to see how these elements are used by Grad-CAM to get visual interpretations. Before you go you should know that there is a shortcut for these calculations by getting both elements from a single Keras function:


```python
# Save multi-input Keras function in a variable
activations_and_gradients_function = K.function([model.input], [layer.output, gradients])

# Run the function on our image
act_x, grad_x = activations_and_gradients_function([x])

# Remove batch dimension for both arrays
act_x = act_x[0, :]
grad_x = grad_x[0, :]
```


```python
# Print actual activations
print(act_x)

# Print actual gradients
print(grad_x)
```

    [[[-0.28619292  0.10541391 -0.0182221  ...  0.1858328  -0.07475314
        0.2012677 ]
      [-0.402128   -0.45972198 -0.82684684 ...  0.2751884  -0.09238511
        0.30270433]
      [-0.29693234 -0.2154413  -0.82008535 ...  0.23475787 -0.08083473
        0.2509424 ]
      ...
      [-0.42700604 -0.27124473 -0.8040217  ...  0.12396325 -0.09499058
        0.16460732]
      [-0.28059694 -0.02322872 -0.4434362  ...  0.26165926 -0.10883205
        0.30085516]
      [-0.26109478  0.26782578 -0.04980515 ...  0.13395041 -0.06601515
        0.19848911]]
    
     [[-0.41598275  0.0965002  -0.43999797 ...  0.24505067 -0.11672373
        0.2949427 ]
      [-0.17238405 -0.3812187  -0.25798112 ...  0.42613187 -0.15315393
        0.45866844]
      [-0.45102215 -0.37757605 -0.4010303  ...  0.26384658 -0.1179072
        0.30986133]
      ...
      [-0.36496073 -0.55866694 -0.5286934  ...  0.13526711 -0.12218544
        0.18795796]
      [-0.56879324 -0.9186896  -0.4996615  ...  0.29658908 -0.12643978
        0.30119514]
      [-0.5947077  -0.18425797  0.02655283 ...  0.16888052 -0.0630455
        0.19739102]]
    
     [[-0.5062514  -0.04932767 -0.34279165 ...  0.25473955 -0.10849679
        0.23219462]
      [-1.2404321  -1.0552493  -0.31671602 ...  0.42680585 -0.14029567
        0.34258664]
      [-0.9068961  -0.55807316 -0.50709426 ...  0.27965742 -0.09860526
        0.20259362]
      ...
      [-1.4605086   0.11721444  0.28143242 ...  0.0613636  -0.0755816
        0.10166696]
      [-0.94331896 -0.08981708  1.1539317  ...  0.17562936 -0.05715244
        0.14061211]
      [-0.816543   -0.24505556  1.0251461  ...  0.07565136 -0.02606466
        0.07648329]]
    
     ...
    
     [[-0.6208235   0.06260616  0.17382732 ...  0.37805706 -0.16250384
        0.5095709 ]
      [-0.42985028  0.70034695 -0.07299207 ...  0.43666208 -0.14708628
        0.7921581 ]
      [-2.0553129  -1.2962095  -0.75738883 ...  0.23869985  0.15986013
        0.9693491 ]
      ...
      [-2.111959   -0.9664068  -0.5826868  ...  0.07937228  0.05796714
        1.3913963 ]
      [-1.6927792   0.04535086 -0.5417167  ...  0.19124544 -0.12741336
        1.4397111 ]
      [-0.9562766   0.07162583  0.28529865 ... -0.10208753  0.09063674
        1.020506  ]]
    
     [[-0.58046687  0.05047078 -0.44128972 ...  0.27323943 -0.15392973
        0.47636122]
      [-0.5297517   0.15784627 -0.462206   ...  0.4511362  -0.2635701
        0.6938297 ]
      [-0.4636375   0.38374984 -1.531445   ...  0.33798808 -0.16819087
        0.5761511 ]
      ...
      [-0.80381656 -0.31637245 -1.0143678  ...  0.09683872 -0.10915622
        0.7357286 ]
      [-0.30914816  0.3277402  -0.9110493  ...  0.28531307 -0.23644474
        0.85661936]
      [-0.53180254 -0.07997909 -0.59733284 ...  0.05930805 -0.10012133
        0.6781993 ]]
    
     [[-0.9340049   0.2298406   0.5758754  ...  0.25713286 -0.10626409
        0.23647718]
      [-1.0297579  -0.15503609  0.4392693  ...  0.35741362 -0.12632793
        0.33936253]
      [-0.73517406 -0.1194941  -0.19677076 ...  0.28121653 -0.09897768
        0.28874812]
      ...
      [-0.60039556  0.5045108   0.01430643 ...  0.17638573 -0.08689473
        0.23746102]
      [-0.65987945  0.24948944  0.30365634 ...  0.24895509 -0.11379367
        0.28807753]
      [-0.64840496  0.2491114   0.60081077 ...  0.18949631 -0.05899787
        0.24392612]]]
    [[[-1.3653503e-09  2.7508462e-09  3.2236332e-07 ...  9.0012662e-05
       -6.0246934e-05  6.2774081e-05]
      [-1.3653503e-09  2.7508462e-09  3.2236332e-07 ...  9.0012662e-05
       -6.0246934e-05  6.2774081e-05]
      [-1.3653503e-09  2.7508462e-09  3.2236332e-07 ...  9.0012662e-05
       -6.0246934e-05  6.2774081e-05]
      ...
      [-1.3653503e-09  2.7508462e-09  3.2236332e-07 ...  9.0012662e-05
       -6.0246934e-05  6.2774081e-05]
      [-1.3653503e-09  2.7508462e-09  3.2236332e-07 ...  9.0012662e-05
       -6.0246934e-05  6.2774081e-05]
      [-1.3653503e-09  2.7508462e-09  3.2236332e-07 ...  9.0012662e-05
       -6.0246934e-05  6.2774081e-05]]
    
     [[-1.3653503e-09  2.7508462e-09  3.2236332e-07 ...  9.0012662e-05
       -6.0246934e-05  6.2774081e-05]
      [-1.3653503e-09  2.7508462e-09  3.2236332e-07 ...  9.0012662e-05
       -6.0246934e-05  6.2774081e-05]
      [-1.3653503e-09  2.7508462e-09  3.2236332e-07 ...  9.0012662e-05
       -6.0246934e-05  6.2774081e-05]
      ...
      [-1.3653503e-09  2.7508462e-09  3.2236332e-07 ...  9.0012662e-05
       -6.0246934e-05  6.2774081e-05]
      [-1.3653503e-09  2.7508462e-09  3.2236332e-07 ...  9.0012662e-05
       -6.0246934e-05  6.2774081e-05]
      [-1.3653503e-09  2.7508462e-09  3.2236332e-07 ...  9.0012662e-05
       -6.0246934e-05  6.2774081e-05]]
    
     [[-1.3653503e-09  2.7508462e-09  3.2236332e-07 ...  9.0012662e-05
       -6.0246934e-05  6.2774081e-05]
      [-1.3653503e-09  2.7508462e-09  3.2236332e-07 ...  9.0012662e-05
       -6.0246934e-05  6.2774081e-05]
      [-1.3653503e-09  2.7508462e-09  3.2236332e-07 ...  9.0012662e-05
       -6.0246934e-05  6.2774081e-05]
      ...
      [-1.3653503e-09  2.7508462e-09  3.2236332e-07 ...  9.0012662e-05
       -6.0246934e-05  6.2774081e-05]
      [-1.3653503e-09  2.7508462e-09  3.2236332e-07 ...  9.0012662e-05
       -6.0246934e-05  6.2774081e-05]
      [-1.3653503e-09  2.7508462e-09  3.2236332e-07 ...  9.0012662e-05
       -6.0246934e-05  6.2774081e-05]]
    
     ...
    
     [[-1.3653503e-09  2.7508462e-09  3.2236332e-07 ...  9.0012662e-05
       -6.0246934e-05  6.2774081e-05]
      [-1.3653503e-09  2.7508462e-09  3.2236332e-07 ...  9.0012662e-05
       -6.0246934e-05  6.2774081e-05]
      [-1.3653503e-09  2.7508462e-09  3.2236332e-07 ...  9.0012662e-05
       -6.0246934e-05  6.2774081e-05]
      ...
      [-1.3653503e-09  2.7508462e-09  3.2236332e-07 ...  9.0012662e-05
       -6.0246934e-05  6.2774081e-05]
      [-1.3653503e-09  2.7508462e-09  3.2236332e-07 ...  9.0012662e-05
       -6.0246934e-05  6.2774081e-05]
      [-1.3653503e-09  2.7508462e-09  3.2236332e-07 ...  9.0012662e-05
       -6.0246934e-05  6.2774081e-05]]
    
     [[-1.3653503e-09  2.7508462e-09  3.2236332e-07 ...  9.0012662e-05
       -6.0246934e-05  6.2774081e-05]
      [-1.3653503e-09  2.7508462e-09  3.2236332e-07 ...  9.0012662e-05
       -6.0246934e-05  6.2774081e-05]
      [-1.3653503e-09  2.7508462e-09  3.2236332e-07 ...  9.0012662e-05
       -6.0246934e-05  6.2774081e-05]
      ...
      [-1.3653503e-09  2.7508462e-09  3.2236332e-07 ...  9.0012662e-05
       -6.0246934e-05  6.2774081e-05]
      [-1.3653503e-09  2.7508462e-09  3.2236332e-07 ...  9.0012662e-05
       -6.0246934e-05  6.2774081e-05]
      [-1.3653503e-09  2.7508462e-09  3.2236332e-07 ...  9.0012662e-05
       -6.0246934e-05  6.2774081e-05]]
    
     [[-1.3653503e-09  2.7508462e-09  3.2236332e-07 ...  9.0012662e-05
       -6.0246934e-05  6.2774081e-05]
      [-1.3653503e-09  2.7508462e-09  3.2236332e-07 ...  9.0012662e-05
       -6.0246934e-05  6.2774081e-05]
      [-1.3653503e-09  2.7508462e-09  3.2236332e-07 ...  9.0012662e-05
       -6.0246934e-05  6.2774081e-05]
      ...
      [-1.3653503e-09  2.7508462e-09  3.2236332e-07 ...  9.0012662e-05
       -6.0246934e-05  6.2774081e-05]
      [-1.3653503e-09  2.7508462e-09  3.2236332e-07 ...  9.0012662e-05
       -6.0246934e-05  6.2774081e-05]
      [-1.3653503e-09  2.7508462e-09  3.2236332e-07 ...  9.0012662e-05
       -6.0246934e-05  6.2774081e-05]]]


**Congratulations on finishing this lecture notebook!** Hopefully you will now have a better understanding of how to leverage Keras's API power for computing gradients. Keep it up!
