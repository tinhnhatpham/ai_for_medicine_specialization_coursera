
# Extract a Sub Section

In this week's assignment you will be extracting sub-sections of the MRI data to train your network. The reason for this is that training on a full MRI scan would be too memory intensive to be practical. To extract a sub-section in the assignment, you will need to write a function to isolate a small "cube" of the data for training. This example is meant to show you how to do such an extraction for 1D arrays. In the assignment you will apply the same logic in 3D.


```python
import numpy as np
import keras
import pandas as pd
```

    Using TensorFlow backend.



```python
# Define a simple one dimensional "image" to extract from
image = np.array([10,11,12,13,14,15])
image
```




    array([10, 11, 12, 13, 14, 15])




```python
# Compute the dimensions of your "image"
image_length = image.shape[0]
image_length
```




    6



## 1. Sub Sections
In the assignment, you will define a "patch size" in three dimensions, that will be the size of the sub-section you want to extract. For this exercise, you only need to define a patch size in one dimension.


```python
# Define a patch length, which will be the size of your extracted sub-section
patch_length = 3
```

To extract a patch of length `patch_length` you will first define an index at which to start the patch.

Run the next cell to define your start index


```python
# Define your start index
start_i = 0
```

At the end of the next cell you are adding 1 to the start index. Run cell a few times to extract some one dimensional sub-sections from your "image"

What happens when you run into the edge of the image (when `start_index` is > 3)?


```python
# Define an end index given your start index and patch size
print(f"start index {start_i}")
end_i = start_i + patch_length
print(f"end index {end_i}")

# Extract a sub-section from your "image"
sub_section = image[start_i: end_i]
print("output patch length: ", len(sub_section))
print("output patch array: ", sub_section)

# Add one to your start index
start_i +=1
```

    start index 7
    end index 10
    output patch length:  0
    output patch array:  []


You'll notice when you run the above multiple times, that eventually the sub-section returned is no longer of length `patch_length`. 

In the assignment, your neural network will be expecting a particular sub-section size and will not accept inputs of other dimensions. For the start indices, you will be randomly choosing values and you need to ensure that your random number generator is set up to avoid the edges of your image object.

The next few code cells include a demonstration of how you could determine the constraints on your start index for the simple one dimensional example.


```python
# Set your start index to 3 to extract a valid patch
start_i = 3
print(f"start index {start_i}")
end_i = start_i + patch_length
print(f"end index {end_i}")
sub_section = image[start_i: end_i]
print("output patch array: ", sub_section)
```

    start index 3
    end index 6
    output patch array:  [13 14 15]



```python
# Compute and print the largest valid value for start index
print(f"The largest start index for which "
      f"a sub section is still valid is "
      f"{image_length - patch_length}")
```

    The largest start index for which a sub section is still valid is 3



```python
# Compute and print the range of valid start indices
print(f"The range of valid start indices is:")

# Compute valid start indices, note the range() function excludes the upper bound
valid_start_i = [i for i in range(image_length - patch_length + 1)]
print(valid_start_i)
```

    The range of valid start indices is:
    [0, 1, 2, 3]


### 1.1 Random Selection of Start Indices
In the assignment, you will need to randomly select a valid integer for the start index in each of three dimensions. The way to do this is by following the logic above to identify valid start indices and then selecting randomly from that range of valid numbers.

Run the next cell to select a valid start index for the one dimensional example


```python
# Choose a random start index, note the np.random.randint() function excludes the upper bound.
start_i = np.random.randint(image_length - patch_length + 1)
print(f"randomly selected start index {start_i}")
```

    randomly selected start index 2



```python
# Randomly select multiple start indices in a loop
for _ in range(10):
    start_i = np.random.randint(image_length - patch_length + 1)
    print(f"randomly selected start index {start_i}")
```

    randomly selected start index 1
    randomly selected start index 2
    randomly selected start index 2
    randomly selected start index 3
    randomly selected start index 0
    randomly selected start index 2
    randomly selected start index 1
    randomly selected start index 1
    randomly selected start index 3
    randomly selected start index 3


## 2. Background Ratio

Another thing you will be doing in the assignment is to compute the ratio of background to edema and tumorous regions. You will be provided with a file containing labels with these categories:

* 0: background
* 1: edema
* 2: non-enhancing tumor
* 3: enhancing tumor

Let's try to demonstrate this in 1-D to get some intuition on how to implement it in 3D later in the assignment.


```python
# We first simulate input data by defining a random patch of length 16. This will contain labels 
# with the categories (0 to 3) as defined above.

patch_labels = np.random.randint(0, 4, (16))
print(patch_labels)
```

    [0 2 3 2 1 3 2 1 1 2 2 1 2 3 3 2]



```python
# A straightforward approach to get the background ratio is
# to count the number of 0's and divide by the patch length

bgrd_ratio = np.count_nonzero(patch_labels == 0) / len(patch_labels)
print("using np.count_nonzero(): ", bgrd_ratio)

bgrd_ratio = len(np.where(patch_labels == 0)[0]) / len(patch_labels)
print("using np.where(): ", bgrd_ratio)
```

    using np.count_nonzero():  0.0625
    using np.where():  0.0625



```python
# However, take note that we'll use our label array to train a neural network
# so we can opt to compute the ratio a bit later after we do some preprocessing. 
# First, we convert the label's categories into one-hot format so it can be used to train the model

patch_labels_one_hot = keras.utils.to_categorical(patch_labels, num_classes=4)
print(patch_labels_one_hot)
```

    [[1. 0. 0. 0.]
     [0. 0. 1. 0.]
     [0. 0. 0. 1.]
     [0. 0. 1. 0.]
     [0. 1. 0. 0.]
     [0. 0. 0. 1.]
     [0. 0. 1. 0.]
     [0. 1. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 1. 0.]
     [0. 0. 1. 0.]
     [0. 1. 0. 0.]
     [0. 0. 1. 0.]
     [0. 0. 0. 1.]
     [0. 0. 0. 1.]
     [0. 0. 1. 0.]]


**Note**: We hardcoded the number of classes to 4 in our simple example above.
In the assignment, you should take into account that the label file can have
a different number of categories


```python
# Let's convert the output to a dataframe just so we can see the labels more clearly

pd.DataFrame(patch_labels_one_hot, columns=['background', 'edema', 'non-enhancing tumor', 'enhancing tumor'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>background</th>
      <th>edema</th>
      <th>non-enhancing tumor</th>
      <th>enhancing tumor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# What we're interested in is the first column because that 
# indicates if the element is part of the background
# In this case, 1 = background, 0 = non-background

print("background column: ", patch_labels_one_hot[:,0])
```

    background column:  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]



```python
# we can compute the background ratio by counting the number of 1's 
# in the said column divided by the length of the patch

bgrd_ratio = np.sum(patch_labels_one_hot[:,0])/ len(patch_labels)
print("using one-hot column: ", bgrd_ratio)
```

    using one-hot column:  0.0625


#### That's all for this lab, now you have the basic tools you need for sub-section extraction in this week's graded assignment!
