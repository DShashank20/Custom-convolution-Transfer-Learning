# Custom Convolution & Transfer Learning

**Name: Shashank Reddy Dasari **    

**ID:700781569**


---

## Overview

This project demonstrates two important deep learning concepts:

1. **Custom convolution using NumPy**
2. **Transfer learning using a pretrained ResNet18 model with PyTorch**

The notebook includes a manual implementation of convolution and experiments with feature extraction and fine-tuning using a pretrained CNN.



## Requirements

The notebook uses:

* NumPy
* Matplotlib
* PyTorch
* Torchvision

Install the required packages:

```bash
pip install numpy matplotlib torch torchvision
```

## Part 1: Custom Convolution

The first part implements a convolution operation from scratch using NumPy.

The input is a **5 × 5 matrix**, and the kernel/filter is a **3 × 3 matrix**.

### Convolution Parameters

* Input size: `5 × 5`
* Kernel size: `3 × 3`
* Stride: `1`
* Padding: `0`

The output size is calculated using:

```text
Output = ((Input - Filter + 2 × Padding) / Stride) + 1
```

Therefore:

```text
Output = ((5 - 3 + 0) / 1) + 1
       = 3 × 3
```

The filter is manually moved across the input matrix, and the dot product is calculated at each position to produce the output feature map.

### Changing the Stride

When the stride is changed from **1 to 2**, the filter moves two pixels at a time.

The output becomes:

```text
((5 - 3) / 2) + 1 = 2 × 2
```

Increasing the stride reduces the spatial size of the output feature map.

## Part 2: Transfer Learning

The second part demonstrates transfer learning using a pretrained **ResNet18** model from Torchvision.

A synthetic dataset is used for binary classification.

### Dataset Configuration

* Training samples: 200
* Validation samples: 50
* Image size: `3 × 64 × 64`
* Number of classes: 2
* Batch size: 32
* Training epochs: 5

## Experiment A: Feature Extraction

In this experiment, the pretrained convolutional layers are frozen.

Only the final fully connected layer is trained for the classification task.

```text
Trainable Parameters: 1,026
Training Time: 6.16 seconds
Accuracy: 52.0%
```

Freezing the pretrained layers reduces the number of parameters that need to be trained and generally requires less computational time.

## Experiment B: Fine-Tuning

In this experiment, the last convolutional block (`layer4`) is unfrozen along with the final fully connected layer.

```text
Trainable Parameters: 8,394,754
Training Time: 10.26 seconds
Accuracy: 42.0%
```

Fine-tuning allows part of the pretrained network to adapt to the new dataset, but it requires more computation.

## Results

| Method                   | Trainable Parameters | Training Time | Accuracy |
| ------------------------ | -------------------: | ------------: | -------: |
| Frozen Feature Extractor |                1,026 |        6.16 s |    52.0% |
| Fine-Tuned Network       |            8,394,754 |       10.26 s |    42.0% |

## Conclusion

This project demonstrates how convolution works at a basic level and how transfer learning can be applied using a pretrained CNN.

The custom convolution implementation shows how a filter moves across an input to produce a feature map. The transfer learning experiments demonstrate the difference between freezing pretrained layers and fine-tuning part of the network.

For this particular synthetic dataset, the frozen feature extractor achieved higher validation accuracy and required less training time.


