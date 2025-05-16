# Project Report: Evaluating Attention Mechanisms in CNNs on Fashion MNIST

## 1. Introduction

The rapid evolution of deep learning has led to increasingly complex architectures capable of modeling intricate patterns in data. One such innovation in architectural design is the use of attention mechanisms, which allow models to dynamically focus on the most informative parts of the input. While transformers and attention-based models dominate large-scale vision tasks, lightweight attention modules such as Squeeze-and-Excitation (SE) and Convolutional Block Attention Module (CBAM) offer a more efficient alternative for enhancing Convolutional Neural Networks (CNNs) without incurring significant computational overhead.

In this project, we explore the impact of incorporating SE and CBAM modules into a ResNet-18 backbone, trained and evaluated on the Fashion MNIST dataset. The goal is to understand how these attention mechanisms influence model performance, parameter efficiency, and computational cost in a lightweight classification scenario.

## 2. Dataset: Fashion MNIST

Fashion MNIST is a widely used benchmark dataset that serves as a direct drop-in replacement for the traditional MNIST dataset. It consists of:

* 60,000 training images and 10,000 test images
* Grayscale images of size 28x28
* 10 classes representing fashion categories: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

The dataset is preprocessed using the following PyTorch transformation pipeline:

```python
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
```

We perform an 90/10 split of the training set to create validation data for early stopping and model selection.

## 3. Model Architecture

### 3.1 Baseline: Adapted ResNet-18

ResNet-18 is a classic residual network architecture that leverages skip connections to enable deeper networks to train effectively without vanishing gradients. Since Fashion MNIST consists of small grayscale images, we adapt ResNet-18 as follows:

* Modify the first convolution layer to accept 1-channel input
* Remove the initial max-pooling layer to preserve spatial resolution
* Modify the final fully connected (fc) layer to output 10 classes

```python
model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = nn.Identity()
model.fc = nn.Linear(model.fc.in_features, 10)
```

This serves as the baseline architecture against which attention-based variants are compared.

### 3.2 SE-ResNet-18: Squeeze-and-Excitation

The SE block introduces channel-wise attention by modeling inter-channel dependencies. It consists of:

* **Squeeze**: Global average pooling to summarize each channel
* **Excitation**: Two fully connected layers with a bottleneck structure and a sigmoid activation to reweight channel importance

The SE module is embedded into each residual block:

```python
class SEBlock(nn.Module):
    def forward(self, x):
        y = self.squeeze(x).view(b, c)
        y = self.excite(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
```

### 3.3 CBAM-ResNet-18: Convolutional Block Attention Module

CBAM extends SE by incorporating spatial attention in addition to channel attention. It works in two sequential steps:

* **Channel Attention (CA)**: Reweights channels based on max/avg-pooled descriptors
* **Spatial Attention (SA)**: Applies attention maps to spatial dimensions by combining channel-wise max and avg pooling

We evaluate two CBAM variants:

* **CBAM-in-blocks**: Inserts a full CBAM block into each residual unit
* **CBAM-before-classifier**: Applies CBAM globally after the final pooling layer, before classification

## 4. Training Configuration

The models are trained using the following hyperparameters:

* Optimizer: Adam
* Learning Rate Scheduler: CosineAnnealingLR
* Loss Function: CrossEntropyLoss
* Epochs: 50 (with early stopping if validation accuracy does not improve for 10 epochs)
* Batch Size: 64
* Hardware: NVIDIA RTX 4070 GPU

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

TensorBoard logging was used for monitoring accuracy and loss during training.

## 5. Evaluation Metrics

Each model was evaluated along the following dimensions:

* **Test Accuracy (%):** Final classification accuracy on the test set
* **Parameter Count (M):** Number of trainable parameters in millions
* **FLOPs (G):** Estimated number of floating point operations per forward pass
* **Inference Time (ms/img):** Average time to process a single image (batch size = 64)

These metrics enable a balanced trade-off analysis between predictive performance and computational cost.

## 6. Results

| Model            | Test Acc (%) | Params (M) | FLOPs (G) | Infer Time (ms/img) |
| ---------------- | ------------ | ---------- | --------- | ------------------- |
| baseline         | 93.27        | 11.17      | 0.92      | 0.06                |
| cbam\_classifier | 93.18        | 11.21      | 0.92      | 0.07                |
| se               | 92.70        | 11.26      | 0.92      | 0.07                |
| cbam\_blocks     | 92.59        | 11.26      | 0.92      | 0.09                |

### Visualizations

#### Test Accuracy

![image](https://github.com/user-attachments/assets/3129383f-2124-4cf6-9f56-568f826a8c17)

#### Inference Time

![image](https://github.com/user-attachments/assets/ecace118-7c76-4b07-aac4-45bee299067c)

#### Accuracy vs. Parameter Count

![image](https://github.com/user-attachments/assets/5158c562-1540-4c4a-af70-6b04bce1c32d)

## 7. Analysis and Interpretation

* **Baseline ResNet-18** achieved the highest test accuracy (93.27%) while being the most efficient in inference speed.
* **SE and CBAM modules**, although theoretically powerful, did not yield significant accuracy improvements on this simple grayscale dataset. In fact, adding CBAM in every block led to a measurable increase in inference time with a small drop in accuracy.
* **CBAM-before-classifier** offers a reasonable trade-off between overhead and accuracy but still doesn't outperform the baseline.
* **Parameter and FLOP counts remain largely unchanged** across all variants, indicating that the attention blocks are lightweight, yet their impact on performance in simple datasets is limited.

These results suggest that attention may yield more benefits in datasets with higher visual complexity or in settings where interpretability or spatial feature emphasis is more critical.

## 8. Conclusion

This project provides a structured benchmarking of lightweight attention mechanisms in CNNs using the Fashion MNIST dataset. The evaluation reveals that:

* ResNet-18 already performs well without attention on small, clean datasets
* SE and CBAM slightly increase computational overhead without significant accuracy gains
* Attention may be more valuable on complex datasets or when interpretability is essential

The project is modular and extensible:

* Can be scaled to CIFAR-100, Tiny ImageNet, or other richer datasets
* Supports swapping in newer attention modules (e.g., ECA, SKNet, Transformer layers)
* Can integrate Grad-CAM or saliency visualization to analyze attention focus

This analysis demonstrates the importance of grounding architectural enhancements not just in accuracy metrics but also in practical deployment constraints such as speed and memory usage.

## 9. Future Work

* Incorporate Grad-CAM or attention map visualization to interpret model behavior
* Benchmark on more complex image datasets to observe if attention yields larger benefits
* Explore mixed-precision training for additional speedup
* Combine SE/CBAM with advanced augmentation strategies like CutMix or RandAugment
