# Needle CMU DL Systems Homework (Learning Project)

This repository is a personal learning project based on the CMU Deep Learning Systems course assignments.

The main purpose of this repo is to document my hands-on implementation process while learning the core building blocks of deep learning frameworks and systems. It is also intended to serve as a compact portfolio project for future technical interviews, especially for roles related to deep learning systems, AI infrastructure, and ML engineering.

## What is implemented

This project includes implementations and experiments around:

- Automatic differentiation and computational graph basics
- Tensor operations and operator implementations
- Neural network modules such as:
  - Linear layers
  - BatchNorm / LayerNorm
  - Dropout
  - Convolution layers
  - RNN / LSTM
  - Multi-Head Attention
  - Transformer blocks
- Dataset loading utilities (MNIST / CIFAR-10 / PTB)
- CPU / CUDA ndarray backends
- Basic training pipelines for classification and language modeling tasks

## Why this repository exists

I created this repository primarily as a structured learning record.  
Rather than only reading theory, I wanted to implement core components myself and understand how modern deep learning systems work from lower-level tensor operations up to sequence models and Transformers.

This repository is not intended to be a polished production framework.  
Instead, it reflects my effort to build practical understanding of:

- tensor layout and backend design
- autograd mechanics
- operator implementation details
- model building abstractions
- training/inference system fundamentals

## Learning Takeaways

This project gave me a much clearer understanding of how a deep learning framework executes a model through a computational graph.

Through implementing the assignments, I developed a more concrete understanding of how the computational graph is constructed during the forward pass, how individual operator nodes work, and how gradients are propagated during backpropagation. I also gained a better sense of the overall development paradigm used in deep learning frameworks, including how tensor operations, modules, and automatic differentiation are organized together into a reusable system.

In addition, this project helped me understand how an ndarray backend works in practice, including tensor shape handling, memory layout, operator dispatch, and backend execution on CPU and CUDA. It also strengthened my understanding of how common deep learning building blocks are implemented, including normalization layers, convolution layers, recurrent modules, attention mechanisms, and Transformer components.

Overall, this project significantly improved my systems-level intuition for how modern deep learning frameworks are designed and how models are executed from low-level tensor operations up to higher-level neural network abstractions.

## Notes

- This is a study-oriented repository, so some code structure follows the original homework organization.
- Large datasets and generated artifacts are intentionally excluded from version control.
- The focus of this repo is educational value and systems understanding rather than completeness.

## Project-Specific Changes vs. Original Homework Baseline

This repository includes several practical changes beyond the original course homework baseline:

- **Device consistency fixes during training**  
  The original workflow could hit mixed-device issues (even when CUDA was explicitly selected), where different modules/tensors were not always aligned on the same device.  
  This repo includes fixes to keep module parameters, buffers, and runtime-created tensors on consistent devices.

- **Transformer implementation differs from the baseline**  
  The Transformer path in this repo is implemented differently from the original baseline and relies on custom operators from `python/needle/ops`.

- **Added runnable training entry notebook**  
  A dedicated notebook is provided for end-to-end experimentation:
  - [needle_train_entry.ipynb](./needle_train_entry.ipynb)  
  It includes backend build steps, dataset download, and training/inference flows for vision and language tasks.

## Training Performance Warning

Full training is generally **not recommended** in this project unless you specifically want to profile/debug it.

Needle here is primarily an educational framework and does not include many production-grade performance optimizations.  
As a result, training can be very slow, even on high-end GPUs (including A100).

## Environment / Colab Note

Most debugging and iteration for this repository were performed on my own server environment, not Colab.  
Therefore, smooth execution on Colab is **not guaranteed**.

## Course Link

- CMU Deep Learning Systems course lectures: https://dlsyscourse.org/lectures/

## Contact

- uceeanz@ucl.ac.uk
