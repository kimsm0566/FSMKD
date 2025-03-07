### **FSMKD Implementation**

---

## **Overview**
This repository contains the implementation of **FSMKD (Federated Split Learning via Mutual Knowledge Distillation)**, a novel FL-SL synergy framework that combines **Federated Learning (FL)** and **Split Learning (SL)** in a two-way manner using **Deep Mutual Knowledge Distillation**. FSMKD enables clients to train personalized local models while collaborating with the server to improve the global model's performance.

---

## **Dependencies**
The code requires the following dependencies:

- **Python**: >= 3.6
- **PyTorch**: >= 1.2.0
- **CUDA**: 11.8

To install the required Python packages, use the following command:

```bash
pip install -r requirements.txt
```

### **Required Packages**
The `requirements.txt` file includes:
- matplotlib==3.7.5
- scikit-learn==1.3.2
- pandas==2.0.3
- pillow==10.4.0
- numpy==1.24.4
- wheel==0.34.2
- setuptools==45.2.0

Additionally, PyTorch and CUDA-related packages can be installed using the following command:

```bash
pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 torchaudio==2.4.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

---

## **Data**
This implementation uses the following datasets:
1. **MNIST**
2. **Fashion MNIST**
3. **CIFAR10**

---

## **Usage**

FSMKD can be run using the following command:

```bash
python main.py --algorithm FSMKD --dataset [dataset] --num_users [num_users] --model [model] --epochs [epochs]
```

### **Explanation of Parameters (Updated Table)**

| Parameter          | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `algorithm`        | Algorithm to run (e.g., FSMKD).                                            |
| `dataset`          | Dataset name (`mnist`, `fashion_mnist`, `cifar10`)                         |
| `num_users`        | Number of clients/users participating in federated learning (`default=5`). |
| `model`            | Model architecture (`vgg`, `vit`).                                         |
| `num_classes`      | Total number of classes in the dataset (`default=10`).                     |
| `frac`             | Fraction of clients participating in each round (`default=1.0`).          |
| `lr`               | Learning rate for optimization (`default=0.001`).                         |
| `round`            | Total number of communication rounds (`default=200`).                     |
| `local_ep`         | Number of local epochs for training on each client (`default=1`).         |
| `gpu`              | GPU ID to use for training (`default=1`; set to -1 for CPU).              |


---

## **FSMKD Framework**

### Model Structure:
FSMKD introduces a two-body structure:
1. **Local Model**: Head → Personalized Local Body → Tail (trained locally on clients).
2. **Global Model**: Head → Shared Server Body → Tail (trained collaboratively on server).

### Training Process:
1. Each client trains its personalized local model using its private dataset.
2. The client sends intermediate smashed data from its local body to the server.
3. The server processes smashed data using its global body and returns processed features.
4. Both client-side and server-side models are trained with two loss functions:
   - Supervised loss (hard labels).
   - Mimicry loss (soft labels for knowledge distillation).

5. At the end of federated learning rounds, FedAvg is used to aggregate the weights of heads and tails across all clients.

---


## **Acknowledgements**
This implementation is based on the paper *"Federated Split Learning via Mutual Knowledge Distillation"* by Linjun Luo and Xinglin Zhang.

For more details, refer to the [paper][(https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/31641994/3cf59dec-806b-4734-813f-0ddd05e09ace/Federated_Split_Learning_via_Mutual_Knowledge_Distillation.pdf).](https://ieeexplore.ieee.org/document/10378869)

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/31641994/3cf59dec-806b-4734-813f-0ddd05e09ace/Federated_Split_Learning_via_Mutual_Knowledge_Distillation.pdf
