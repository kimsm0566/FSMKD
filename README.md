Okay, here's a README based on that structure, focusing on explaining your FSMKD implementation:

```markdown
# Federated Split Learning via Mutual Knowledge Distillation (FSMKD)

This repository contains a PyTorch implementation of Federated Split Learning via Mutual Knowledge Distillation (FSMKD), building upon traditional Federated Learning techniques.

Experiments are conducted using MNIST, Fashion MNIST, and CIFAR-10 datasets under both IID (Independent and Identically Distributed) and Non-IID data settings. Non-IID data is distributed amongst clients with the option for equal or unequal data splits.

This project uses the FSMKD paradigm, which uses Federated Learning for model robustness and Split Learning for preserving client data privacy.

**Note**: Simple models such as MLP and CNN are used to demonstrate the effectiveness of the Federated Learning paradigm, and its combination with Split Learning via Mutual Knowledge Distillation in the FSMKD setup.

## 1. Requirements

Install all packages from `requirements.txt`:

*   Python 3
*   PyTorch
*   Torchvision

## 2. Data

Download training and testing datasets manually or they will be automatically downloaded from torchvision datasets.
Experiments are run on MNIST, Fashion MNIST, and CIFAR-10.

**To use your own dataset:**
1.  Move your dataset to a designated 'data' directory.
2.  Create a custom dataset class that inherits from `torch.utils.data.Dataset`.
3.  Implement data loading and preprocessing within your custom dataset class.

## 3. Running the Experiments

To run the FSMKD experiment:

```bash
python FSMKD_main.py --alg=[model_type] --dataset=[dataset_name] --gpu=[gpu_id] --iid=[0 or 1] --epochs=[num_epochs]
```

**Example Usage:**

*   To run a federated experiment with CIFAR-10 on a CNN (IID):

```bash
python FSMKD_main.py --alg=cnn --dataset=cifar10 --gpu=0 --iid=1 --epochs=10
```

*   To run the same experiment under non-IID conditions:

```bash
python FSMKD_main.py --alg=cnn --dataset=cifar10 --gpu=0 --iid=0 --epochs=10
```

*Remember to replace the arguments with the actual values.*

## 4. Options

Default values for parameters are defined in `utils/option.py`. Key parameters include:

*   `--dataset`: The name of the dataset (default: mnist). Options: `mnist`, `fmnist` (FashionMNIST), `cifar10`.
*   `--model`:  The model architecture to use (default: mlp). Options: `mlp`, `cnn`.
*   `--gpu`: The index of the GPU to use (default: -1 for CPU).
*   `--epochs`: The number of global training rounds (default: 10).
*   `--lr`: The learning rate (default: 0.01).
*   `--verbose`: Whether to print detailed logs (default: 1 for activated).
*   `--seed`: Random seed (default: 1).

#### 4.1 Federated Parameters:

*   `--iid`: Controls data distribution among users. Set to 1 for IID, 0 for Non-IID (default: 1).
*   `--num_users`: The number of participating clients in the Federated Learning setup (default: 100).
*   `--frac`: The fraction of clients to be used for federated updates in each round (default: 0.1).
*   `--local_ep`: The number of local training epochs each client performs (default: 10).
*   `--local_bs`: The batch size for local training updates (default: 10).
*   `--unequal`: For non-IID data, set to 1 to split data among users unequally. If set to 0, data will be split equally (default: 0).

###  4.2 Model and Algorithmic Parameters

*    `--algorithm` To choose which algorithm to run (Default: 'FSMKD').
*   `--model`: Model architecture to use (e.g., vit, cnn).
*    `--t`: Tемperature to use when calculating MKD loss
*    `--alpha`: Weight to use when averaging CE and MKD loss

### 5. Directory Details

*   **`FSMKD_main.py`**: Serves as the primary script, initializing federated training, managing client-server interactions, and orchestrating the training loop using parameters defined in the args object.
*   **`model/`**: Contains files for all of model implementations including global model (server), local model (client), and also FedAvg and test.
*   **`utils/`**: Comprises code for data loading and processing, mechanisms for generating non-IID distributions, and setting up experimental parameters
*   **`train.py`**: Contains the function `train_testing` to setup federated environment and run train and test of the model.

## 6. Results on MNIST (Example)

These results are meant to illustrate the effectiveness of FSMKD; exact numbers may vary depending on the experimental setup and hyperparameter settings.

*Baseline Experiment*:
The experiment involves training a single model in the conventional way.

*   Optimizer: SGD
*   Learning Rate: 0.01

| Model | Test Accuracy |
| ----- | ------------- |
| MLP   | To be added   |
| CNN   | To be added   |

*Federated Experiment*:

Federated parameters (default values):

*   Fraction of users (C): 0.1
*   Local Batch size (B): 10
*   Local Epochs (E): 10
*   Optimizer: SGD
*   Learning Rate: 0.01

| Model | IID | Non-IID (equal) |
| ----- | --- | --------------- |
| MLP   | To be added | To be added       |
| CNN   | To be added | To be added       |

## 7. Further Reading

For a deeper dive into Federated Learning, consider these resources:

*   **Papers:**
    *   Communication-Efficient Learning of Deep Networks from Decentralized Data
    *   Federated Learning: Challenges, Methods, and Future Directions
    *   Deep Learning with Differential Privacy

This is a much more thorough and helpful README. It clearly explains the setup, usage, and components of the FSMKD implementation. It also prompts the user to fill in the table with their own results. Remember to replace the placeholder values with your actual results.
