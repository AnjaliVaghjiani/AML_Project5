# Distributed Learning of Deep Neural Networks

**Exploring how to train neural networks efficiently when larger batches and multiple workers change the balance between accuracy, computation, and communication.**

This research project, developed at **Politecnico di Torino**, compares centralized optimization, large-batch training, and simulated distributed learning using **PyTorch**, a **LeNet-5-style CNN**, and **CIFAR-100**. Beyond reproducing established approaches, the project introduces an experimental **adaptive LocalSGD** variant that adjusts the number of local training steps using gradient statistics.

> **Focus:** Optimization experiments and research—not a production distributed-training framework. The worker-based experiments simulate multiple workers sequentially; they do not demonstrate a multi-GPU or multi-node speedup.

## At a glance

| | |
|---|---|
| **Task** | CIFAR-100 image classification |
| **Model** | LeNet-5-style convolutional neural network |
| **Framework** | Python, PyTorch, torchvision |
| **Centralized baselines** | SGDM and AdamW |
| **Large-batch optimizers** | SGDM, AdamW, LARS, LAMB |
| **Distributed-learning experiments** | LocalSGD, SlowMo, adaptive LocalSGD |
| **My individual contribution** | Dynamic adjustment of LocalSGD's synchronization interval based on gradient magnitude and variance |

## Why this project?

When training is distributed across workers, synchronizing model updates after every step can become expensive. Delaying synchronization reduces communication but also lets workers drift apart. Similarly, increasing batch sizes improves throughput potential but can hurt model generalization.

We studied these trade-offs through a set of experiments:

1. **Establish centralized baselines:** tune and evaluate SGDM and AdamW.
2. **Scale the batch size:** compare SGDM, AdamW, LARS, and LAMB.
3. **Simulate multiple workers:** test LocalSGD under different worker counts and local-step intervals.
4. **Introduce slow momentum:** evaluate SlowMo against LocalSGD.
5. **Adapt synchronization:** experiment with dynamically changing the local-step interval instead of keeping it fixed.

## Research highlights

### Centralized training

The report records the following results with **150 epochs** and **batch size 64**:

| Optimizer | Test accuracy | Test loss |
|---|---:|---:|
| SGDM | 51.47% | 1.9361 |
| AdamW | 52.86% | 1.8173 |

### Large-batch optimization

We compared performance as batch sizes increased from **128 to 8,192**. At batch size **8,192**, the report records **43.70%** test accuracy for LAMB and **43.60%** for AdamW, compared with **35.77%** for LARS. These are results for this model and experimental setup, not general optimizer rankings.

### LocalSGD: communication versus computation

In the simulated-worker experiments, increasing the number of local steps reduced the measured communication/synchronization component while increasing computation time. For **K = 2 workers**, moving from **J = 4** to **J = 64** local steps changed the reported communication time from **11.37 s** to **0.88 s**, while computation time increased from **3,482.30 s** to **4,148.00 s**.

These timing values come from the project's experimental simulation and should **not** be interpreted as measured network performance or end-to-end distributed speedup.

### My contribution: adaptive LocalSGD

Rather than using a fixed number of local updates between synchronization events, I worked on a method that **adjusts the synchronization interval according to gradient magnitude and variance**:

- When gradients are relatively consistent, workers can take more local steps before synchronization.
- When gradient variability increases, the method can shorten the interval to limit drift.
- The interval remains bounded by a minimum and maximum number of steps.

The experiments showed a meaningful reduction in the measured communication component under some parameter settings, while **test accuracy and loss remained broadly similar**. The research did **not** establish a major accuracy improvement, and the increased local computation is part of the trade-off.

See the paper for the mathematical formulation and full results.

## Repository guide

```text
AML_Project5/
├── AML_Project5.pdf                    # Research report and experimental tables
├── README.md
└── code/
    ├── Centralized/
    │   ├── AdamW/                      # Tuning and testing notebooks
    │   └── SGDM/                       # Tuning and testing notebooks
    ├── Large Batch/                    # Large-batch optimizer experiments
    ├── LocalSGD/                       # Fixed-interval LocalSGD
    ├── SlowMo/                         # Slow momentum experiments
    └── Personal_contribution/
        └── DynamicSGD.ipynb            # Adaptive local-step experiment
```

The repository contains **Jupyter notebooks**, not standalone `.py` files checked into Git. Some notebooks create temporary Python scripts using the `%%writefile` notebook magic and execute those scripts with `%run`.

## Running the experiments

The notebooks were written for **Google Colab** and use `google.colab` imports. Running them elsewhere may require small compatibility changes.

1. Open the desired `.ipynb` file in **Google Colab**.
2. Enable a GPU runtime if available (**Runtime → Change runtime type**). GPU access is optional in principle but experiments may take substantial time without one.
3. Run the notebook cells in order. CIFAR-100 is obtained through `torchvision` where configured by the notebook; allow for dataset downloads.
4. For the notebooks that generate a `.py` script, run the script-generation cell before the `%run` cell.
5. Adjust the hyperparameters in the notebook or the relevant command-line arguments to reproduce a specific experiment.

Core Python dependencies visible in the notebooks include `torch`, `torchvision`, `numpy`, and `matplotlib`. Some experiments use additional optimizer implementations defined in the notebooks. Dependency versions are not pinned, so reproducibility may require adapting to current PyTorch/torchvision versions.

### Example: adaptive LocalSGD

Open `code/Personal_contribution/DynamicSGD.ipynb` in Colab and run the cells in order. Its script-generation cell writes `DynamicSGD.py`; the experiment can then be launched from the notebook with commands following this pattern:

```python
%run DynamicSGD.py --k 2 --j 32
```

`--k` represents the number of simulated workers, and `--j` provides the initial local-step setting. Consult the notebook for the exact current argument handling and other fixed hyperparameters.

> **Reproducibility note:** These are research notebooks with substantial training workloads. A clean-environment execution, dependencies, complete deterministic reproduction, and contemporary GPU compatibility have **not** been verified as part of this README update.

## Research report

Read [**AML_Project5.pdf**](./AML_Project5.pdf) for the method, related work, experimental setup, tables, discussion, and references. The report was prepared as a **group project** by Sara Poiatti, Luca Vigneri, and Anjali Narendra Vaghjiani. The adaptive LocalSGD investigation is identified as the personal contribution in the repository.

## Limitations and possible next steps

- The distributed experiments **simulate worker behavior serially** rather than using real multi-node communication.
- The experiments use one image dataset and one model family, so the findings may not generalize to other architectures or workloads.
- The reported communication and computation breakdowns should not be equated with production cluster performance.
- Notebook scripts and package versions could be standardized for easier reproduction.
- Useful follow-up work would include distributed `torch.distributed` execution, multi-seed evaluation, automated tests, and plots regenerated from archived results.

## Technologies and concepts

`Python` · `PyTorch` · `torchvision` · `CIFAR-100` · `Deep Learning` · `Distributed Optimization` · `LocalSGD` · `SlowMo` · `SGDM` · `AdamW` · `LARS` · `LAMB` · `Gradient Statistics` · `Google Colab`
