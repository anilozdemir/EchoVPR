[ArXiv pre-print with supplementary materials](https://arxiv.org/abs/2110.05572)


- You can find two example notebooks ([standard ESN and SpaRCe](https://colab.research.google.com/github/anilozdemir/EchoVPR/blob/main/notebooks/example_train_single_ESN.ipynb) & [hierarchical ESN and SpaRCe](https://colab.research.google.com/github/anilozdemir/EchoVPR/blob/main/notebooks/example_train_hier_ESN.ipynb)) and the [source code](https://github.com/anilozdemir/EchoVPR/tree/main/src).
- [Repo](https://github.com/anilozdemir/EchoVPR) for results Section IV. A,B,E: **GardensPoint, SPEDTest, ESSEX3IN1, Corridor**, and (subset) **Nordland** datasets, and **supplementary materials**
- [Repo](https://github.com/mscerri/EchoVPR) for results Section IV. C,D,E: **Nordland** and **Oxford RobotCar** datasets


## Paper Abstract

Recognising previously visited locations is an important, but unsolved, task in autonomous navigation. Current visual place recognition (VPR) benchmarks typically challenge models to recover the position of a query image (or images) from sequential datasets that include both spatial and temporal components. Recently, Echo State Network (ESN) varieties have proven particularly powerful at solving machine learning tasks that require spatio-temporal modelling. These networks are simple, yet powerful neural architectures that—exhibiting memory over multiple time-scales and non-linear high-dimensional representations—can discover temporal relations in the data while still maintaining linearity in the learning. In this paper, we present a series of ESNs and analyse their applicability to the VPR problem. We report that the addition of ESNs to pre-processed convolutional neural networks led to a dramatic boost in performance in comparison to non-recurrent networks in four standard benchmarks (GardensPoint, SPEDTest, ESSEX3IN1, Nordland) demonstrating that ESNs are able to capture the temporal structure inherent in VPR problems. Moreover, we show that ESNs can outperform class-leading VPR models which also exploit the sequential dynamics of the data. Finally, our results demonstrate that ESNs also improve generalisation abilities, robustness, and accuracy further supporting their suitability to VPR application.

## Link To The Datasets

To be added.