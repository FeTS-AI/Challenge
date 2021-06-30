---
title: Challenge Description
permalink: /tasks/
---

## Challenge Description

International challenges have become the standard for validation of biomedical image analysis methods. We argue, though, that the actual performance even of the winning algorithms on “real-world” clinical data often  remains unclear, as the data included in these challenges are usually acquired in very controlled settings at few institutions. The seemingly obvious solution of just collecting increasingly more data from more geographically distinct institutions in such challenges does not scale well due to privacy, ownership, and technical hurdles.

The Federated Tumor Segmentation (FeTS) challenge 2021 is the first challenge to ever be proposed for federated learning in medicine, and intends to address these hurdles, for both the creation and the evaluation of tumor segmentation models. Specifically, the FeTS 2021 challenge uses clinically acquired, multi-institutional MRI scans from the BraTS 2020 challenge, as well as from various remote independent institutions included in the collaborative network of a real-world federation ([FeTS initiative](https://www.fets.ai/)).

The FeTS challenge focuses on the construction and evaluation of a consensus model for the segmentation of intrinsically heterogeneous (in appearance, shape, and histology) brain tumors, namely gliomas. Compared to the BraTS 2020 challenge, the ultimate goal of FeTS is 1) the creation of a consensus segmentation model that has gained knowledge from data of multiple institutions without pooling their data together (i.e., by retaining the data within each institution), and 2) the evaluation of segmentation models in such a federated configuration (i.e., in the wild).
The FeTS 2021 challenge is structured in two explicit tasks:

- Task 1 ("Federated Training") aims at effective weight aggregation methods for the creation of a consensus model given a pre-defined segmentation algorithm for training, while also (optionally) accounting for network outages.
- Task 2 ("Federated Evaluation") aims at robust segmentation algorithms, evaluated during the testing phase on unseen datasets from various remote independent institutions of the collaborative network of the fets.ai federation.

These tasks are described in more detail below. Participants are free to choose whether they want to focus on only one or multiple tasks.

The clinical relevance and importance of the FeTS challenge is that it addresses challenges related to privacy, legal, bureaucratic, and ownership concerns raised in the current paradigm of multi-site collaborations through data sharing. The official challenge design document can be found [here](https://zenodo.org/record/4573128#.YJKcEcCSk4s) and the accompanying arXiv-manuscript [here](https://arxiv.org/abs/2105.05874).

**License Conformance for Participants' code**

By participating and submitting your contribution to the FeTS 2021 challenge, for review and evaluation during the testing/ranking phase, you confirm that your code follows a license conforming to one of the standards: Apache 2.0, BSD-style, or MIT.

### Task 1: Federated Training (FL Weight Aggregation Methods) {#task1-description}

The specific focus of this task is to identify the best way to aggregate the knowledge coming from segmentation models trained on the individual institutions, instead of identifying the best segmentation method. More specifically, the focus is on the methodological portions specific to federated learning (e.g., aggregation, client selection, training-per-round), and not in the development of segmentation algorithms (that the BraTS challenge focuses on).
Provided Infrastructure

**Provided Infrastructure**

To facilitate this task, an existing infrastructure for federated tumor segmentation using federated averaging is provided to all participants in [GitHub](https://github.com/FETS-AI/Challenge/tree/main/Task_1), indicating the exact places that the participants are allowed and expected to make changes. This infrastructure can be found in GitHub, at: [https://github.com/FETS-AI/Challenge/tree/main/Task_1](https://github.com/FETS-AI/Challenge/tree/main/Task_1)

Specific instructions are given to the participants on the parts/functions that they would need to alter the federated algorithm in the following ways:

- The aggregation function used to fuse the collaborator model updates.
- Which collaborators are chosen to train in each federated round.
- The training parameters for each federated round.
- The validation metrics to be computed each round (that can then be used as inputs to the other functions).

The primary goal involved in this task comprises the aggregation of local segmentation models given the partitioning of the data following their real-world distribution.

**Performance Evaluation**
The evaluation metrics considered for this task are:

- Dice Similarity Coefficient
- Hausdorff Distance - 95th percentile
- Communication cost, during model training, i.e., Budget time (product of bytes sent/received * number of federated rounds)
- Sensitivity (this will not be used for ranking purposes)
- Specificity (this will not be used for ranking purposes)

### Task 2: Federated Evaluation (Generalization “In The Wild”) {#task2-description}

<!--SB: I find this aprt too fuzzy for the description of the task. Good for the manuscript though :)
The discrepancy between AI systems’ performance in research environments and real-life applications is one of the key challenges in our field.  This  “AI  chasm”  can  be  attributed  in  part  to  the  limited  diversity of  training  datasets,  which  do  not  necessarily  reflect  the  variety  of  real-world datasets  “in  the  wild”.  As  a  consequence,  most  deep  learning  models  exhibit limited generalizability when applied to datasets acquired from different imaging devices  and  populations.  Federated  setups  are  not  only  beneficial  for  learning models; they also allow to extend the size and diversity of typical test datasets substantially, as clinicians may contribute data to a challenge without having to publicly release them, thus constituting an important step towards the evaluation of model robustness in the wild.
-->
In this task, the goal is to find algorithms that robustly produce accurate brain tumor segmentations across different medical institutions, MRI scanners, image acquisition parameters and populations. To this end, we use a real-world federated evaluation environment (based on the [FeTS initiative](https://www.fets.ai/)). In the training phase, the participants will be provided the training set including information on the data origin (see also the [data page](data.md/#non-imaging-data-description)). They can explore the effects of distribution shifts between contributing sites to find tumor segmentation algorithms that are able to generalize to data acquired at institutions that did not contribute to the training dataset. Note that *training on pooled data is allowed* in this particular task, so that the participants can develop methods that optimally exploit the meta-information of data origin.

After training, all participating algorithms will be evaluated in a distributed way on data from multiple institutions of the first real-world federation reported in the [FeTS initiative](https://www.fets.ai/) that have graciously accepted to be part of the FeTS challenge, such that the test data are always retained within their owner's server.

This “real-world semantic segmentation challenge” is to hopefully provide a blueprint for future similar “phase 2” challenge endeavors. From a methodical perspective, the main goal of this task is to identify segmentation algorithms that are robust to unknown and realistic distribution shifts between training/validation and test data.

**Performance Evaluation**
The evaluation metrics considered for this task are:

- Dice Similarity Coefficient
- Hausdorff Distance - 95th percentile
- Sensitivity (this will not be used for ranking purposes)
- Specificity (this will not be used for ranking purposes)
