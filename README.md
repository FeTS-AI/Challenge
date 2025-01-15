<a href="https://arxiv.org/abs/2105.05874" alt="Citation"><img src="https://img.shields.io/badge/cite-citation-blue" /></a>
<a href="https://twitter.com/FeTS_Challenge" alt="Citation"><img src="https://img.shields.io/twitter/follow/fets_challenge?style=social" /></a>

# Federated Tumor Segmentation Challenge

The repo for the FeTS Challenge: The 1st Computational Competition on Federated Learning.

## Quickstart

The official challenge website with detailed information on the challenge is:

https://www.synapse.org/#!Synapse:syn28546456

As the challenge is currently inactive, submitting algorithms is not possible. However, the challenge data is accessible [here](https://www.synapse.org/Synapse:syn54079892/wiki/626854). Please check the instructions there for downloading it and the conditions of use.

This repository complements the challenge website above by providing code for developing and testing algorithm submissions to the two task of the FeTS Challenge:

### Task 1

The first task of the challenge involves customizing core functions of a baseline federated learning system implementation. The goal is to improve over the baseline consensus models in terms of robustness in final model scores to data heterogeneity across the simulated collaborators of the federation. For more details, please see [Task_1](./Task_1).

### Task 2

This task utilizes decentralized testing across various sites of the FeTS initiative in order to evaluate model submissions across data from different medical institutions, MRI scanners, image acquisition parameters and populations. The goal of this task is to find algorithms (by whatever training technique you wish to apply) that score well across these data. For more details, please see [Task_2](./Task_2).

## Documentation and Q&A

Please visit the [challenge website](https://synapse.org/fets) and [forum](https://www.synapse.org/#!Synapse:syn28546456/discussion/default).

<!-- ## Frequently asked questions

Please see [FAQ](https://fets-ai.github.io/Challenge/faq/). -->

## Citation

Please cite [this paper](https://arxiv.org/abs/2105.05874) when using the data:

```latex
@misc{pati2021federated,
      title={The Federated Tumor Segmentation (FeTS) Challenge}, 
      author={Sarthak Pati and Ujjwal Baid and Maximilian Zenk and Brandon Edwards and Micah Sheller and G. Anthony Reina and Patrick Foley and Alexey Gruzdev and Jason Martin and Shadi Albarqouni and Yong Chen and Russell Taki Shinohara and Annika Reinke and David Zimmerer and John B. Freymann and Justin S. Kirby and Christos Davatzikos and Rivka R. Colen and Aikaterini Kotrotsou and Daniel Marcus and Mikhail Milchenko and Arash Nazer and Hassan Fathallah-Shaykh and Roland Wiest and Andras Jakab and Marc-Andre Weber and Abhishek Mahajan and Lena Maier-Hein and Jens Kleesiek and Bjoern Menze and Klaus Maier-Hein and Spyridon Bakas},
      year={2021},
      eprint={2105.05874},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```
