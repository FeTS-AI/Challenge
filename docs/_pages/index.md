---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

permalink: /
title: Federated Tumor Segmentation Challenge 2021
# callouts: home_callouts
hide_hero: false
hero_height: is-medium
menubar_toc: false
show_sidebar: true
---

Welcome to the webpages of the FeTS 2021 challenge! Here you can find general information on the two challenge tasks, their data, as well as details on how to participate. The official challenge design document can be found [here](https://zenodo.org/record/4573128#.YJKcEcCSk4s) and the accompanying arXiv-manuscript [here](https://arxiv.org/abs/2105.05874).

## News

No news so far

## Important Dates

All deadlines are for **23:59 Eastern Time**

| Date | Description|
| --- | --- |
| **21 May** | Training phase (Release of training data + associated ground truth). |
| **14 Jun** | Validation phase (Release of validation data. Hidden ground truth). |
| **19 Jul** | Submission of short paper and prediction algorithm (incl. model weights).|
| **20 Jul-27 Aug** | Testing phase (Evaluation by organizers, only for methods with submitted papers) |
| **3 Sep** | Contacting top-ranked methods to prepare their oral presentation at MICCAI |
| **1 Oct (PM)** | Announcement of top 3 ranked teams at MICCAI FeTS 2021. |
| **10 Oct** | Submission deadline for extended LNCS papers (12-14 pages) |
| **24 Oct** | Reviewers' feedback. |
| **10 Nov** | Camera-ready paper submission. |
| **15 Dec** | Summarizing meta-analysis manuscript. |

## Challenge Description

International challenges have become the standard for validation of biomedical image analysis methods. We argue, though, that the actual performance even of the winning algorithms on “real-world” clinical data often  remains unclear, as the data included in these challenges are usually acquired in very controlled settings at few institutions. The seemingly obvious solution of just collecting increasingly more data from more geographically distinct institutions in such challenges does not scale well due to privacy, ownership, and technical hurdles.

The Federated Tumor Segmentation (FeTS) challenge 2021 is the first challenge to ever be proposed for federated learning in medicine, and intends to address these hurdles, for both the creation and the evaluation of tumor segmentation models. Specifically, the FeTS 2021 challenge uses clinically acquired, multi-institutional MRI scans from the BraTS 2020 challenge, as well as from various remote independent institutions included in the collaborative network of a real-world federation ([FeTS initiative](https://www.fets.ai/)).

The FeTS challenge focuses on the construction and evaluation of a consensus model for the segmentation of intrinsically heterogeneous (in appearance, shape, and histology) brain tumors, namely gliomas. Compared to the BraTS 2020 challenge, the ultimate goal of FeTS is 1) the creation of a consensus segmentation model that has gained knowledge from data of multiple institutions without pooling their data together (i.e., by retaining the data within each institution), and 2) the evaluation of segmentation models in such a federated configuration (i.e., in the wild).

The FeTS 2021 challenge is structured in two explicit tasks:

### Task 1: Federated Training (FL Weight Aggregation Methods)

The first task of the challenge involves creating a robust consensus model for segmentation of brain tumor sub-regions that has gained knowledge from data acquired at multiple sites, without pooling data together. The specific focus of this task is to identify the best way to aggregate the knowledge  coming  from  segmentation  models  trained  on  individual  institutions,  instead  of  identifying  the  best  segmentation  method.  More  precisely,the focus is on the methodological portions specific to federated learning (e.g. aggregation,  client  selection,  training-per-round,  compression,  communication efficiency),  and  not  on  the  development  of  segmentation  algorithms  (which  is the focus of the BraTS challenge). To facilitate this, an existing infrastructure for  federated  tumor  segmentation  using  federated  averaging  will  be  provided to all participants indicating the exact places that the participants are allowed and expected to make changes. The primary objective of this task is to develop methods for effective aggregation of local segmentation models, given the partitioning of the data into their real-world distribution. As an optional sub-task, participants will be asked to account for network communication outages, i.e. dealing with stragglers.

### Task 2: Federated Evaluation (Generalization “In The Wild”)

The discrepancy between AI systems’ performance in research environments and real-life applications is one of the key challenges in our field.  This  “AI  chasm”  can  be  attributed  in  part  to  the  limited  diversity of  training  datasets,  which  do  not  necessarily  reflect  the  variety  of  real-world datasets  “in  the  wild”.  As  a  consequence,  most  deep  learning  models  exhibit limited generalizability when applied to datasets acquired from different imaging devices  and  populations.  Federated  setups  are  not  only  beneficial  for  learning models; they also allow to extend the size and diversity of typical test datasets substantially, as clinicians may contribute data to a challenge without having to publicly release them, thus constituting an important step towards the evaluation of model robustness in the wild.

In  this  task,  the  goal  is  to  find  algorithms  that  robustly  produce  accurate brain tumor segmentations across different medical institutions, MRI scanners, image acquisition parameters and populations. To this end, we use a real-world federated evaluation environment (based on the [FeTS initiative](https://www.fets.ai/)). In the training phase, the participants will be provided the training set including information on the data origin (see also the [data page](data.md/#non-imaging-data-description)). They can explore the effects of distribution shifts between contributing sites to find tumor segmentation algorithms that are able to generalize to data acquired at institutions that did not contribute to the training dataset. Note that *training  on  pooled  data  is  allowed* in this task, so that the participants can develop methods that optimally exploit the meta-information of data origin. After training, all participating algorithms will be evaluated in a distributed way on datasets from various institutions of the FeTS federation, such that the test data are always retained within their owners’ servers.

## People

### Organizing Committee

(in alphabetical order, except lead organizers)

- [Spyridon (Spyros) Bakas, Ph.D.](https://www.med.upenn.edu/cbica/sbakas/) *--- [Task 1 Lead Organizer]*,  [Center for Biomedical Image Computing and  Analytics (CBICA)](https://www.med.upenn.edu/cbica/), UPenn, Philadelphia, PA, USA
- Maximilian Zenk   *--- [Task 2 Lead Organizer]*,    [Div. Medical Image Computing (MIC)](https://www.dkfz.de/en/mic/index.php), German Cancer Research - Center (DKFZ), Heidelberg, Germany
- Shadi Albarqouni, Ph.D.,    Technical University of Munich, Germany
- Ujjwal Baid, Ph.D.,    CBICA, UPenn, Philadelphia, PA, USA
- Yong Chen, Ph.D.,    UPenn, Philadelphia, PA, USA
- Brandon Edwards, Ph.D.,    Intel, USA
- Patrick Foley,    Intel, USA
- Alexey Gruzdev,    Intel, USA
- Jens Kleesiek, M.D., Ph.D.,    Translational Image-guided Oncology, Institute for AI in Medicine (IKIM), - University Hospital Essen, Germany
- Klaus Maier-Hein, Ph.D.,    MIC, DKFZ, Heidelberg, Germany
- Lena Maier-Hein, Ph.D.,    Div. Computer Assisted Medical Interventions (CAMI), DKFZ, Heidelberg, Germany
- Jason Martin,    Intel, USA
- Bjoern Menze, Ph.D.,    University of Zurich, Switzerland
- Sarthak Pati,    CBICA, UPenn, Philadelphia, PA, USA
- Annika Reinke,    Div. Computer Assisted Medical Interventions (CAMI), DKFZ, Heidelberg, Germany
- Micah J Sheller,    Intel, USA
- Russell Taki Shinohara, Ph.D.,    UPenn, Philadelphia, PA, USA
- David Zimmerer,    MIC, DKFZ, Heidelberg, Germany

### Data Contributors

- John B. Freymann & Justin S. Kirby - on behalf of The Cancer Imaging Archive (TCIA),    Cancer Imaging Program, - NCI, National Institutes of Health (NIH), USA
- Christos Davatzikos, Ph.D.,    CBICA, UPenn, Philadelphia, PA, USA
- Rivka R. Colen, M.D., & Aikaterini Kotrotsou, Ph.D.,    MD Anderson Cancer Center, TX, USA
- Daniel Marcus, Ph.D., & Mikhail Milchenko, Ph.D., & Arash Nazeri, M.D.,    Washington University School of - Medicine in St. Louis, MO, USA
- Hassan Fathallah-Shaykh, M.D., Ph.D.,    University of Alabama at Birmingham, AL, USA
- Roland Wiest, M.D.,    University of Bern, Switzerland
- Andras Jakab, M.D., Ph.D.,    University of Debrecen, Hungary
- Marc-Andre Weber, M.D.,    Heidelberg University, Germany
- Abhishek Mahajan, M.D., & Ujjwal Baid, Ph.D.,    Tata Memorial Centre, Mumbai, India, & SGGS Institute of - Engineering and Technology, Nanded, India

### Clinical Evaluators and Annotation Approvers

- Michel Bilello, MD, Ph.D.,    UPenn, Philadelphia, PA, USA
- Suyash Mohan, MD, Ph.D.,    UPenn, Philadelphia, PA, USA

### Awards Sponsor

- Prashant Shah – on behalf of Intel Corporation

### Acknowledgements

- Chiharu Sako, Ph.D.,  (Data Analysts - CBICA, UPenn, PA, USA) for her invaluable assistance in the datasets' organization.
- G Anthony Reina, M.D.,    Intel AI
