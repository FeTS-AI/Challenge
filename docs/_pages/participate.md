---
title: Participation Details
permalink: /participate/
---

## Timeline

For the exact dates and deadlines, please see the schedule on the [challenge schedule](index.md/#important-dates).

**Training Phase.** Register (see [FAQ](/faq)) to download the co-registered, skull-stripped, and annotated training data.

**Validation Phase.** An independent set of validation scans will be made available to the participants in June, with the intention to allow them assess the generalizability of their methods in unseen data, via [CBICA's Image Processing Portal (IPP)](https://ipp.cbica.upenn.edu/). Note that this may not reflect the out-of-distribution generalization aimed at in task 2. The leaderboard will be available through a link from this page.

**Short Paper submission deadline.** Participants will have to evaluate their methods on the training and validation datasets, and submit their short paper (8-10 LNCS pages — together with the "[LNCS Consent to Publish](https://resource-cms.springernature.com/springer-cms/rest/v1/content/15433008/data/Contract_Book_Contributor_Consent_to_Publish_LNCS_SIP)" form), describing their method and results to the BrainLes [CMT submission system](https://cmt3.research.microsoft.com/BrainLes2019/), and make sure you choose FeTS as the "Track". Please ensure that you include the appropriate citations, mentioned at the bottom of the "Data" section. This unified scheme should allow for appropriate preliminary comparisons and the creation of the pre- and post-conference proceedings. Participants are allowed to submit longer papers to the [MICCAI 2021 BrainLes Workshop](http://www.brainlesion-workshop.org/), by choosing "BrainLes" as the "Track". FeTS papers will be part of the BrainLes workshop proceedings distributed by Springer LNCS. All paper submissions should use the LNCS template, available both in LaTeX and in MS Word format, directly from Springer ([link here](https://www.springer.com/us/computer-science/lncs/conference-proceedings-guidelines)).

**Testing Phase.** The test scans are not made available to participating teams. The organizers will evaluate the submitted contributions instead for all participants that submitted a short paper, and an appropriate version of their algorithm, as described in each task ([evaluation section](#evaluation)). *Participants that have not submitted a short paper, and the copyright form, will not be evaluated*.

**Oral Presentations.** The top-ranked participants will be contacted in September to prepare slides for orally presenting their method during the FeTS satellite event at MICCAI 2021, on Oct. 1.

**Announcement of Final Results (Oct 1).** The final rankings will be reported during the FeTS 2021 challenge, which will run in conjunction with MICCAI 2021.

**Post-conference LNCS paper (Oct 10).** All participanting teams are invited to extend their papers to 11-14 pages for inclusion to the LNCS proceedings of the BrainLes Workshop.

**Joint post-conference journal paper.** All participating teams have the chance to be involved in the joint manuscript summarizing the results of FeTS 2021, that will be submitted to a high-impact journal in the field. To be involved in this manuscript, the participating teams will need to participate in all phases of at least one of the FeTS tasks.

## Participation policies

- Only automatic segmentation methods allowed
- Participants are NOT allowed to use additional public and/or private data (from their own institutions) for extending the provided data. Similarly, using models that were pretrained on such datasets is NOT allowed. This is due to our intentions to provide a fair comparison among the participating methods.
- **!!!TODO!!!** open source policy? accessibility of participants' code?
- The top 3 performing methods for each task will be announced publicly at the conference and the participants will be invited to present their method.
- Inclusion criteria for the test phase of task 2: As we are going to perform a real-world federated evaluation in task 2, the computation capabilities are heterogeneous and restricted. Therefore, we reserve the right to limit the number of task-2 submissions included in the final ranking. Details are given in [below](#federated-evaluation-process).
- We reserve the right to exclude teams and team members if they do not adhere to the challenge rules.

## Registration and Data Access

To register and request the training and the validation data of the FeTS 2021 challenge, please follow the steps below. Please note that the i) training data includes ground truth annotations, ii) validation data does not include annotations, and iii) testing data are not available to either challenge participants or the public.

1. Create an account in [CBICA's Image Processing Portal](https://ipp.cbica.upenn.edu/) (IPP) and wait for its approval. Note that a confirmation email will be sent so make sure that you also check your Spam folder. This approval process requires a manual review of the account details and might take 3-4 days until completed.
2. Once your IPP account is approved, login to [IPP](https://ipp.cbica.upenn.edu/) and then click on the application **FeTS 2021: Registration**, under the **MICCAI FeTS 2021** group.
3. Fill in the requested details and press "Submit Job".
4. Once your request is recorded, you will receive an email pointing to the "results" of your submitted job. You need to login to IPP, access the "Results.zip" file, in which you will find the file `REGISTRATION_STATUS.txt`. In this txt file you will find the links to download the FeTS 2021 data. The training data will include for each subject the 4 structural modalities, ground truth segmentation labels and accompanying text information relating to the source institution, whereas the validation data will include only the 4 modalities.

## Submission Process

### Task 1 Submission

Description coming soon...

### Task 2 Submission

To provide high implementation flexibility to the participants while also facilitating the federated evaluation on different computation infrastructures, algorithm submissions for this task have to be [singularity containers](https://sylabs.io/singularity/). The container application should be able to produce segmentations for a list of test cases. Details on the interface and examples for how to build such a container are given in the [challenge repository](https://github.com/FETS-AI/Challenge/tree/main/Task_2).

Each participating team will be provided a gitlab project where they can upload their submission. To make a submission to task 2:

1. Sign up at [https://gitlab.hzdr.de/](https://gitlab.hzdr.de/) by either clicking *Helmholtz AAI* (login via your institutional email) or via your github login. Both buttons are in the lower box on the right.
2. Send an email to [challenge@fets.ai](mailto:challenge@fets.ai), asking for a Task 2-gitlab project and stating your gitlab handle (@your-handle) and team name. We will create a project for you and invite you to it within a day.
3. Follow the instructions in the newly created project to make a submission.

To make sure that the containers submitted by the participants also run successfully on the remote institutions in the FeTS federation, we offer functionality tests on toy cases. Details are provided in the gitlab project.

## Evaluation

Participants are called to produce segmentation labels of the different glioma sub-regions:

1. the “enhancing tumor” (ET), equivalent to label 4
2. the “tumor core” (TC), comprising labels 1 and 4
3. the “whole tumor” (WT), comprising labels 1, 2 and 4

For each region, the predicted segmentation is compared with the groundtruth segmentation using the "Dice score" and "Hausdorff distance (95%)". Additionally for task 1, the communication cost during model training is taken into account.

### Task 1 Evaluation Details

Description coming soon...

### Task 2 Evaluation Details

#### Code Review

To make sure that the submitted containers are functional and to prevent misconduct, we are going to review each submission manually before the actual federated evaluation. Regarding functionality, we intend to check the validity of the algorithm output and measure the execution time of the container on a small dataset using a pre-defined GPU setup (RTX 2080 Ti). Regarding security, we will inspect the code being executed by the container and discuss any unclear points with the participants.

#### Federated Evaluation Process

Participants have to adhere to the challenge rules described [above](#participation-policies) to be eligible for evaluation on the test set. Furthermore, the following rules apply to the submissions:

- Only submissions that include a complete short paper will be considered for evaluation.
- Only submissions that pass the code review will be considered for evaluation.
- Each submitted container is given 180 seconds per case in the code review phase to produce a prediction. Submissions that fail to stay in this time budget will not be considered for the federated evaluation.
- Algorithms will be evaluated on the test set in the chronological order they were submitted in (sorted by date-time of last container upload). This means the later an algorithm is submitted, the higher is the risk it cannot be evaluated on all federated test sets before the end of the testing phase. Note that this is a worst-case rule and we will work hard to include every single valid submission in the ranking. In case of very high participation numbers, however, we reserve the right to limit the number of participants in the final MICCAI ranking this way.

<!-- - MAYBE Short papers will be checked for completeness (i.e. are all parts of the template present and described sufficiently) and those with missing parts/insufficient qualitites will receive lower priority for evaluation. -->
<!-- - MAYBE Challenge results will be updated after MICCAI if necessary, after all submission have been evaluated. -->

#### Ranking

Only the external FeTS testing institutions (that are not part of the training data) are used for the ranking. First, on institution `k`, algorithms are ranked on all `N_k` test cases, three regions and two metrics, yielding `N_k * 3 * 2` ranks for each algorithm. Averaging these produces a score equivalent to a per-institution rank for each algorithm (rank-then-aggregate approach). The final rank of an algorithm is computed from the average of its per-institution ranks. Ties are resolved by assigning the minimum rank.
