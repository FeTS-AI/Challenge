---
title: Participation Details
permalink: /participate/
---
# Participation policies
- only automatic segmentation methods allowed
- Participants are NOT allowed to use additional public and/or private data (from their own institutions) for extending the provided data. Similarly, using models that were pretrained on such datasets is NOT allowed. This is due to our intentions to provide a fair comparison among the participating methods.
- open source policy? accessibility of participants' code?
- The top 3 performing methods for each task will be announced publicly at the conference and the participants will be invited to
present their method.
- Inclusion criteria for the test phase of task 2: As we are going to perform a real-world federated evaluation in task 2, the computation capabilities are heterogeneous and restricted. Therefore, we reserve the right to limit the number of task-2 submissions included in the final ranking. Details are given in [the evaluation section](#federated-evaluation-process)
- We reserve the right to exclude teams and team members if they do not adhere to the challenge rules.

# Submission Process

## Task 1
todo
## Task 2
Something like
1. registration at e.g. gitlab (tbd)
2. upload of singularity container
3. (optionally) requesting functionality test (only X trials)
4. Submission form and short paper in IPP

Examples for how to build a singularity container are given in the [challenge repository](https://github.com/FETS-AI/Challaenge)

# Evaluation
What are the requirements for a submission to be included in the final evaluation?

Metrics here or together with [Data](/data/)?
## Task 2

### Code Review

- Maybe how we'll do it
- hardware used for container test


### Federated Evaluation Process
To be eligible for evaluation on the test set, participants need to fulfill the general requirements described [above](#evaluation). Furthermore, the following rules apply to the submissions (including short paper and algorithm):
- Only submissions that pass the code review will be evaluated
- Each algorithm is given 180 seconds per test case to produce a prediction. Algorithms that fail to stay in this time budget during the code review phase will not be considered for evaluation.

<!-- - MAYBE Short papers will be checked for completeness (i.e. are all parts of the template present and described sufficiently) and those with missing parts/insufficient qualitites will receive lower priority for evaluation. -->

- Algorithms will be evaluated on the test set in the chronological order they were submitted in (sorted by date-time of last container upload). This means the later an algorithm is submitted, the higher the risk it cannot be evaluated on the test set before the ranking is computed. Note that this is a worst-case rule and we will work hard to include every single valid submission in the ranking. In case of very high participation numbers, however, we reserve the right to limit the number of participants in the final MICCAI ranking this way.

<!-- - MAYBE Challenge results will be updated after MICCAI if necessary, after all submission have been evaluated. -->
# Publication

todo