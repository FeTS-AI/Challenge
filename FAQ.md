# Frequently asked questions

## Table of contents
* [What is the main website for the challenge?](#what-is-the-main-website-for-the-challenge)
* [How do I register for the challenge?](#how-do-i-register-for-the-challenge)
* [How do I obtain the data for the challenge?](#how-do-i-obtain-the-data-for-the-challenge)
* [What is the data citation?](#what-is-the-data-citation)

## What is the main website for the challenge?

The main website is https://miccai2021.fets.ai/. Additionally, participants may follow the related [Twitter handle](https://twitter.com/FeTS_Challenge) for updates.

## How do I register for the challenge?

To register and request the training and the validation data of the FeTS 2021 challenge, please follow the steps below. Please note that the i) training data includes ground truth annotations, ii) validation data does not include annotations, and iii) testing data are not available to either challenge participants or the public.

1. Create an account in [CBICA's Image Processing Portal](https://ipp.cbica.upenn.edu/) and wait for its approval. Note that a confirmation email will be sent so make sure that you also check your Spam folder. This approval process requires a manual review of the account details and might take 3-4 days until completed.
2. Once your IPP account is approved, login to https://ipp.cbica.upenn.edu/ and then click on the application **FeTS 2021: Registration**, under the **MICCAI FeTS 2021** group.
3. Fill in the requested details and press "Submit Job".
4. Once your request is recorded, you will receive an email pointing to the "results" of your submitted job. You need to login to IPP, access the "Results.zip" file, in which you will find the file `REGISTRATION_STATUS.txt`. In this txt file you will find the links to download the FeTS 2021 data. The training data will include for each subject the 4 structural modalities, ground truth segmentation labels and accompanying text information relating to the source institution, whereas the validation data will include only the 4 modalities.

## How do I obtain the data for the challenge?

Please see [above](#how-do-i-register-for-the-challenge).

## What is the data citation?

Please cite [this paper](https://arxiv.org/abs/2105.05874) when using data from this challenge:
```
@misc{pati2021federated,
      title={The Federated Tumor Segmentation (FeTS) Challenge}, 
      author={Sarthak Pati and Ujjwal Baid and Maximilian Zenk and Brandon Edwards and Micah Sheller and G. Anthony Reina and Patrick Foley and Alexey Gruzdev and Jason Martin and Shadi Albarqouni and Yong Chen and Russell Taki Shinohara and Annika Reinke and David Zimmerer and John B. Freymann and Justin S. Kirby and Christos Davatzikos and Rivka R. Colen and Aikaterini Kotrotsou and Daniel Marcus and Mikhail Milchenko and Arash Nazer and Hassan Fathallah-Shaykh and Roland Wiest Andras Jakab and Marc-Andre Weber and Abhishek Mahajan and Lena Maier-Hein and Jens Kleesiek and Bjoern Menze and Klaus Maier-Hein and Spyridon Bakas},
      year={2021},
      eprint={2105.05874},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```
