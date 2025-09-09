<a href="https://arxiv.org/abs/2105.05874" alt="Citation"><img src="https://img.shields.io/badge/cite-citation-blue" /></a>
<a href="https://twitter.com/FeTS_Challenge" alt="Citation"><img src="https://img.shields.io/twitter/follow/fets_challenge?style=social" /></a>

# Federated Tumor Segmentation Challenge

The repo for the FeTS Challenge: The 1st Computational Competition on Federated Learning.

## Quickstart

The official challenge website with detailed information on the challenge is:

https://www.synapse.org/#!Synapse:syn28546456

As the challenge is currently inactive, submitting algorithms is not possible. However, the challenge data is accessible [here](https://www.synapse.org/Synapse:syn54079892/wiki/626854). Please check the instructions there for downloading it and the conditions of use.

This repository complements the challenge website above by providing code for developing and testing algorithm submissions to the two task of the FeTS Challenge. Furthermore, it allows reproducing the results of the summarizing paper by providing source code and instructions.

### Task 1

The first task of the challenge involves customizing core functions of a baseline federated learning system implementation. The goal is to improve over the baseline consensus models in terms of robustness in final model scores to data heterogeneity across the simulated collaborators of the federation. For more details, please see [Task_1](./Task_1).

### Task 2

This task utilizes decentralized testing across various sites of the FeTS initiative in order to evaluate model submissions across data from different medical institutions, MRI scanners, image acquisition parameters and populations. The goal of this task is to find algorithms (by whatever training technique you wish to apply) that score well across these data. For more details, please see [Task_2](./Task_2).

### Source Code for the Paper Analysis

The code is located in [paper_analysis](./paper_analysis/). Please follow these instructions to reproduce the analysis from our paper:

1. Download the source data from the article [website](https://www.nature.com/articles/s41467-025-60466-1#Sec38).
1. Clone repository:
      ```
      git clone https://github.com/FETS-AI/Challenge.git
      cd Challenge/paper_analysis
      ```
1. Prepare the python environment (_note:_ only tested with python version 3.10):
      ```
      conda create -y -n fets_analysis python=3.10
      conda activate fets_analysis
      pip install -r requirements.txt
      ```
1. Run the analysis script
      ```
      python make_main_figures.py /path/to/sourcedata/directory /path/to/output/directory
      ```

This should produce the rankings and figures from the main article. If you are interested in the figures from the supplementary material, feel free to contact us!

## Challenge documentation and Q&A

Please visit the [challenge website](https://synapse.org/fets) and [forum](https://www.synapse.org/#!Synapse:syn28546456/discussion/default).

## Citation

Please cite [this paper](https://www.nature.com/articles/s41467-025-60466-1) when using the data:

```latex

@article{zenk_towards_2025,
	title = {Towards fair decentralized benchmarking of healthcare {AI} algorithms with the {Federated} {Tumor} {Segmentation} ({FeTS}) challenge},
	volume = {16},
	copyright = {2025 The Author(s)},
	issn = {2041-1723},
	url = {https://www.nature.com/articles/s41467-025-60466-1},
	doi = {10.1038/s41467-025-60466-1},
	language = {en},
	number = {1},
	urldate = {2025-09-09},
	journal = {Nature Communications},
	author = {Zenk, Maximilian and Baid, Ujjwal and Pati, Sarthak and Linardos, Akis and Edwards, Brandon and Sheller, Micah and Foley, Patrick and Aristizabal, Alejandro and Zimmerer, David and Gruzdev, Alexey and Martin, Jason and Shinohara, Russell T. and Reinke, Annika and Isensee, Fabian and Parampottupadam, Santhosh and Parekh, Kaushal and Floca, Ralf and Kassem, Hasan and Baheti, Bhakti and Thakur, Siddhesh and Chung, Verena and Kushibar, Kaisar and Lekadir, Karim and Jiang, Meirui and Yin, Youtan and Yang, Hongzheng and Liu, Quande and Chen, Cheng and Dou, Qi and Heng, Pheng-Ann and Zhang, Xiaofan and Zhang, Shaoting and Khan, Muhammad Irfan and Azeem, Mohammad Ayyaz and Jafaritadi, Mojtaba and Alhoniemi, Esa and Kontio, Elina and Khan, Suleiman A. and Mächler, Leon and Ezhov, Ivan and Kofler, Florian and Shit, Suprosanna and Paetzold, Johannes C. and Loehr, Timo and Wiestler, Benedikt and Peiris, Himashi and Pawar, Kamlesh and Zhong, Shenjun and Chen, Zhaolin and Hayat, Munawar and Egan, Gary and Harandi, Mehrtash and Isik Polat, Ece and Polat, Gorkem and Kocyigit, Altan and Temizel, Alptekin and Tuladhar, Anup and Tyagi, Lakshay and Souza, Raissa and Forkert, Nils D. and Mouches, Pauline and Wilms, Matthias and Shambhat, Vishruth and Maurya, Akansh and Danannavar, Shubham Subhas and Kalla, Rohit and Anand, Vikas Kumar and Krishnamurthi, Ganapathy and Nalawade, Sahil and Ganesh, Chandan and Wagner, Ben and Reddy, Divya and Das, Yudhajit and Yu, Fang F. and Fei, Baowei and Madhuranthakam, Ananth J. and Maldjian, Joseph and Singh, Gaurav and Ren, Jianxun and Zhang, Wei and An, Ning and Hu, Qingyu and Zhang, Youjia and Zhou, Ying and Siomos, Vasilis and Tarroni, Giacomo and Passerrat-Palmbach, Jonathan and Rawat, Ambrish and Zizzo, Giulio and Kadhe, Swanand Ravindra and Epperlein, Jonathan P. and Braghin, Stefano and Wang, Yuan and Kanagavelu, Renuga and Wei, Qingsong and Yang, Yechao and Liu, Yong and Kotowski, Krzysztof and Adamski, Szymon and Machura, Bartosz and Malara, Wojciech and Zarudzki, Lukasz and Nalepa, Jakub and Shi, Yaying and Gao, Hongjian and Avestimehr, Salman and Yan, Yonghong and Akbar, Agus S. and Kondrateva, Ekaterina and Yang, Hua and Li, Zhaopei and Wu, Hung-Yu and Roth, Johannes and Saueressig, Camillo and Milesi, Alexandre and Nguyen, Quoc D. and Gruenhagen, Nathan J. and Huang, Tsung-Ming and Ma, Jun and Singh, Har Shwinder H. and Pan, Nai-Yu and Zhang, Dingwen and Zeineldin, Ramy A. and Futrega, Michal and Yuan, Yading and Conte, Gian Marco and Feng, Xue and Pham, Quan D. and Xia, Yong and Jiang, Zhifan and Luu, Huan Minh and Dobko, Mariia and Carré, Alexandre and Tuchinov, Bair and Mohy-ud-Din, Hassan and Alam, Saruar and Singh, Anup and Shah, Nameeta and Wang, Weichung and Sako, Chiharu and Bilello, Michel and Ghodasara, Satyam and Mohan, Suyash and Davatzikos, Christos and Calabrese, Evan and Rudie, Jeffrey and Villanueva-Meyer, Javier and Cha, Soonmee and Hess, Christopher and Mongan, John and Ingalhalikar, Madhura and Jadhav, Manali and Pandey, Umang and Saini, Jitender and Huang, Raymond Y. and Chang, Ken and To, Minh-Son and Bhardwaj, Sargam and Chong, Chee and Agzarian, Marc and Kozubek, Michal and Lux, Filip and Michálek, Jan and Matula, Petr and Ker{\textasciicircum}kovský, Miloš and Kopr{\textasciicircum}ivová, Tereza and Dostál, Marek and Vybíhal, Václav and Pinho, Marco C. and Holcomb, James and Metz, Marie and Jain, Rajan and Lee, Matthew D. and Lui, Yvonne W. and Tiwari, Pallavi and Verma, Ruchika and Bareja, Rohan and Yadav, Ipsa and Chen, Jonathan and Kumar, Neeraj and Gusev, Yuriy and Bhuvaneshwar, Krithika and Sayah, Anousheh and Bencheqroun, Camelia and Belouali, Anas and Madhavan, Subha and Colen, Rivka R. and Kotrotsou, Aikaterini and Vollmuth, Philipp and Brugnara, Gianluca and Preetha, Chandrakanth J. and Sahm, Felix and Bendszus, Martin and Wick, Wolfgang and Mahajan, Abhishek and Balaña, Carmen and Capellades, Jaume and Puig, Josep and Choi, Yoon Seong and Lee, Seung-Koo and Chang, Jong Hee and Ahn, Sung Soo and Shaykh, Hassan F. and Herrera-Trujillo, Alejandro and Trujillo, Maria and Escobar, William and Abello, Ana and Bernal, Jose and Gómez, Jhon and LaMontagne, Pamela and Marcus, Daniel S. and Milchenko, Mikhail and Nazeri, Arash and Landman, Bennett and Ramadass, Karthik and Xu, Kaiwen and Chotai, Silky and Chambless, Lola B. and Mistry, Akshitkumar and Thompson, Reid C. and Srinivasan, Ashok and Bapuraj, J. Rajiv and Rao, Arvind and Wang, Nicholas and Yoshiaki, Ota and Moritani, Toshio and Turk, Sevcan and Lee, Joonsang and Prabhudesai, Snehal and Garrett, John and Larson, Matthew and Jeraj, Robert and Li, Hongwei and Weiss, Tobias and Weller, Michael and Bink, Andrea and Pouymayou, Bertrand and Sharma, Sonam and Tseng, Tzu-Chi and Adabi, Saba and Xavier Falcão, Alexandre and Martins, Samuel B. and Teixeira, Bernardo C. A. and Sprenger, Flávia and Menotti, David and Lucio, Diego R. and Niclou, Simone P. and Keunen, Olivier and Hau, Ann-Christin and Pelaez, Enrique and Franco-Maldonado, Heydy and Loayza, Francis and Quevedo, Sebastian and McKinley, Richard and Slotboom, Johannes and Radojewski, Piotr and Meier, Raphael and Wiest, Roland and Trenkler, Johannes and Pichler, Josef and Necker, Georg and Haunschmidt, Andreas and Meckel, Stephan and Guevara, Pamela and Torche, Esteban and Mendoza, Cristobal and Vera, Franco and Ríos, Elvis and López, Eduardo and Velastin, Sergio A. and Choi, Joseph and Baek, Stephen and Kim, Yusung and Ismael, Heba and Allen, Bryan and Buatti, John M. and Zampakis, Peter and Panagiotopoulos, Vasileios and Tsiganos, Panagiotis and Alexiou, Sotiris and Haliassos, Ilias and Zacharaki, Evangelia I. and Moustakas, Konstantinos and Kalogeropoulou, Christina and Kardamakis, Dimitrios M. and Luo, Bing and Poisson, Laila M. and Wen, Ning and Vallières, Martin and Loutfi, Mahdi Ait Lhaj and Fortin, David and Lepage, Martin and Morón, Fanny and Mandel, Jacob and Shukla, Gaurav and Liem, Spencer and Alexandre, Gregory S. and Lombardo, Joseph and Palmer, Joshua D. and Flanders, Adam E. and Dicker, Adam P. and Ogbole, Godwin and Oyekunle, Dotun and Odafe-Oyibotha, Olubunmi and Osobu, Babatunde and Shu’aibu Hikima, Mustapha and Soneye, Mayowa and Dako, Farouk and Dorcas, Adeleye and Murcia, Derrick and Fu, Eric and Haas, Rourke and Thompson, John A. and Ormond, David Ryan and Currie, Stuart and Fatania, Kavi and Frood, Russell and Simpson, Amber L. and Peoples, Jacob J. and Hu, Ricky and Cutler, Danielle and Moraes, Fabio Y. and Tran, Anh and Hamghalam, Mohammad and Boss, Michael A. and Gimpel, James and Kattil Veettil, Deepak and Schmidt, Kendall and Cimino, Lisa and Price, Cynthia and Bialecki, Brian and Marella, Sailaja and Apgar, Charles and Jakab, Andras and Weber, Marc-André and Colak, Errol and Kleesiek, Jens and Freymann, John B. and Kirby, Justin S. and Maier-Hein, Lena and Albrecht, Jake and Mattson, Peter and Karargyris, Alexandros and Shah, Prashant and Menze, Bjoern and Maier-Hein, Klaus and Bakas, Spyridon},
	month = jul,
	year = {2025},
	note = {Publisher: Nature Publishing Group},
	keywords = {Cancer imaging, Cancer screening, Computational models, Machine learning},
	pages = {6274},
}
```
