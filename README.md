# multimodal-fusion
This repository contains codes of our some recent works aiming at multimodal fusion, including Divide, Conquer and Combine: Hierarchical Feature Fusion Network with Local and Global Perspectives for Multimodal Affective Computing, Locally Confined Modality Fusion Network With a Global Perspective for Multimodal Human Affective Computing, etc.

The data are originally released in https://github.com/A2Zadeh/CMU-MultimodalSDK and are finally provided in https://github.com/soujanyaporia/multimodal-sentiment-analysis. If you need to use these data, please cite their corresponding papers.

Belows are the detailed introductions of our fusion methods:
1. HFFN (Hierarchical Feature Fusion Network)
Some of the codes are borrowed from https://github.com/soujanyaporia/multimodal-sentiment-analysis. We thank very much for their sharing.

Some modifications have been made to obtain better performance (80.45 on MOSI) such that some details are different from the paper:

For raw datasets, please download them from: https://github.com/soujanyaporia/multimodal-sentiment-analysis/tree/master/dataset (you need to create a 'dataset' folder and place the downloaded data in it.)

To run the code: python mosi_acl.py      
We test the code with python2, and the framework is Keras. You can also change the hyperparameters.

If you need to use the codes, please cite our paper:

Mai, Sijie, Haifeng Hu, and Songlong Xing. "Divide, Conquer and Combine: Hierarchical Feature Fusion Network with Local and Global Perspectives for Multimodal Affective Computing." Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. 2019.

2. LMFN

If you need to use the codes, please cite our paper:

Mai, Sijie, Songlong Xing, and Haifeng Hu. "Locally Confined Modality Fusion Network With a Global Perspective for Multimodal Human Affective Computing." IEEE Transactions on Multimedia 22.1 (2019): 122-137.

3. ARGF

If you need to use the codes, please cite our paper:

Mai, Sijie, Haifeng Hu, and Songlong Xing. "Modality To Modality Translation: An Adversarial Representation Learning and Graph Fusion Network for Multimodal Fusion." AAAI-20 (2020).
