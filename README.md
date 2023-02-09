My Master Thesis focused on overcoming class imbalance and limited training data in COVID-19 datasets by developing a novel data augmentation method based on 
Conditional Generative Adversarial Networks. This cutting-edge machine learning approach generated new, high-quality Chest X-ray images, 
which were then added to the dataset to ensure equal representation of each class. This method offers higher variability and avoids altering image semantics, 
making it a preferred solution over existing methods for resolving class imbalance and limited training data.

This repository contains the code used to evaluate and compare existing augmentation methods with the proposed cGAN-based method. The cGANs were trained with
[StudioGAN](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN) and [FastGAN](https://github.com/lucidrains/lightweight-gan), on the dataset [Covid19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database).