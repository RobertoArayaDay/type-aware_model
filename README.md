# Evaluation and Comparison of Fashion Recommendation Models

This work is based on the [PyTorch](http://pytorch.org/) implementation for the **Learning Type Aware Embeddings for Fashion Recommendation** [paper](https://arxiv.org/pdf/1803.09196.pdf).  


## Usage

You can download the Polyvore Outfits dataset including the splits and questions for the compatibility and fill-in-the-blank tasks from [here (6G)](https://drive.google.com/file/d/13-J4fAPZahauaGycw3j_YvbAHO7tOTW5/view?usp=sharing).


This work presents the implementation of the following models:

* Type-Aware Model
* CSA-Net
* CSA Fully-Connected Net
* CSA Roberta Net
* BYOL Dual-Net
* Simple Siamese Dual-Net

Each model is available for both training and testing the results in a recommendation system. In other words, it returns a set of clothing items that are compatible with an input image.
