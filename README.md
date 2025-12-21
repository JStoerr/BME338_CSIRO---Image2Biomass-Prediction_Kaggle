# Image2Biomass – CSIRO Kaggle Competition - Authors Julian Stoerr & David Hesse

Course: BME338(UZH, HS25)

This repository contains a python based Machine Learning Project for participating in the CSIRO Image2Biomass Prediction Kaggle Competition for the BME338 Course Project. The goal of this project was to explore Feature extraction using a Vision Transformer (ViT) and create a simple MLP for predicting 5 different Biomass values.

## Project Structure
├── data/
│ ├── train/
│ ├── test/
│ ├── derived/
│ │ ├── dino_embeddings.pkl
│ │ └── dino_subm_embeddings.pkl
│ ├── submission/
│ │ └── submission.csv
│ ├── train.csv
│ ├── test.csv
│ └── sample_submission.csv
│
├── scripts/
│ ├── models/
│ │ └── mlp.py
│ ├── Feature_Extraction.ipynb
│ ├── MLP_build.ipynb
│ ├── Submission.ipynb
│ └── Overview.ipynb
│
├── documentation/
│ └── BME338_Project_Presentation.pdf
│
├── environment.yml
├── requirements.txt
├── Competition_Overview.md
├── README.md
└── .gitignore


### Data Availability

Data can be found in the data folder as well as on https://www.kaggle.com/competitions/csiro-biomass

### Workflow

The workflow includes

- EDA and Overview of Tabular data
- UMAP for further EDA
- Feature Extraction using the small DINOv2 ViT
- Pytorch Multi-Layer Perceptron build for analysing the extracted features

#### Requirements 

Found in the environments.yml and environment.txt files for pip and conda

### License and Usage

- This repository is for educational and research purposes only and only uses openly available models, code packages and open data published by Kaggle. For further data information please visit the corresponding Kaggle Competition.
