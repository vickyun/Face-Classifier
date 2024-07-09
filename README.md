## Transfer Learning - Gender Classifier
This repository contains the code to perform experiments with Transfer Learning. 

As an example task I took Gender Classification. This kind of neural network might be interesting for e.g. 
retail analytics to get customer insights such as understanding gender demographics of in-store visitors; 
for monitoring public spaces for safety by analyzing gender distribution in crowds or analyzing user demographics 
for better content strategy.

### Input Data
The input data is taken from the MMLAB - [CelebA Dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

This dataset contains 202,599 face images of celebrities and corresponding 
annotations including landmarks and dozens of binary attribute's annotations 
per image such as gender, face type, etc.

You can download the [dataset](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg) and [annotations](https://drive.google.com/drive/folders/0B7EVK8r0v71pOC0wOVZlQnFfaGs?resourcekey=0-pEjrQoTrlbjZJO2UL8K_WQ) through the provided links.

Input data examples : ToDo

![](http://url/to/img.png)

### Exploratory Data Analysis 
EDA is available in Jupyter Notebook: `data/Gender_Classifier_EDA.ipynb`

### Requirements 
- python: 3.8
- pandas:  2.1.4
- mlflow: 2.10.2
- torch: 1.12.1
- torchvision: 0.13.1
- cv2: 4.6.0
- seaborn: 0.12.1
- matplotlib: 3.8.3

Install dependencies: `pip install requirements.txt`

### Run the training script

To run train script it is necessary to have input data folders:
`data/input/train` and `data/input/val` containing a folder with images per each class.
(see the Data Splitting part in `Gender_Classifier_EDA.ipynb`)

Similarly for inference script: it is necessary to have test image data folders:
`data/input/test` and `data/input/test_arbitrary`.

#### Run the train script
`python train.py`

#### Run the inference script
`python inference.py`


### Parameters

Use `utils.constant` module to modify training parameters such as:

* Number of epochs 
* Model Learning Rate
* Momentum of Optimizer
* Random seed for reproducibility
* Device, e.g. cpu or gpu

and others:
* Input data path
* Run name for mlflow tracking
* Checkpoints output path

### Checkpoints 
Model checkpoints for ResNet-18 is available in `checkpoints`
