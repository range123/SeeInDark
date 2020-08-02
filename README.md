# See In The Dark

This project is used to brighten dark (low light) images and allow us to observe the objects and details in the low light images.
Uses a fully convolutional neural network (contains convolution and transposed convolution layers) for brightening the images. This repo contains a flask app where the model is deployed. 
The model is trained on the "See In The dark" [sony](https://storage.googleapis.com/isl-datasets/SID/Sony.zip) dataset and based on [Learning-to-See-in-the-Dark](https://github.com/cchen156/Learning-to-See-in-the-Dark).

## Key Features 
- The network is relatively small and runs interactively on edge devices.
- Works well on even almost completely dark images.

## Prerequisites
 - Python3.6+
 - pip3/conda

## Installation
- Clone this repository.
- (optional) Create a virtual environment.
- Install the requirements.
```sh
pip install -r requirements.txt
```
 - Run the flask app
 ```sh
 export FLASK_APP=app.py
flask run
```

## Screenshots

![MainPage](https://www.github.com/range123/SeeInDark/blob/master/screenshots/1.png)
![Results](https://www.github.com/range123/SeeInDark/blob/master/screenshots/2.png)



