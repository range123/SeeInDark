#/mnt/c/Users/aashi/Desktop/Flaskdev/flaskenv/bin/env/python3.7
from flask import Flask,render_template,session,redirect,url_for,request
#from flask_wtf import FlaskForm
#from wtforms import StringField, TextField, SubmitField
#from __future__ import print_function, division
import os
import torch
import pandas as pd
import requests
import cv2
#from skimage import io, transform
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
torch.set_printoptions(linewidth=120)
#import cv2
# Ignore warnings
import json
import warnings
warnings.filterwarnings("ignore")
import time
from flask import make_response
from functools import wraps, update_wrapper
from datetime import datetime

plt.ion()   # interactive mode

app = Flask(__name__)

def nocache(view):
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = make_response(view(*args, **kwargs))
        response.headers['Last-Modified'] = datetime.now()
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '-1'
        return response
        
    return update_wrapper(no_cache, view)

class NNetwork(nn.Module):
    def __init__(self):
        super(NNetwork,self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=4,kernel_size=3,padding=0)#16
        self.conv2 = nn.Conv2d(in_channels=4,out_channels=8,kernel_size=3,padding=0)
        
        self.deconv1 = nn.ConvTranspose2d(in_channels=8,out_channels=4,kernel_size=4)
        self.deconv2 = nn.ConvTranspose2d(in_channels=4,out_channels=3,kernel_size=4)
        
    
    def forward(self,t):
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t,kernel_size=3, stride = 1)
        
        t = self.conv2(t)
        t = F.relu(t)
        
        t = self.deconv1(t)
        t = F.relu(t)

        t = self.deconv2(t)
        t = F.relu(t)

        print("yoyo:",t.shape)
        
        return t


fcn = NNetwork().to('cpu')
fcn.load_state_dict(torch.load('./tiny_final.pt',map_location= torch.device('cpu')))

@app.route('/test', methods=['GET'])
@nocache
def test1():
	return (render_template("result.html",data={"ori":"./static/fileToUpload.jpg","pro":"./static/fileToUpload1.jpg"}))


@app.route('/', methods=['POST','GET'])
@nocache
def index():
	if request.method == 'GET':
		return render_template('file.html')	
	returnfile=request.files['fileToUpload']
	timeval = str(time.time()).replace('.','')

	returnfile.save("./static/"+timeval+".jpg")



	imgs=cv2.imread("./static/"+timeval+".jpg")

	in_image = np.array(imgs)
	height,width = in_image.shape[0],in_image.shape[1]
	in_image = cv2.resize(in_image,(512,512))
	in_image = np.moveaxis(in_image, -1, 0)

	dark_imgs=[]
	dark_imgs.append(in_image)
	out = fcn(torch.as_tensor(dark_imgs).float())
	out = out.detach().cpu().numpy()

	out_pr = np.moveaxis(out[0], -1, 0)
	out_pr = np.moveaxis(out_pr, -1, 0)
	print(out_pr.shape)
	print(height,width)
	out_pr = cv2.resize(out_pr,(width,height))

	cv2.imwrite("./static/"+timeval+"1.jpg",out_pr)

	# plt.imshow(out_pr.astype('int'))
	return render_template("result.html",data={"ori":"./static/"+timeval+".jpg","pro":"./static/"+timeval+"1.jpg"})


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


	

#if __name__=='__main__':
app.run("0.0.0.0",5001,debug=True)

