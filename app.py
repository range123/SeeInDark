from flask import Flask,render_template,session,redirect,url_for,request
import os
import torch
import pandas as pd
import requests
import cv2
import numpy as np

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
torch.set_printoptions(linewidth=120)
import json
import warnings
warnings.filterwarnings("ignore")
import time
from flask import make_response
from functools import wraps, update_wrapper
from datetime import datetime
from network import NNetwork
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




fcn = NNetwork().to('cpu')
fcn.load_state_dict(torch.load('./models/tiny_final.pt',map_location= torch.device('cpu')))

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


	

if __name__=='__main__':
    app.run("0.0.0.0",5001,debug=True)

