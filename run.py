from flask import Flask, render_template, request, url_for, jsonify, make_response, send_file
import os, subprocess, time, shutil, os.path
from datetime import datetime, timedelta
import requests
from PIL import Image, ImageOps
import numpy as np
import io, re, random
from pydicom import dcmread
import win32gui, win32con
from pathlib import Path
import json
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from yolactcnn import eval1


# this hides the cmd window!
#https://stackoverflow.com/questions/764631/how-to-hide-console-window-in-python
#os.system("title webdicom")
#whnd = win32gui.FindWindowEx(None, None, None, "webdicom")
#if not (whnd == 0):
#  #the_program_to_hide = win32gui.GetForegroundWindow()
#    win32gui.ShowWindow(whnd , win32con.SW_HIDE)

app = Flask(__name__)

app.config['WADOServer'] = " http://hackathon.siim.org/dicomweb/"

#add your api key here
app.config['apikey'] = ""


def getImage(StudyUID, SeriesUID, InstanceUID):

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/34.0.1847.131 Safari/537.36',
        'Accept': 'image/png',
        'apikey': app.config['apikey']
    }
    
    session = requests.Session()
    session.headers.update(headers)
    retry = Retry(connect=1, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    url = app.config['WADOServer'] + "studies/" +  StudyUID+ "/series/"+ SeriesUID +"/instances/"+InstanceUID+"/frames/1/rendered"
    response = session.get(url, allow_redirects=True, timeout=5)
    desired_size = 512
    stream = io.BytesIO(response.content)
    #print(response.content)
    im = Image.open(stream).convert("RGBA")    
    # partially from: https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
    old_size = im.size  # old_size[0] is in (width, height) format
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    im = im.resize(new_size, Image.ANTIALIAS)
    # create a new image and paste the resized on it
    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size-new_size[0])//2,
                        (desired_size-new_size[1])//2))

    return new_im

### CORS section
@app.after_request
def after_request_func(response):
    origin = request.headers.get('Origin')
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Headers', 'x-csrf-token')
        response.headers.add('Access-Control-Allow-Methods',
                            'GET, POST, OPTIONS, PUT, PATCH, DELETE')
        if origin:
            response.headers.add('Access-Control-Allow-Origin', origin)
    else:
        response.headers.add('Access-Control-Allow-Credentials', 'true')
        if origin:
            response.headers.add('Access-Control-Allow-Origin', origin)

    return response
### end CORS section


@app.route("/")
def main():

    return render_template('index.html')

@app.route("/getimg", methods=['GET'])
def getimg():

    StudyUID = request.args.get('StudyUID')
    SeriesUID = request.args.get('SeriesUID')
    InstanceUID = request.args.get('InstanceUID')

    îm = getImage(StudyUID, SeriesUID, InstanceUID)

    print("inferencing the image now....")
    numpy_image = eval1.startinf(request.get_json(), np.array(îm), "yes")

    im2 = Image.fromarray(np.uint8(numpy_image)).convert('RGB')

    byteImgIO = io.BytesIO()
    im2.save(byteImgIO, "JPEG")
    byteImgIO.seek(0)

    return send_file(byteImgIO, mimetype='image/jpeg')


@app.route("/inferencedimage", methods=['GET'])
def inferencedimage():

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/34.0.1847.131 Safari/537.36',
        'Accept': 'application/dicom+json'
    }

    session = requests.Session()
    session.headers.update(headers)
    retry = Retry(connect=1, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    url = request.args.get('url')
    print(url)

    #response = requests.get(url)
    #a = session.get(url, allow_redirects=True, timeout=5)
    îm = Image.open(requests.get(url, stream=True).raw)
    #im = Image.open(io.BytesIO(a.content))
    #îm = Image.open(session.get(url, allow_redirects=True, timeout=5))

    numpy_image = eval1.startinf("-", np.array(îm), "yes")

    im2 = Image.fromarray(np.uint8(numpy_image)).convert('RGB')

    byteImgIO = io.BytesIO()
    im2.save(byteImgIO, "JPEG")
    byteImgIO.seek(0)

    return send_file(byteImgIO, mimetype='image/jpeg')

@app.route("/getlist", methods=['GET'])
def getlist():

    # set the time to look back for images
    try:
        delta = int(request.args.get('delta'))
    except:
        delta = 60

    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/34.0.1847.131 Safari/537.36',
        'Accept': 'application/dicom+json',
        'apikey': app.config['apikey']
    }

    session = requests.Session()
    session.headers.update(headers)
    retry = Retry(connect=1, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)


    #day = (datetime.now() - timedelta(minutes = delta)).strftime('%Y%m%d')
    #minutes = (datetime.now() - timedelta(minutes = delta)).strftime('%H%M')
    #print("Query from: ",day,  minutes)
    #url= app.config['WADOServer'] + "instances?StudyDate="+day+"&StudyTime="+minutes+"-&Modality=CR,DX"

    url= app.config['WADOServer'] + "instances?Modality=CR,DX"

    a = session.get(url, allow_redirects=True, timeout=5)
    
    return a.content.decode('utf-8-sig')

@app.route("/inference", methods=['POST'])
def inference():
    #print(" request: ", request.get_json() )
    return eval1.startinf(request.get_json(), np.array(getImage(request.get_json()['StudyUID'],request.get_json()['SeriesUID'],request.get_json()['InstanceUID'] )), "no")


@app.route("/opencase", methods=['GET'])
def opencase():
    file = 'C:\\Users\\Sanyi\\Desktop\\WPy64-3860\\projects\\SiiM\\viewer.vbs'
    os.system('"' + file + '"')

    return True


if __name__ == "__main__":
    app.run(debug= True)
