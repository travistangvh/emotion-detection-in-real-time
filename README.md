# Face Classification 
The following repository is a real-time face detection and emotion classification model.

<p align="center">
    <img width="320" height="240" src="https://raw.githubusercontent.com/travistangvh/emotion-detection-in-real-time/master/images/demo1.gif">
    <img width="320" height="240" src="https://raw.githubusercontent.com/travistangvh/emotion-detection-in-real-time/master/images/demo2.gif">
</p>


The face detection is powered by MTCNN and openCV. The emotion classification model is a built on an CNN architecture called VGGFace with weights trained on the fer2013 dataset.

## The Model
The model is trained on a CNN architecture called VGGFace. 


<p align="center">
    <img src="https://raw.githubusercontent.com/travistangvh/emotion-detection-in-real-time/master/images/VGGFaceNetwork.jpg">
</p>


## Instructions on getting started
### To run the demo.
* Clone this commit to your local machine using `git clone https://github.com/travistangvh/emotion-detection-in-real-time.git`

* Install these dependencies with pip install 
`pip install -r ../REQUIREMENTS.txt`

* Download pretrained model and weight `trained_vggface.h5` from [here](https://drive.google.com/file/d/1Wv_Z4lAa7BgYqSAeceK9TxJNfwoLTwKy/view?usp=sharing).

* Place `trained_vggface.h5` into `../datasets/trained_models/`.

* Run `emotion_webcam_demo.py` using `python3 emotion_webcam_demo.py`

### To train previous/new models for emotion classification:

* Download the fer2013.tar.gz file from [here](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)

* Move the downloaded file to the `../datasets/raw/` directory inside this repository.

* Untar the file:
`tar -xzf fer2013.tar`

* Ensure that the file `../datasets/raw/fer2013.csv` exists

* Run the `training_emotion_classification.py` file
`python3 training_emotion_classifier.py`

# Citations
* [Deep Face Recognition](http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf) by Parkhi et. al.

