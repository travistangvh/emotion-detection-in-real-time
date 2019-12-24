# Face Classification 
The following repository is a real-time face detection and emotion classification model.
[![Demo CountPages alpha](https://share.gifyoutube.com/KzB6Gb.gif)]

The face detection is powered by MTCNN and openCV. The emotion classification model is a built on an CNN architecture called VGGFace with weights trained on the fer2013 dataset.

#The Model
The model is trained on a CNN architecture called VGGFace. 
[![Trained VGGFace](https://doc-08-04-docs.googleusercontent.com/docs/securesc/24pgih7v62fnbhccaab3r6n993ftsk5r/4knhe0gv3cq017nudrc77965arih41rf/1577181600000/18352591164129673512/18352591164129673512/1_pU1QOF4VwVE3qWS-QXb0h0ZGDEVpnOC?e=view&authuser=0&nonce=kfi6vs2k6g208&user=18352591164129673512&hash=g121glo088rh4ea1pkad9fe5v2d0pisb)]

## Instructions on getting started
### To run the demo.
* Clone this commit to your local machine using `git clone https://github.com/travistangvh/emotion-detection-in-real-time.git'

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

