#Face Classification 

The following repository is a real-time face detection and emotion classification model. The face detection is powered by MTCNN and openCV. The emotion classification model is a built on an CNN architecture called VGGFace with weights trained on the fer2013 dataset. 

### To train previous/new models for emotion classification:

* Download the fer2013.tar.gz file from [here](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)

* Move the downloaded file to the `../datasets/raw/` directory inside this repository.

* Untar the file:
`tar -xzf fer2013.tar`

* Ensure that the file `../datasets/raw/fer2013.csv` exists

* Run the `training_emotion_classification.py` file
`python3 training_emotion_classifier.py`

