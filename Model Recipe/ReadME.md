## How can you make costom dataset and how to trian it 

* ### To make contom dataset you can reffer to [this repo](https://github.com/kinivi/hand-gesture-recognition-mediapipe).
    * After runing downloading the repo create a csv file and change the csv file ref. in 285 line in app.py file.
    * Run app.py or you can use my version of app.py which  included z co-ordinate also in training datasest.Also don't forget to change 329 line for csv file ref.
    * Press k to go in dataset creating mode.
    * Press any key (0-9) to add a datapoint in dataset
> Remarks - remember to make a balence dataset.
* ### To train the model
    * You can use same repo to train the data using tensorflow.
    * you can also used my model which is in Pytorch(model.ipynb).

---

### gesture_recognizer.py 
It is a file that can only detect the 21 joints points using `MediaPipe` library.

---