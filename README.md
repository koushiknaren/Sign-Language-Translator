# Sign Language Translator

This project provides a vision-based system to recognize isolated hand gestures for Sign Language using a Long Short-Term Memory (LSTM) network and MediaPipe. The system tracks holistic landmarks (pose, face, left hand, right hand) from a live webcam feed and predicts the performed gesture.

Currently, it is configured to recognize the following actions:
- `hello`
- `thanks`
- `iloveyou`

## Prerequisites

Make sure you have Python installed. You can install all required dependencies using:

```bash
pip install -r requirements.txt
```

## How to use

The project is split into three main components: data collection, training, and testing.

### 1. Collect Data
To train the LSTM network, you need a dataset. Run `collect_data.py` to record your own dataset via webcam.

```bash
python collect_data.py
```
* Instructions: The script will prompt you to start a new action and wait for 2 seconds. It will collect 30 sequences, each of 30 frames, for each of the predefined actions (`hello`, `thanks`, `iloveyou`). The data will be saved as numpy arrays `.npy` in the `Feature_Extraction` directory. Press `q` to quit early.

### 2. Train the Model
After collecting the data, run `train_model.py` to compile and train the LSTM network.

```bash
python train_model.py
```
* Note: This will load the `.npy` sequences, split them into training/testing sets, build the LSTM model using TensorFlow/Keras, and train for 100 epochs. It will output a confusion matrix and save the trained model weights as `lstm_model.h5`.

### 3. Real-Time Testing
Once the model is trained, you can launch the live webcam translator.

```bash
python real_time_test.py
```
* Instructions: Perform the gestures in front of the camera. The script will buffer 30 frames and feed the extracted keypoints into the LSTM model for prediction. The predicted word will appear on the top-left of the window along with colored probability bars. Press `q` to exit.

## Project Structure
- `utils.py`: Helper functions for MediaPipe landmark extraction and rendering.
- `collect_data.py`: Script to generate the dataset.
- `train_model.py`: Script to train the LSTM neural network.
- `real_time_test.py`: Script to test the model using a live video feed.
- `requirements.txt`: Python dependencies.
