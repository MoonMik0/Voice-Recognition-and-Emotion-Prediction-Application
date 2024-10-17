# Voice Recognition and Emotion Analysis Project

In this project, we developed a real-time voice recognition and emotion analysis system using machine learning techniques. We recorded the voices of four developers, applied various preprocessing methods, and split the audio into smaller segments to create a large dataset. Leveraging this data, we built an AI model capable of distinguishing between the voices of different speakers in real time.

The system was optimized for real-time performance, allowing it to predict the speaker's identity as soon as the audio is detected. We also developed an interactive interface where the system identifies the speaker in real time and displays a speaker icon beneath the photo of the recognized individual.

In the second phase of the project, we focused on emotion detection. We recorded conversations where the developers expressed different emotions such as **happy**, **sad**, **neutral**, and **angry**. These recordings were preprocessed and used to train a machine learning model capable of identifying emotions from two different sources:

1. **Emotion Analysis Based on Words**: The system performs sentiment analysis on the words spoken by each speaker to determine the emotions being expressed based on the language used.
2. **Emotion Analysis Based on Tone of Voice**: A custom model analyzes the tone of the speaker's voice to detect emotional cues, independent of the actual words spoken.

This model was integrated into the interface, enabling real-time speaker identification along with both types of emotion analysis during the recording.

After the recording ends, the system generates a comprehensive report. The report includes:

- **Transcription** of what was said
- **Percentage of speaking time** for each speaker compared to others
- **Emotion analysis**, detailing the emotional distribution in each speaker’s conversation (from both words and tone of voice)

In addition, a **pie chart** is generated to visually represent the speaking time distribution for each speaker.

## Features:
- **Real-time voice recognition**
- **Emotion analysis based on words** (sentiment analysis)
- **Emotion analysis based on tone of voice**
- **Interactive interface** for speaker and emotion identification
- **Post-recording report** with transcription, speaking time, and emotion distribution analysis

## How to Use the Project

This project enables real-time voice recognition and emotion analysis using machine learning models. To use the application, you'll need to create and train your own models for speaker identification and emotion detection. Here’s how to get started:

### Step 1: Clone the Repository
First, clone this repository to your local machine:

```bash
git clone https://github.com/your-repo/voice-recognition-emotion-analysis.git
cd voice-recognition-emotion-analysis
```

### Step 2: Install the Dependencies

```bash
pip install numpy
pip install librosa
pip install pyaudio
pip install SpeechRecognition
pip install tensorflow
pip install matplotlib
pip install joblib
```
### Step 3: Train Your Models
The project requires two models:

1. Speaker Identification Model: A model to recognize who is speaking in real-time.
2. Emotion Detection Model: A model to identify emotions from both the spoken words and tone of voice.
Speaker Identification Model:
- You will need to collect voice samples from each individual you want the system to recognize. The voices should be preprocessed, and features extracted using librosa.
- Train a model (e.g., RandomForest, SVM) that takes these features as input and predicts the speaker.
- Save the trained model using joblib:

```bash
import joblib
joblib.dump(trained_model, 'your_speaker_model.pkl')
```

Emotion Detection Model:
- Record voice samples where different emotions (e.g., happy, sad, angry, neutral) are expressed. Use those samples to extract features and build an emotion classifier.
- You can train a neural network using Keras or any other framework. Here’s an example of how to save the model:

```bash
from keras.models import save_model
model.save('your_emotion_model.h5')
```

### Step 4: Replace Pretrained Models
After training your models, place them in the project folder and rename them as follows:

- The speaker identification model should be named AhmetModeli2.pkl.
- The emotion detection model should be named modelim.h5.

Alternatively, you can change the model paths in the code:

```bash
model1 = load_model("./modelim.h5")  # your emotion detection model
model = joblib.load("./AhmetModeli2.pkl")  # your speaker identification model
```

### Step 5: Run the Application
Once your models are in place, you can run the application using:

```bash
python app.py
```
This will open the real-time voice recognition and emotion analysis interface. The application will start listening for voice input and display the recognized speaker and emotion in real-time.

### Step 6: Record and Analyze
The interface has two buttons:

- Play Button: Starts recording and analyzing voices in real-time.
- Stop Button: Stops recording and shows the results, including:
  - Who spoke and for how long (word count and percentage)
  - Sentiment/emotion analysis based on both the spoken words and the tone of voice
  - A pie chart that shows the word distribution among the speakers
Additional Notes:
- Make sure your microphone is properly configured and has sufficient input quality for best results.
- You can expand the project by training on more speakers or adding more emotional states for emotion detection.
