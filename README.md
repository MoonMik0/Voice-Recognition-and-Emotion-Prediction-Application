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

## Additional Details:
As part of our project, we first collected and prepared the necessary voice data for emotion analysis. Each developer's voice was recorded, and preprocessing steps were applied to ensure the data was in the correct format. For transcription, we utilized **Google Speech Recognition** to analyze the spoken words.

We created a custom model to detect the speaker’s tone of voice and used this model to predict emotions based on both the spoken words and the tone of voice. This functionality was integrated into a user-friendly application, which features two buttons:

- **Play Button**: Begins the recording session
- **Stop Button**: Stops the recording

During the recording, the system identifies the speaker, analyzes the words they use, performs emotion analysis on the speech content, and detects emotions based on the speaker's tone. Once the recording is stopped, all of this information is displayed to the user in a clear, interactive manner.
