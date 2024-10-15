import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk
import numpy as np
import os
import librosa
from sklearn.ensemble import RandomForestClassifier
import joblib
import pyaudio
import threading
import time
import speech_recognition as sr
from textblob import TextBlob
from deep_translator import GoogleTranslator
from keras.models import load_model
import matplotlib.pyplot as plt

# Loading model
model1 = load_model("./modelim.h5")  # the model trained for identify emotions from real time record voice

model = joblib.load("./AhmetModeli2.pkl")  # the model trained for identify who is speaking from real time record voice

# Starting engines
recognizer = sr.Recognizer()
microphone = sr.Microphone()

# Persons list
melih_list = []
ozlem_list = []
ali_list = []
can_list = []

def analiz_et(cumle):
    # Translating sentences to english
    translated = GoogleTranslator(source='auto', target='en').translate(cumle)
    analiz = TextBlob(translated)
    return analiz.sentiment.polarity

def duygu_analizi(cumle):
    duygu_polaritesi = analiz_et(cumle)
    
    if duygu_polaritesi > 0: # if polarite bigger than 0 its means happy emotion
        if duygu_polaritesi > (2/3):
            sonuc = f"Çok mutlu, Duygu Yoğunluğu: %{duygu_polaritesi*100:.2f}"
        else:
            sonuc = f"Az az mutlu, Duygu Yoğunluğu: %{duygu_polaritesi*100:.2f}"
    elif duygu_polaritesi < 0: # if polarite fewer than 0 its means unhappy emotion
        if duygu_polaritesi < -(2/3):
            sonuc = f"Çok mutsuz, Duygu Yoğunluğu: %{abs(duygu_polaritesi)*100:.2f}"
        else:
            sonuc = f"Az mutsuz, Duygu Yoğunluğu: %{abs(duygu_polaritesi)*100:.2f}"
    else:
        sonuc = f"Nötr, Duygu Yoğunluğu: %{duygu_polaritesi*100:.2f}"
    
    return sonuc

# Extracting features for prediction
def extract_features(audio_data, sr):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(y=audio_data)
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio_data), sr=sr)
    
    features = np.concatenate((np.mean(mfccs.T, axis=0), 
                               np.mean(zcr.T, axis=0), 
                               np.mean(chroma.T, axis=0),
                               np.mean(spectral_contrast.T, axis=0),
                               np.mean(tonnetz.T, axis=0)))
    return features

# Record parameter
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 22050
CHUNK = 1024
DURATION = 3  # prediction interval in seconds

is_recording = False

# Making application window full screen
def set_window_center(root):
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.geometry(f"{screen_width}x{screen_height}")
    root.state('zoomed')  

def recognize_realtime_speech():
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        audio_data = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio_data, language="tr-TR")  # recognition in Turkish
            return text
        except sr.UnknownValueError:
            return "Anlaşılamadı"
        except sr.RequestError as e:
            return f"Sunucuya ulaşılamıyor; {e}"

def record_and_predict():
    global is_recording
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    while is_recording:
        frames = []
        start_time = time.time()
        while time.time() - start_time < DURATION:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)

        audio_data = np.frombuffer(b''.join(frames), dtype=np.float32)
        audio_data = librosa.util.normalize(audio_data)
        features = extract_features(audio_data, RATE).reshape(1, -1)
        if model.predict(features)[0]== "ali":
            prediction = "Ali"
        elif model.predict(features)[0]=="ozlem":
            prediction = "Özlem"
        elif model.predict(features)[0]=="can":
            prediction = "Can"
        elif model.predict(features)[0] =="melih":
            prediction = "Melih"
        

        recognized_text = recognize_realtime_speech() # Appending the words to the list which understand
        if prediction == "Melih":
            melih_list.append(recognized_text)
        elif prediction == "Özlem":
            ozlem_list.append(recognized_text)
        elif prediction == "Ali":
            ali_list.append(recognized_text)
        elif prediction == "Can":
            can_list.append(recognized_text)
        
        # Predict emotions
        prediction1 = model1.predict(features)[0]
        predicted_class = np.argmax(prediction1)
        
        # Printing prediction
        classes = ['angry', 'happy', 'neutral', 'sad']  
        print("Tahmin edilen sınıf:", classes[predicted_class])

        update_ui(prediction, classes[predicted_class]) # update ui

    stream.stop_stream()
    stream.close()
    p.terminate()
    reset_ui()  #  After recording stops reset UI
    show_results()  # Show the results after stop


def start_recording():
    global is_recording
    if is_recording:
        return
    is_recording = True
    threading.Thread(target=record_and_predict).start()

def stop_recording():
    global is_recording
    is_recording = False

def update_ui(prediction,emotion):  # Showing emotion prediction on UI
    emotion_label.config(text=f"Tahmin edilen duygu: {emotion}")
    for name, widgets in profile_widgets.items():
        frame, name_label, record_label = widgets
        if name == prediction:
            frame.config(borderwidth=2, relief="solid", style="Highlighted.TFrame")
            name_label.config(foreground="white", background="#1874cd")
            record_label.config(image=record_photo, background="#83FFFD")
        else:
            frame.config(borderwidth=0, relief="flat", style="Profile.TFrame")
            name_label.config(foreground="black", background="#00868b")
            record_label.config(image=empty_photo, background="#00868b")
    

def reset_ui(): # Reseting UI to begining
    for name, widgets in profile_widgets.items(): 
        frame, name_label, record_label = widgets
        frame.config(borderwidth=0, relief="flat", style="Profile.TFrame")
        name_label.config(foreground="black", background="#00868b")
        record_label.config(image=empty_photo, background="#00868b")

def show_results():  # How many words they spoke
    melih_word_count = sum(len(sentence.split()) for sentence in melih_list if sentence != "Anlaşılamadı")
    ozlem_word_count = sum(len(sentence.split()) for sentence in ozlem_list if sentence != "Anlaşılamadı")
    ali_word_count = sum(len(sentence.split()) for sentence in ali_list if sentence != "Anlaşılamadı")
    can_word_count = sum(len(sentence.split()) for sentence in can_list if sentence != "Anlaşılamadı")

    total_word_count = melih_word_count + ozlem_word_count + ali_word_count + can_word_count # Total count of words which understood

    # Calculating percentage of speaking by diveding understood words to total words
    melih_percentage = (melih_word_count / total_word_count) * 100 if total_word_count != 0 else 0
    ozlem_percentage = (ozlem_word_count / total_word_count) * 100 if total_word_count != 0 else 0
    ali_percentage = (ali_word_count / total_word_count) * 100 if total_word_count != 0 else 0
    can_percentage = (can_word_count / total_word_count) * 100 if total_word_count != 0 else 0

    #Printing results
    result_text = f"Melih ({melih_word_count} kelime, %{melih_percentage:.2f}):{' '.join(melih_list)}\n\n"
    result_text += duygu_analizi(' '.join(melih_list))+"\n\n\n"

    result_text += f"Özlem ({ozlem_word_count} kelime, %{ozlem_percentage:.2f}): {' '.join(ozlem_list)}\n\n\n"
    result_text += duygu_analizi(' '.join(ozlem_list))+"\n\n\n"

    result_text += f"Ali ({ali_word_count} kelime, %{ali_percentage:.2f}):{' '.join(ali_list)}\n\n"
    result_text += duygu_analizi(' '.join(ali_list))+"\n\n\n"

    result_text += f"Can ({can_word_count} kelime, %{can_percentage:.2f}):{' '.join(can_list)}\n\n"
    result_text += duygu_analizi(' '.join(can_list))+"\n\n\n"

    result_text += f"Toplam kelime sayısı: {total_word_count}"

    result_window = tk.Toplevel(root)
    result_window.title("Sonuçlar")
    result_textbox = ScrolledText(result_window, wrap=tk.WORD, width=100, height=30)
    result_textbox.pack(expand=True, fill=tk.BOTH)
    result_textbox.insert(tk.END, result_text)
    result_textbox.config(state=tk.DISABLED)

    names = ["Melih", "Özlem", "Ali", "Can"]
    word_counts = [melih_word_count, ozlem_word_count, ali_word_count, can_word_count]
    
    # Showing percentage in piechart
    plt.figure(figsize=(8, 8))
    plt.pie(word_counts, labels=names, autopct='%1.1f%%', startangle=140)
    plt.axis('equal') 
    plt.title('Kişilere Göre Kelime Sayısı')
    plt.show()

    melih_list.clear()
    ozlem_list.clear()
    ali_list.clear()
    can_list.clear()

root = tk.Tk()
root.title("Ses Tanımlama Projesi")
set_window_center(root)
root.configure(background="#00868b")

# Defining style
style = ttk.Style()
style.configure("Normal.TFrame")
style.configure("Highlighted.TFrame", background="#1874cd")
style.configure("Red.TFrame", background="#00868b")
style.configure("Profile.TFrame", background="#00868b")

profiles = [ # Profile picture of persons
    ("./melih.jpg", "Melih"), 
    ("./ozlem.jpg", "Özlem"),
    ("./ali.jpg", "Ali"),
    ("./Can.jpg", "Can")
]
# Adjusting frame
profile_frame = ttk.Frame(root, style="Red.TFrame")
profile_frame.pack(pady=150)  
profile_frame.config(borderwidth=2)

profile_widgets = {}

for path, name in profiles:
    sub_frame = ttk.Frame(profile_frame, style="Profile.TFrame")
    sub_frame.pack(side=tk.LEFT, padx=50, pady=40)

    img = Image.open(path).resize((150, 150))
    photo = ImageTk.PhotoImage(img)

    label = tk.Label(sub_frame, image=photo, background="white")
    label.image = photo
    label.pack()

    name_label = tk.Label(sub_frame, text=name, foreground="black", font=("Arial", 18, "bold"), background="#00868b")
    name_label.pack()

    empty_image = Image.new('RGBA', (45, 45), (255, 255, 255, 0))
    empty_photo = ImageTk.PhotoImage(empty_image)

    record_icon = Image.open("./record_icon3.png").resize((100, 100))
    record_photo = ImageTk.PhotoImage(record_icon)

    record_label = tk.Label(sub_frame, image=empty_photo, background="#00868b")
    record_label.image = empty_photo
    record_label.pack()

    profile_widgets[name] = (sub_frame, name_label, record_label)

emotion_label = tk.Label(root, text="Merhaba", foreground="white", font=("Arial", 14), background="#00868b")
emotion_label.pack(pady=(20, 0))  # This will show real time predictions of emotions

button_frame = ttk.Frame(root, width=800, height=50, style="Red.TFrame")
button_frame.pack(pady=(10, 20)) 

record_img = Image.open("./play-button.png").resize((100, 90))  
stop_img = Image.open("./pause-button.png").resize((100, 90))  

record_icon = ImageTk.PhotoImage(record_img)
stop_icon = ImageTk.PhotoImage(stop_img)

record_button = ttk.Button(button_frame, image=record_icon, command=start_recording, cursor="hand2")
record_button.grid(row=0, column=0, padx=20, pady=20)  

stop_button = ttk.Button(button_frame, image=stop_icon, command=stop_recording, cursor="hand2")
stop_button.grid(row=0, column=1, padx=20, pady=20)  

button_frame.grid_columnconfigure(0, weight=1)
button_frame.grid_columnconfigure(1, weight=1)

root.mainloop()
