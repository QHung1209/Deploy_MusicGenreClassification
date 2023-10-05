from flask import Flask, render_template, request
import pickle
import librosa
import numpy as np
app = Flask(__name__)

scaler = pickle.load(open("models\Transform.pkl", 'rb'))
clf = pickle.load(open("models/Classification.pkl", 'rb'))


@app.route("/")
def home():
    return render_template("index.html")


def getmetadata(filename):

    y, sr = librosa.load(filename)
    
    audio_length_samples = len(y)
    test_metadata = []

    # Độ dài của mỗi đoạn (3 giây)
    segment_length_samples = sr * 3

    # Tạo danh sách để chứa các đoạn âm thanh
    collection = []

    # Chia tệp âm thanh thành các đoạn
    start = 0
    while start < audio_length_samples:
        end = start + segment_length_samples
        if end > audio_length_samples:
            end = audio_length_samples
        segment = y[start:end]
        collection.append(segment)
        start = end
    for y in collection:
        # fetching tempo

        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)

        # fetching beats

        y_harmonic, y_percussive = librosa.effects.hpss(y)
        tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)

        # chroma_stft

        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)

        # rmse

        rmse = librosa.feature.rms(y=y)

        # fetching spectral centroid

        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

        # spectral bandwidth

        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)

        # fetching spectral rolloff

        spec_rolloff = librosa.feature.spectral_rolloff(y=y+0.01, sr=sr)[0]

        # zero crossing rate

        zero_crossing = librosa.feature.zero_crossing_rate(y)

        # mfcc

        mfcc = librosa.feature.mfcc(y=y, sr=sr)

        # metadata dictionary

        metadata_dict = [np.mean(chroma_stft), np.var(chroma_stft), np.mean(rmse), np.var(rmse),
                         np.mean(spec_centroid), np.var(spec_centroid),
                         np.mean(spec_bw),  np.var(spec_bw),
                         np.mean(spec_rolloff),  np.var(spec_rolloff),
                         np.mean(zero_crossing), np.var(zero_crossing), tempo]

        for i in range(1, 21):
            metadata_dict.extend(
                [np.mean(mfcc[i-1]), np.var(mfcc[i-1])])

        test_metadata.append(metadata_dict)
    return np.array(test_metadata)


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files['audiofile']
    audio_path = "./audio/" + file.filename
    file.save(audio_path)
    metadata = scaler.transform(getmetadata(audio_path))
    prediction = clf.predict(metadata)
    list_genre, counts = np.unique(prediction, return_counts=True)
    total_elements = len(metadata)
    result = list_genre[counts / total_elements >= 0.4]

    return render_template("index.html", prediction=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
