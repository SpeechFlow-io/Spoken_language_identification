from signal import signal
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[0:1], 'GPU')
from vocab.vocab import Vocab
import librosa
import numpy as np
import sys
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score


vocab = Vocab("vocab/vocab.txt")
model = tf.saved_model.load('saved_models/lang14/pb/2/')


def predict_wav(wav_path):
    signal, _ = librosa.load(wav_path, sr=16000)
    output, prob = model.predict_pb(signal)
    language = vocab.token_list[output.numpy()]
    print(language, prob.numpy()*100)

    return output.numpy(), prob.numpy()


if __name__ == '__main__':
    wav_path = sys.argv[1]
    predict_wav(wav_path)

