import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[0:1], 'GPU')
from vocab.vocab import Vocab
from dataset import create_dataset
from configs.config import Config
import sys
from featurizers.speech_featurizers import TFSpeechFeaturizer, NumpySpeechFeaturizer
from models.model import MulSpeechLR as Model
import librosa


weights_dir = './saved_weights/20230228-084356/'
config_file = weights_dir + 'config.yml'
model_file = weights_dir + 'last/model'
vocab_file = weights_dir + 'vocab.txt'
config = Config(config_file)
speech_featurizer = TFSpeechFeaturizer(config.speech_config)
lr_vocab = Vocab(vocab_file)
lr_model = Model(**config.model_config, vocab_size=len(lr_vocab.token_list))
lr_model.load_weights(model_file)
lr_model.add_featurizers(speech_featurizer)
lr_model.init_build([None, config.speech_config['num_feature_bins']])
lr_model.summary()


def predict_wav(wav_path):
    sample_rate = 16000
    signal, _ = librosa.load(wav_path, sr=sample_rate)
    predict, prob = lr_model.predict_pb(signal)
    language = lr_vocab.token_list[predict.numpy()]
    print("predict language={}  prob={:.4f}".format(language, prob.numpy()*100))

if __name__ == '__main__':
    wav_path = sys.argv[1]
    predict_wav(wav_path)