from featurizers.speech_featurizers import SpeechFeaturizer
from configs.config import Config
from random import shuffle
import numpy as np
from vocab.vocab import Vocab
import os
import math
import librosa
import tensorflow as tf


def wav_padding(wav_data_lst, wav_max_len, fbank_dim):
        wav_lens = [len(data) for data in wav_data_lst]
        # input wav from 1200 frames down sample 8 times to 150 frames
        wav_lens = [math.ceil(x/8) for x in wav_lens]
        wav_lens = np.array(wav_lens)
        new_wav_data_lst = np.zeros((len(wav_data_lst), wav_max_len, fbank_dim))
        for i in range(len(wav_data_lst)):
            new_wav_data_lst[i, :wav_data_lst[i].shape[0], :] = wav_data_lst[i]
        return new_wav_data_lst, wav_lens


class DatDataSet:
    def __init__(self,
        batch_size,
        data_type,
        vocab: Vocab,
        speech_featurizer: SpeechFeaturizer,
        config: Config):
        self.batch_size = batch_size
        self.data_type = data_type
        self.vocab = vocab
        self.data_path =config.dataset_config['data_path']
        self.corpus_name = config.dataset_config['corpus_name']
        self.fbank_dim = config.speech_config['num_feature_bins']
        self.max_audio_length =config.dataset_config['max_audio_length']
        self.mel_banks = config.speech_config['num_feature_bins']
        self.file_nums = config.dataset_config['file_nums']
        self.language_classes = config.running_config['language_classes']
        self.suffix = config.dataset_config['suffix']
        self.READ_BUFFER_SIZE = 2 * 1024 * 1024 * 1024
        self.shuffle = True
        self.blank = 0
        self.source_init()
        
        
    def source_init(self):
        self.dat_file_list, self.txt_file_list = self.get_dat_txt_list(self.data_type)
        print('>>', self.data_type, 'load dat files:', len(self.dat_file_list))
        print('>>', self.data_type, 'load txt files:', len(self.txt_file_list))
        max_binary_file_size = max([os.path.getsize(dat) for dat in self.dat_file_list])
        print('>> max binary file size:', max_binary_file_size)
        # alloc a huge memory block
        self.feature_binary = np.zeros(max_binary_file_size // 4 + 1, np.float32)  


    def get_dat_txt_list(self, dir_name):
        corpus_dir = self.data_path+'/'+self.corpus_name + '/'
        print('!!', corpus_dir)
        file_lst = os.listdir(corpus_dir)
        txt_file_lst = []
        dat_file_lst = []

        for align_file in file_lst:
            if align_file.endswith(self.suffix):
                file_name = align_file[:-len(self.suffix)]
                dat_file = file_name + '.dat'
                if dir_name in file_name:
                    # if dir_name in ['dev', 'test']:
                    #     dat_file = dat_file.replace(dir_name, 'train')
                    dat_file_lst.append(corpus_dir + dat_file)
                    txt_file_lst.append(corpus_dir + align_file)
        print('*********',dir_name, txt_file_lst, dat_file_lst)          
        return dat_file_lst, txt_file_lst

    
    def load_dat_file(self, dat_file_path):
        f = open(dat_file_path, 'rb')
        pos = 0
        buf = f.read(self.READ_BUFFER_SIZE)
        while len(buf) > 0:
            nbuf = np.frombuffer(buf, np.float32)
            self.feature_binary[pos: pos + len(nbuf)] = nbuf
            pos += len(nbuf)
            buf = f.read(self.READ_BUFFER_SIZE)

            
    def get_batch(self):
        while 1:
            shuffle_did_list = [i for i in range(len(self.dat_file_list))]
            if self.shuffle:
                shuffle(shuffle_did_list)
            for did in shuffle_did_list:
                wav_lst = []
                label_lst = []
                self.load_dat_file(self.dat_file_list[did])
                txt_file = open(self.txt_file_list[did], 'r', encoding='utf8')
                utt_lines = txt_file.readlines()
                txt_lines = utt_lines
                if self.shuffle:
                    shuffle(txt_lines)
                # sort lines by wav len
                # txt_lines = sorted(
                #     txt_lines, 
                #     key=lambda line: int(line.split('\t')[0].split(':')[2]) - int(line.split('\t')[0].split(':')[1]), 
                #     reverse=False)
                for line in txt_lines:
                    wav_file, label = line.split('\t')
                    wav_lst.append(wav_file)
                    label_lst.append(label.strip('\n'))
                shuffle_list = [i for i in range(len(wav_lst) // self.batch_size)]
                if self.shuffle:
                    shuffle(shuffle_list)
                for i in shuffle_list:
                    begin = i * self.batch_size
                    end = begin + self.batch_size
                    sub_list = list(range(begin, end, 1))
                    # label batch
                    label_data_lst = [label_lst[index] for index in sub_list]
                    prediction = np.array(
                    [self.vocab.token_list.index(line) for
                     line in label_data_lst],
                    dtype=np.int32)

                    feature_lst = []
                    wav_path = []
                    get_next_batch = False
                    for index in sub_list:
                        # data_aishell/wav/test/S0764/BAC009S0764W0121.wav:0:33680	chinese
                        _, start, end = wav_lst[index].split(':')
                        feature = self.feature_binary[int(start): int(end)]
                        feature = np.reshape(feature, (-1, 80))
                        feature = feature[:self.max_audio_length, :]
                        feature_lst.append(feature)
                        wav_path.append(wav_lst[index])  
                    
                    if get_next_batch:
                        continue
                    features, input_length = wav_padding(feature_lst, self.max_audio_length, self.fbank_dim)
                                        
                    yield features, input_length, prediction    


class TxtDataSet:
    def __init__(self,
                batch_size,
                data_type,
                vocab,
                speech_featurizer: SpeechFeaturizer,
                config: Config
                ):
        self.batch_size = batch_size
        self.data_type = data_type
        self.vocab = vocab
        self.feature_extracter = speech_featurizer
        self.data_path = config.dataset_config['data_path']
        self.corpus_name = config.dataset_config['corpus_name']
        self.fbank_dim = config.speech_config['num_feature_bins']
        self.max_audio_length =config.dataset_config['max_audio_length']
        self.mel_banks = config.speech_config['num_feature_bins']
        self.file_nums = config.dataset_config['file_nums']
        self.data_length = config.dataset_config['data_length']
        self.shuffle = True
        self.sentence_list = []
        self.wav_lst = []
        self.label_lst = []
        self.max_sentence_length = 0
        self.source_init()

    def source_init(self):
        read_files = []
        if self.data_type == 'train':
            read_files.append(self.corpus_name + '_train.txt')
        elif self.data_type == 'dev':
            read_files.append(self.corpus_name + '_dev.txt')
        elif self.data_type == 'test':
            read_files.append(self.corpus_name + '_test.txt')
        print('data type:{} \n files:{}'.format(self.data_type, read_files))
        total_lines = 0
        for sub_file in read_files:
            with open(sub_file, 'r', encoding='utf8') as f:
                for line in f:
                    wav_file, label = line.split(' ', 1)
                    label = label.strip('\n').split()
                    
                    self.label_lst.append(label)
                    self.wav_lst.append(wav_file)
                    total_lines += 1
                    if self.data_length:
                        if total_lines == self.data_length:
                            break
                    if total_lines % 10000 == 0:
                        print('\rload', total_lines, end='', flush=True)
        
        if not self.data_length:
            self.wav_lst = self.wav_lst[:self.data_length]
            self.label_lst = self.label_lst[:self.data_length]
        print('number of', self.data_type, 'data:', len(self.wav_lst))


    def get_batch(self):
        shuffle_list = [i for i in range(len(self.wav_lst))]
        while 1:
            if self.shuffle:
                shuffle(shuffle_list)
            for i in range(len(self.wav_lst) // self.batch_size):
                begin = i * self.batch_size
                end = begin + self.batch_size
                sub_list = shuffle_list[begin:end]

                label_data_lst = [self.label_lst[index] for index in sub_list]
                prediction = np.array(
                    [self.vocab.token_list.index(line[0]) for
                     line in label_data_lst],
                    dtype=np.int32)
                feature_lst = []
                wav_path = []
                get_next_batch = False
                for index in sub_list:
                    # start = time.time()
                    audio, _ = librosa.load(self.data_path + self.wav_lst[index], sr=16000)
                    if len(audio) == 0:
                        get_next_batch = True
                        break
                    feature = self.feature_extracter.extract(audio)
              
                    feature_lst.append(feature)
                    wav_path.append(self.wav_lst[index])

                if get_next_batch:
                    continue  # get next batch

                features, input_length = wav_padding(feature_lst, self.max_audio_length, self.fbank_dim)
             
                yield features,input_length, prediction


def create_dataset(batch_size, load_type, data_type, speech_featurizer, config, vocab):
    """
    batch_size: global batch size
    data_type: the type of lode data, supports type: txt, dat()

    """
    if load_type == 'dat':
        dataset = DatDataSet(batch_size, data_type, vocab, speech_featurizer, config)
        dataset = tf.data.Dataset.from_generator(dataset.get_batch,
                                                output_types=(tf.float32, tf.int32, tf.int32),
                                                output_shapes=([None, None, config.speech_config['num_feature_bins']],
                                                                [None], [None]))
    elif load_type == 'txt':
        dataset = TxtDataSet(batch_size, data_type, vocab, speech_featurizer, config)
        dataset = tf.data.Dataset.from_generator(dataset.get_batch,
                                             output_types=(tf.float32, tf.int32, tf.int32),
                                             output_shapes=([None, None, config.speech_config['num_feature_bins']],
                                             [None], [None]))
    else:
        print('load_type must be dat or txt!!')
        return

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA.DATA
    dataset = dataset.with_options(options)
    return dataset