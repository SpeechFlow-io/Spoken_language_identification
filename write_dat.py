""" Read list.txt and generate .dat & .txt file as DAT format
input:  (read /mnt/sd1/aishell_train.txt)
    data_path  (/mnt/sd1/)
    corpus_name  (aishell)
    date_type  (train)

output:  (write /mnt/sd1/aishell/train/1.txt, /mnt/sd1/aishell/train/1.dat)
    data_path/corpus_nama/date_type/*.dat
    data_path/corpus_nama/date_type/*.txt

Example line of .txt:
    data_aishell/wav/test/S0764/BAC009S0764W0121.wav:0:33680	chinese
"""
import librosa
from tqdm import tqdm
from featurizers.speech_featurizers import NumpySpeechFeaturizer


speech_config = {
    "sample_rate": 16000,
    "frame_ms": 25,
    "stride_ms": 10,
    "num_feature_bins": 80,
    "feature_type": "log_mel_spectrogram",
    "preemphasis": 0.97,
    "normalize_signal": True,
    "normalize_feature": True,
    "normalize_per_feature": False
}
speech_featurizer = NumpySpeechFeaturizer(speech_config)


def gen_dat_file(file_id, src_file_path, src_wav_dir, dst_dir_path, dst_file_name):
    print('start reading file',src_file_path + str(file_id))
    fin = open(src_file_path + str(file_id), 'r')
    fout_txt = open(dst_dir_path + dst_file_name + '-' + str(file_id) + '.txt', 'w')
    fout_dat = open(dst_dir_path + dst_file_name + '-' + str(file_id) + '.dat', 'wb')
    
    start = 0
    lines = fin.readlines()
    for line in tqdm(lines):
        try:
            wav, txt = line.strip().split(' ')
            audio_path = src_wav_dir + wav
            signal, _ = librosa.load(audio_path, sr=16000)
            # if len(signal) < 1 * 16000 and len(signal) > 12*16000: continue
            feature = speech_featurizer.extract(signal)
        except Exception as e:
            print(e)
            continue
        end = start + feature.shape[0] * feature.shape[1]
        fbank = feature.tobytes()
        fout_dat.write(fbank)
        fout_txt.write(wav + ':' + str(start) + ':' + str(end) + '\t' + txt + '\n')
        start = end
    
    fout_dat.flush()
    fout_txt.flush()
    fout_dat.close()
    fout_txt.close()


if __name__ == "__main__":
    src_file_path='./data/txt/demo.txt'    
    src_wav_dir='./data/wavs/'
    dst_dir_path='./data/dat/'
    dst_file_name='demo_train'

    start_id = 1
    end_id = 1
    for i in range(start_id, end_id + 1, 1):
        gen_dat_file(file_id=i, 
                    src_file_path=src_file_path,
                    src_wav_dir=src_wav_dir,
                    dst_dir_path=dst_dir_path,
                    dst_file_name=dst_file_name)

    print('done')