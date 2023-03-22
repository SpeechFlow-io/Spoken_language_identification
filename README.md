<div align=center><img width='800' height='210' src='https://github.com/SpeechFlow-io/Spoken_language_identification/blob/main/speechflow.jpg'></div>

[**SpeechFlow**](https://speechflow.io/?ref=github) is an advanced speech-to-text API that offers exceptional accuracy for businesses of all sizes and industries. With SpeechFlow, users can transcribe audio and video content into text with high precision, making it an ideal solution for companies that need to quickly and accurately convert speech into text for various purposes, such as captioning, transcription, and analysis. With support for multiple languages and dialects, SpeechFlow is a versatile tool that can cater to a wide range of businesses and industries.

 # Spoken_language_identification 
* [Objective](#objective)
* [Technology](#technology)
* [Available models and languages](#available-models-and-languages)
* [Environment Setup](#environment-setup)
* [Code Implementation](#code-implementation)
   * [Audio Format](#audio-format)
   * [Features](#features)
   * [Prepare your input data](#prepare-your-input-data)
   *  [Train model](#train-model)
   *  [Inference](#inference)
* [ LICENSE](#license)

## Objective 
Spoken Language Identification (LID) is defined as detecting language from an audio clip by an unknown speaker, regardless of gender, manner of speaking, and distinct age speaker. It has numerous applications in speech recognition, multilingual machine translations, and speech-to-speech translations. 

Our model currently supports 13 languages: English, Spanish, Italian, French, German, Portuguese, Russian, Turkish, Vietnamese, Indonesian, Chinese, Japanese, and Korean.

## Technology
The model uses convolutional and recurrent neural networks trained on two thousands of hours of speech data(private). Approximately 150 hours of speech supervision per language.

<img width='400' height='600' src='https://github.com/SpeechFlow-io/Spoken_language_identification/blob/main/network.png'><br/>

## Available models and languages
 The figure below shows a ACC (Accuracy) breakdown by languages of the [FLEURS](https://arxiv.org/pdf/2205.12446.pdf) test-set using pretrained model.</br>
**FLEURS dataset** downloads can be fount here: [Downloads](https://www.tensorflow.org/datasets/catalog/xtreme_s#xtreme_sfleurstr_tr)
![](https://github.com/zhong-ying-china/Multi-Spoken-language-recognition/blob/main/fleurs.jpg)
     
## Environment Setup
The models are implemented in TensorFlow.
To use all of the functionality of the library, you should have:</br>
**tensorflow==2.4.1</br>
tensorflow-gpu==2.4.1</br>
tensorflow-addons==0.15.0</br>
matplotlib==3.5.0</br>
numpy==1.19.5</br>
scikit-learn==1.0.1</br>
librosa==0.8.1</br>
SoundFile==0.10.3.post1</br>
PyYAML==6.0**</br>

Download the codebase and open up a terminal in the root directory. Make sure python 3.7 is installed in the current environment. Then execute
```
pip install -r requirements.txt
```

## Code Implementation
### **Audio Format** 
The wav files have 16KHz sampling rate, single channel, and 16-bit Signed Integer PCM encoding.

### **Features** 
As speech features, 80-dimensional log mel-filterbank outputs were computed from 25ms window for each 10ms. Those log mel-filterbank features were further normalized to have zero mean and unit variance over the training partition of the dataset.

### **Prepare your input data**
You must prepare your own data before training the model, refer to 'data/demo_txt/demo_train.txt' file.

### **Train model**
To get start, please config 'congfigs/config.yml' file,  and simple run this command in the console:

```
python train.py
```
This will train Spoken_language_identification model by data in the 'data/demo_txt/demo_train.txt', then store the model on saved_weights folder, perform inference on 'demo_txt/demo_test.txt', print the inference results, and save the averaged accuracy in a text file.
### **Inference**
[![ Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16-Nre8aDvn0wN2dsgGa3xUsZ7S61e1h8#scrollTo=Is60zUMuPqSi)

The pretrained model is provided in this [project](https://github.com/SpeechFlow-io/Spoken_language_identification/tree/main/saved_weights/20230228-084356), simple run this command:
```
python predict_by_pb.py test_audios/chinese.wav
```
or
```
python predict_by_weights.py test_audios/chinese.wav
```

The provided chinese.wav audio needs to meet the [Audio Format](#audio-format), if your audio file is not wav format(eg: mp3), you can convert the audio to wav format by ffmpeg. Run the following command in your audio directory convert  to wav format.
```
ffmpeg -i audio.mp3 -ab 256k -ar 16000 -ac 1 -f wav audio.wav
```
If you don't have installed ffmpeg, please installed it first.
```
sudo apt-get update
sudo apt-get install ffmpeg
```
## LICENSE
Spoken_language_identification is released under the Apache License, version 2.0. The Apache license is a popular BSD-like license. Spoken_language_identification can be redistributed for free, even for commercial purposes, although you can not take off the license headers (and under some circumstances, you may have to distribute a license document).
