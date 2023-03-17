# Spoken_language_identification
Tensorflow python speech
## Objective 
Spoken Language Identification (LID) is defined as detecting language from an audio clip by an unknown speaker, regardless of gender, manner of speaking, and distinct age speaker. It has numerous applications in speech recognition, multilingual machine translations, and speech-to-speech translations. 

Our model currently supports 13 languages: English, Spanish, Italian, French, German, Portuguese, Russian, Turkish, Vietnamese, Indonesian, Chinese, Japanese, and Korean.

## Technology
The model uses convolutional and recurrent neural networks trained on two thousands of hours of speech data(private). Approximately 150 hours of speech supervision per language.

<img width='400' height='600' src='https://github.com/SpeechFlow-io/Spoken_language_identification/blob/main/network.png'><br/>

## Available models and languages
 The figure below shows a ACC (Accuracy) breakdown by languages of the [**FLEURS dataset**](https://www.tensorflow.org/datasets/catalog/xtreme_s#xtreme_sfleurstr_tr) [(paper)](https://arxiv.org/pdf/2205.12446.pdf) test-set using pretrained model.

![](https://github.com/zhong-ying-china/Multi-Spoken-language-recognition/blob/main/fleurs.jpg)
     
## Environment Setup
Download the codebase and open up a terminal in the root directory. Make sure python 3.7 is installed in the current environment. Then execute
```
pip install -r requirements.txt
```
This should install all the necessary packages for the code to run.

## Code Implementation
### **Audio Format** 
The wav files have 16KHz sampling rate, single channel, and 16-bit Signed Integer PCM encoding.

### **Features** 
As speech features, 80-dimensional log mel-filterbank outputs were computed from 25ms window for each 10ms. Those log mel-filterbank features were further normalized to have zero mean and unit variance over the training partition of the dataset.

### **Prepare your input data**
You must prepare your own data before training the model, refer to data/demo_txt/demo_train.txt. 

### **Train model**
To get start, please config congfigs/config.yml file,  and simple run this command:

```
python train.py
```
This will train Spoken_language_identification model using data/demo_txt/demo_train.txt, then store the model on saved_weights folder, perform inference on demo_txt/demo_test.txt, print the inference results, and save the averaged accuracy in a text file.
### **Inference**
The pretrained model is provided in this [project](https://github.com/SpeechFlow-io/Spoken_language_identification/tree/main/saved_weights/20230228-084356), simple run this command:
```
python predict_by_weights.py
```
or
```
python predict_by_pb.py
```

## LICESES
Spoken_language_identification is released under the Apache License, version 2.0. The Apache license is a popular BSD-like license. Spoken_language_identification can be redistributed for free, even for commercial purposes, although you can not take off the license headers (and under some circumstances, you may have to distribute a license document).
