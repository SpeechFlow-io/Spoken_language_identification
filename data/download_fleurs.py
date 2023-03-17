"""
download FLEURS dataset from huggingface, reface to https://huggingface.co/datasets/google/xtreme_s
"""

from datasets import load_dataset
fleurs_asr = load_dataset("google/xtreme_s", "fleurs.tr_tr") # for turkish
# to download all data for multi-lingual fine-tuning uncomment following line
# fleurs_asr = load_dataset("google/xtreme_s", "fleurs.all")
print(fleurs_asr)

# load audio sample on the fly
audio_input = fleurs_asr["train"][0]["audio"]  # first decoded audio sample
transcription_train = fleurs_asr["train"][0]["transcription"]  # first transcription

fleurs_train = fleurs_asr['train']
fleurs_dev = fleurs_asr['validation']
fleurs_test = fleurs_asr['test']

raw_transcription = fleurs_train["raw_transcription"]
transcription = fleurs_train["transcription"]

print(raw_transcription, transcription)


