# Copyright 2023 by zhongying
#


from . import load_yaml
from util.utils import preprocess_paths


class Config:
    """ User configs class for training, testing or infering """

    def __init__(self, path: str):
        print('configs file path:', path)
        config = load_yaml(preprocess_paths(path))
        self.speech_config = config.get("speech_config", {})
        self.model_config = config.get("model_config", {})
        self.dataset_config = config.get("dataset_config", {})
        self.optimizer_config = config.get("optimizer_config", {})
        self.running_config = config.get("running_config", {})

    def print(self):
        print('==================================================')
        print('speech configs:', self.speech_config)
        print('--------------------------------------------------')
        print('model configs:', self.model_config)
        print('--------------------------------------------------')
        print('dataset configs:', self.dataset_config)
        print('--------------------------------------------------')
        print('optimizer configs', self.optimizer_config)
        print('--------------------------------------------------')
        print('running configs:', self.running_config)
        print('==================================================')

    def toString(self):
        string = ''
        string += '#==================================================' + '\n'
        string += '#speech config: ' + str(self.speech_config) + '\n'
        string += '#--------------------------------------------------' + '\n'
        string += '#model config: ' + str(self.model_config) + '\n'
        string += '#--------------------------------------------------' + '\n'
        string += '#dataset config: ' + str(self.dataset_config) + '\n'
        string += '#--------------------------------------------------' + '\n'
        string += '#optimizer config: ' + str(self.optimizer_config) + '\n'
        string += '#--------------------------------------------------' + '\n'
        string += '#running config: ' + str(self.running_config) + '\n'
        string += '#==================================================' + '\n'
        return string
