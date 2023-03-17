
class Vocab:
    def __init__(self, file_path):
        self.token_list = []
        self.load_vocab_from_file(file_path)

    def load_vocab_from_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            vocabs = f.readlines()
            for vocab in vocabs:
                self.token_list.append(vocab.strip('\n'))
