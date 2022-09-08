import argparse
import re
from os import listdir
import numpy as np
import pickle as pcl


class N_gram:

    def __init__(self, N=2):
        self.N = N
        self.model_data = {}
        self.generate_text = []

    def fit(self, input_path):
        if input_path == '':
            text = ''
            while True:
                try:
                    str = input()
                    text += str + ' '
                except EOFError:
                    break
            self.formatting_text(text)
        else:
            dirs = listdir(input_path)
            for file in dirs:
                with open(input_path + "\\" + file, "r",
                          encoding='cp1251') as f:
                    text = f.read()
                    self.formatting_text(text)
        for key in self.model_data.keys():
            unique_values, counts \
                = np.unique(self.model_data[key], return_counts=True)
            prob = counts / len(self.model_data[key])
            self.model_data[key] = np.column_stack((unique_values, prob))

    def save(self, save_path):
        with open(save_path, 'wb') as save_file:
            pcl.dump(self.model_data, save_file, protocol=pcl.HIGHEST_PROTOCOL)

    def load(self, load_path):
        with open(load_path, 'rb') as load_file:
            self.model_data.clear()
            self.model_data = pcl.load(load_file)
            self.N = len(list(self.model_data.keys())[0])

    def generate(self, length, seed, prefix=''):
        np.random.seed(seed)
        self.generate_text = []
        cur_phrase = ''

        if prefix != '' and tuple(prefix.split()) in self.model_data:
            cur_phrase = tuple(prefix.split())
        else:
            random_index = np.random.choice(len(self.model_data))
            cur_phrase = list(self.model_data.keys())[random_index]
        for word in cur_phrase:
            self.generate_text.append(word)

        i = self.N
        while i < length:
            words = self.model_data[cur_phrase][:, 0]
            prob = self.model_data[cur_phrase][:, 1].astype(float)
            next_word = np.random.choice(words, p=prob)
            self.generate_text.append(next_word)
            cur_phrase = list(cur_phrase)
            cur_phrase.append(next_word)
            cur_phrase = tuple(cur_phrase[-self.N:])
            i += 1

            if cur_phrase not in self.model_data:
                random_index = np.random.choice(len(self.model_data))
                cur_phrase = list(self.model_data.keys())[random_index]
                for word in cur_phrase:
                    self.generate_text.append(word)
                i += self.N

        generated_text = self.generate_text[:length]
        for word in generated_text:
            print(word, end=' ')

    def formatting_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Zа-яА-ЯёЁ ]+', ' ', text)
        split_text = text.split()
        for i in range(0, len(split_text) - self.N):
            if tuple(split_text[i:i + self.N]) in self.model_data:
                self.model_data[tuple(split_text[i:i + self.N])]\
                    .append(split_text[i + self.N])
            else:
                self.model_data[tuple(split_text[i:i + self.N])]\
                    = list([split_text[i + self.N]])

    def print_model(self, size=-1):
        print()
        cnt = 0
        for k in self.model_data:
            print(k, ":", ' {', sep='')
            print(self.model_data[k])
            print('},')
            cnt += 1
            if cnt > size and size != -1:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir',
                        type=str,
                        default='',
                        help='Path to data directory')
    parser.add_argument('--model',
                        type=str,
                        default='',
                        help='Path to file where the model is saved')
    args = parser.parse_args()

    n_gram = N_gram(3)
    n_gram.fit(args.input_dir)
    n_gram.save(args.model)
