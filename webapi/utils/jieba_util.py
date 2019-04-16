import jieba
import os


class JiebaUtils(object):

    def __init__(self,file_dir='dics'):
        self.stop_words = []
        print('initializing jieba......')
        dir = os.listdir(file_dir)
        for file in dir:
            if file != "stopWords.csv":
                jieba.load_userdict(file_dir + '/' + file)
            else:
                self.stop_words = [line.strip() for line in open(file_dir + "/" +file, 'r',encoding="utf-8").readlines()]

    def seg_sentence(self,sentence):
        sentence_seged = jieba.cut(sentence.strip())
        # loading stop word list
        out_seg_list = []
        for word in sentence_seged:
            if word not in self.stop_words:
                if word != '\t':
                    out_seg_list.append(word)

        return out_seg_list
