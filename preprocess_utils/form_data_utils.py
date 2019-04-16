import os
import pandas as pd
from sklearn.model_selection import train_test_split


def process_str_sentence(str_sentence):
    '''bert的中文预训练是基于char的，所以不需要分词'''
    sentence = str_sentence.lower().strip()
    # print(sentence)
    # sentence = [process_word(word) for word in sentence]
    # sentence_cut = jieba.lcut(sentence)
    # while " " in sentence_cut:
    #     sentence_cut.remove(" ")
    # while "\r\n" in sentence_cut:
    #     sentence_cut.remove("\r\n")
    # while "\n" in sentence_cut:
    #     sentence_cut.remove("\n")
    # while "\t" in sentence_cut:
    #     sentence_cut.remove("\t")
    # while u'\u3000' in sentence_cut:
    #     sentence_cut.remove(u'\u3000')
    # while u'\xa0' in sentence_cut:
    #     sentence_cut.remove(u'\xa0')
    return sentence

def comma_csv_to_tab_csv_for_comb_sents(csvFile):
    '''对型如：sentence1，sentence2，label的以逗号分割的csv做tsv转换'''
    data = pd.read_csv(csvFile, header=None, sep=",")
    for index, row in data.iterrows():
        # print(row)
        data[0].ix[index] = process_str_sentence(row[0])
        data[1].ix[index] = process_str_sentence(row[1])

    base_path = csvFile.split(".")
    head_path = base_path[0] + "_pre"
    sfx = base_path[-1]
    tsv_file = head_path + "." + sfx
    data.to_csv(tsv_file, sep="\t", header=False, encoding='utf_8_sig')
    with open(tsv_file, 'r', encoding='utf_8_sig') as f:
        data_valid = pd.read_csv(f, header=None, sep="\t")
        print(data_valid)

def split_train_val(trainFile, validation_size=0.1):
    """
    分割训练和测试集并保存
    """
    save_file_train = trainFile.replace(trainFile.split("\\")[-1], 'train_sep.tsv')
    save_file_test = trainFile.replace(trainFile.split("\\")[-1], 'valid_sep.tsv')
    with open(trainFile, 'r') as f:
        data = pd.read_csv(trainFile, sep="\t", dtype=str)
    train, valid = train_test_split(data, test_size=validation_size)
    with open(save_file_train, 'w', encoding='utf_8_sig') as f:
        train.to_csv(f, index=False, sep="\t")
    with open(save_file_test, 'w', encoding='utf_8_sig') as f:
        valid.to_csv(f, index=False, sep="\t")

    return train, valid

if __name__ == '__main__':
    # csv_file = 'C:\\workProgram\\PycharmProjects\\bert-programs\\same_dif\\train_all.tsv'
    # comma_csv_to_tab_csv_for_comb_sents(csv_file)
    tsv_file = 'C:\\workProgram\\PycharmProjects\\bert-programs\\same_dif\\train_all.tsv'
    split_train_val(tsv_file, 0.2)
