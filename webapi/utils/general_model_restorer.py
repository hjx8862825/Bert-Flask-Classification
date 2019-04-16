import tensorflow as tf
from configs import general_config
from utils import PaddedDataIterator
from application.utils.decorators.singleton_dec import singleton
from application.utils import constants
import os
import platform

'''
通用的模型加载方法，计算图不需要重新构建，只需要加载
placeholder input:X为name = sentences_placeholder
placeholder input:X_len为name = sentence_lengths_placeholder
预测op: name = predict_op
'''


class GeneralPredictModel(object):
    def __init__(self, model_path_name, model_type="CRNN"):
        self.model_path_name = model_path_name
        self.model_type = model_type
        general_config.change_type_save_dir(model_path_name)
        self.graph_dir, self.app_dir = self._init_basic_params()
        self.meta_file = self._get_meta_file()
        self.session = tf.Session()

    # 获取模型的基本路径，因为在init中修改过了模型基本参数，所以general_config里面的值已经改变
    def _init_basic_params(self):
        # 当前文件的路径为基准,就算在cache里面也是向上两级
        base_dir = os.path.abspath(os.path.curdir)
        dir_list = base_dir.split(constants.separator)
        # 调用两次回到根目录
        dir_list.remove(dir_list[len(dir_list) - 1])
        dir_list.remove(dir_list[len(dir_list) - 1])
        # 取得模型文件夹路径
        app_dir = constants.separator.join(dir_list) + constants.separator
        graph_dir = constants.separator.join(
            dir_list) + constants.separator + general_config.save_dir + constants.separator + self.model_type
        return graph_dir, app_dir

    def _get_meta_file(self):
        files = os.listdir(self.graph_dir)
        for i in range(len(files)):
            file = files[-i]
            if "meta" in file:
                return file

    # 通用预测
    def general_predict(self, sentences):

        # 加载计算图
        saver = tf.train.import_meta_graph(self.graph_dir + "/" + self.meta_file)
        # 加载保存的参数
        saver.restore(self.session, tf.train.latest_checkpoint(self.graph_dir))

        # 预测句子的封装其
        test_generator = PaddedDataIterator(sentences=sentences,
                                            vocab2intPath=self.app_dir + general_config.global_nonstatic_v2i_path,
                                            sent_len_cut=general_config.max_seq_len)

        # 载入计算图
        graph = tf.get_default_graph()
        input1 = graph.get_tensor_by_name("input_layer/sentences_placeholder:0")
        input2 = graph.get_tensor_by_name("input_layer/sentence_lengths_placeholder:0")
        predict_op = graph.get_tensor_by_name("Accuracy/predict_op:0")
        # 执行预测
        cur_loop = test_generator.loop
        res = {}
        while (test_generator.loop == cur_loop):
            batch_idx, batch_seqs, _, batch_lens = test_generator.next(batch_size=128, need_all=True)
            predicted = self.session.run(predict_op,
                                         feed_dict={input1: batch_seqs, input2: batch_lens})
            for (id, label) in zip(batch_idx, predicted):
                res[id] = int(label)

        return res


if __name__ == '__main__':
    gpm = GeneralPredictModel("/five")
    f = open(
        "C:\workProgram\PycharmProjects\DrTextClassification\data_helpers\dataset\\anouncement\\sentences_test.txt",
        'r', encoding='utf_8_sig')
    sentences = []
    for line in f:
        print(line)
        sentences.append(line)
    res = gpm.general_predict(sentences)
    print(res)
