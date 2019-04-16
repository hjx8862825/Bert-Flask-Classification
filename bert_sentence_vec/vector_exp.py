import tensorflow as tf
import modeling
import tokenization
from modeling import BertConfig
from extract_features import model_fn_builder,convert_lst_to_features,PoolingStrategy
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.estimator.estimator import Estimator
import os

#gpu config
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6
# 各类路径
model_dir = "C:\\workProgram\\PycharmProjects\\bert-programs\\chinese_L-12_H-768_A-12"
config_fp = os.path.join(model_dir, 'bert_config.json')
checkpoint_fp = os.path.join(model_dir, 'bert_model.ckpt')
vocab_fp = os.path.join(model_dir, 'vocab.txt')
# 中文tokenizer
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_fp)
# 定义输入迭代器
max_seq_len = 500
msg = ['塞力斯：持股5.1189%股东天沐君合减持0.1189%股份，减持后持股比例4.9999%。']
def input_fn_builder(msg):
    def gen():
        print("msg size %d" % len(msg))
        tmp_f = list(convert_lst_to_features(msg, max_seq_len, tokenizer))
        yield {
                'input_ids': [f.input_ids for f in tmp_f],
                'input_mask': [f.input_mask for f in tmp_f],
                'input_type_ids': [f.input_type_ids for f in tmp_f]
        }

    def input_fn():
        return (tf.data.Dataset.from_generator(
            gen,
            output_types={'input_ids': tf.int32,
                          'input_mask': tf.int32,
                          'input_type_ids': tf.int32},
            output_shapes={
                'input_ids': (None, max_seq_len),
                'input_mask': (None, max_seq_len),
                'input_type_ids': (None, max_seq_len)}))

    return input_fn

# 获取bert的estimator
model_fn = model_fn_builder(
            bert_config=modeling.BertConfig.from_json_file(config_fp),
            init_checkpoint=checkpoint_fp
        )

estimator = Estimator(model_fn=model_fn,config=RunConfig(session_config=config))
input_fn = input_fn_builder(msg)

if __name__ == '__main__':

    # 获取bert模型文本向量
    result = estimator.predict(input_fn, yield_single_examples=False)

    for prediction in result:
        print(prediction)
        print(prediction['encodes'].shape)
