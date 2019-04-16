import tensorflow as tf
import modeling
from webapi.app_configs import SameDifModelConfig
from extract_features import predict_model_builder,convert_lst_to_features
from threading import Thread
import tokenization
from queue import Queue

class SameDifModel(object):

    def __init__(self):
        # tokenizer
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=SameDifModelConfig.vocab_fp, do_lower_case=True)
        # config
        self.max_seq_len = SameDifModelConfig.seq_max_len
        self.run_config = tf.estimator.RunConfig(
            model_dir=SameDifModelConfig.fine_tune_model_dir,
            save_checkpoints_steps=1000,
        )
        # 创建fine-tune bert model
        self.bert_config = modeling.BertConfig.from_json_file(SameDifModelConfig.config_fp)
        self.model_fn = predict_model_builder(
            bert_config=self.bert_config,
            init_checkpoint=SameDifModelConfig.checkpoint_fp,
            num_labels=2
        )

        # estimator
        self.estimator = tf.estimator.Estimator(
            model_fn=self.model_fn,
            config=self.run_config)

    def predict(self, input_features):
        return self.estimator.predict(input_features)


class SameDifPredictThread(SameDifModel):
    '''使用队列解决tensorflow使用至web服务时每次都需要启动session的问题'''
    def __init__(self):
        super(SameDifPredictThread, self).__init__()
        self.input_quene = Queue(maxsize=1)
        self.output_quene = Queue(maxsize=1)

        self.prediction_thread = Thread(target=self.predict_from_quene, daemon=True)
        self.prediction_thread.start()

    def generate_from_quene(self):
        '''队列item的生成器'''
        while True:
            tmp_f = list(convert_lst_to_features(self.input_quene.get(), self.max_seq_len, self.tokenizer))
            yield {
                'input_ids': [f.input_ids for f in tmp_f],
                'input_mask': [f.input_mask for f in tmp_f],
                'input_type_ids': [f.input_type_ids for f in tmp_f]
            }

    def predict_from_quene(self):
        '''从input队列中取值进行预测'''
        for i in self.estimator.predict(self.input_fn_builder, yield_single_examples=False):
            self.output_quene.put(i)

    def predict(self, lst_str):
        '''把预测特征放入队列，并从输出队列取出预测值'''
        self.input_quene.put(lst_str)
        prediction = self.output_quene.get()

        return prediction

    def input_fn_builder(self):
        return (tf.data.Dataset.from_generator(
                self.generate_from_quene,
                output_types={'input_ids': tf.int32,
                              'input_mask': tf.int32,
                              'input_type_ids': tf.int32},
                output_shapes={
                    'input_ids': (None, self.max_seq_len),
                    'input_mask': (None, self.max_seq_len),
                    'input_type_ids': (None, self.max_seq_len)}))


    # def quene_predict_input_fn(self):
    #     '''构造input输入'''
    #     dataset = tf.data.Dataset.from_generator(self.generate_from_quene,
    #                                              output_types={
    #                                                  'input_ids': tf.int32,
    #                                                   'input_mask': tf.int32,
    #                                                   'input_type_ids': tf.int32})
    #     return dataset