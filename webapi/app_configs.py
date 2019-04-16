import os

class General_db_config(object):
    mincached = 5
    maxcached = 30

class Tag_db_config(object):
    tag_host_url = 'rm-wz99i0g2g7z8e59f76o.mysql.rds.aliyuncs.com'
    tag_username = 'fid_tag_user'
    tag_password = 'Fid123456'
    tag_db = 'fid_tag'
    tag_port = 3306
    tag_charset = 'utf8'

class Log_config(object):
    log_dir = os.path.dirname(os.path.abspath(os.getcwd())) + '/logs/'

class Model_path(object):
    wv_path = os.path.dirname(os.path.abspath(os.getcwd())) + "/data_helpers/word2vec/model-128"

class SameDifModelConfig(object):
    # 各类路径
    model_dir = "C:\\workProgram\\PycharmProjects\\bert-programs\\chinese_L-12_H-768_A-12"
    fine_tune_model_dir = "C:\\workProgram\\PycharmProjects\\bert-programs\\same_dif_output"
    config_fp = os.path.join(model_dir, 'bert_config.json')
    checkpoint_fp = os.path.join(model_dir, 'bert_model.ckpt')
    vocab_fp = os.path.join(model_dir, 'vocab.txt')
    seq_max_len = 256
    class_type = ['否','是']