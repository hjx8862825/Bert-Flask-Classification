from webapi.utils.decorators.singleton_dec import singleton
from webapi.tf_models.same_dif_model_estimator import SameDifPredictThread


"""
用于初始化各类模型和标签的单例
"""
@singleton
class ModelCache(object):

    def __init__(self):
        self.models = {}
        self._init_same_dif_model()

    def _init_same_dif_model(self):
        same_dif_threaded_model = SameDifPredictThread()
        self.models['sameDif'] = same_dif_threaded_model
        pass

