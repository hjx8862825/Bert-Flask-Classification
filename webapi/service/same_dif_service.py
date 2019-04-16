from webapi.cache.Model_Caches import ModelCache

class SameDifService(object):

    def __init__(self):
        pass

    def predict_same_dif(self, sentence1, sentence2):
        # estimator
        cache = ModelCache()
        lst_str = []
        for s1, s2 in zip(sentence1, sentence2):
            comb_str = s1 + ' ||| ' + s2
            lst_str.append(comb_str)

        same_dif_predict_thread = cache.models['sameDif']
        result = same_dif_predict_thread.predict(lst_str)

        return result
