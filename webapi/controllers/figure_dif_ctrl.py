from flask_restful import Resource
from flask_restful import reqparse
from webapi.cache.Model_Caches import ModelCache
import jieba.analyse as analyse
import jieba.posseg as pseg

'''
    区分举例词不在同类的controller
    如:男人,女人,单车 --->单车
'''

class FigureDifferent(Resource):

    # 初始化一个参数解析器
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument('text')
        self.parser.add_argument('testWord')
        data = self.parser.parse_args()
        #'tag','vn','ns','n','nr','stock','theme','nt',
        allow_pos = ('tag','vn','ns','n','nr','stock','theme','nt','keyword')
        # 文章
        text = data.get('text')
        # 获取testWords
        testWords = data.get('testWord').split(',')
        top_k_rank = len(testWords) // 5 * 3
        tags = analyse.textrank(text, topK=top_k_rank, withWeight=False, allowPOS=allow_pos)
        # cuts = pseg.lcut(data.get('text'))
        # print(cuts)
        cache = ModelCache()
        wv_model = cache.vec_dic_model

        # 获取交集
        intersection = []
        for tag in tags:
            if tag in testWords:
                intersection.append(tag)

        excp_words = []
        for word in testWords:
            temp_tags = intersection.copy()
            temp_tags.append(word)
            excp_word = wv_model.doesnt_match(temp_tags)
            if excp_word in testWords and excp_word not in intersection:
                testWords.remove(excp_word)
                excp_words.append(excp_word)

        print("intersection:",intersection)
        print("final:",testWords)
        print("except:",excp_words)

        return testWords