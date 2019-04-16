from flask_restful import Resource
from flask_restful import reqparse
import tokenization
from webapi.service.same_dif_service import SameDifService
from webapi.app_configs import SameDifModelConfig

class SameDifController(Resource):
    '''去重controller'''

    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        '''判断句子是否重复的api'''
        samDifService = SameDifService()
        self.parser.add_argument("sentence1", action="append")
        self.parser.add_argument("sentence2", action="append")

        data = self.parser.parse_args()
        sentence1 = data.get("sentence1")
        sentence2 = data.get("sentence2")

        result = samDifService.predict_same_dif(sentence1, sentence2)

        class_type = SameDifModelConfig.class_type
        apply_data = []
        for prediction in result:
                temp_data = {}
                for i in range(prediction.shape[-1]):
                    temp_data[class_type[i]] = float(prediction[i])
                apply_data.append(temp_data)

        return apply_data