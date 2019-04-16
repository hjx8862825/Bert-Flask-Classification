from flask_restful import Resource
from flask_restful import reqparse
from webapi.cache.Model_Caches import ModelCache

class SingleModel(Resource):

    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("sentences",action="append")
