from flask import Flask
from flask_restful import Api
from webapi.controllers.figure_dif_ctrl import FigureDifferent
from webapi.controllers.same_dif_ctrl import SameDifController
from webapi.cache.Model_Caches import ModelCache
import os

app = Flask(__name__)
api = Api(app)

api.add_resource(FigureDifferent,'/dif')
api.add_resource(SameDifController,'/samedif')

if __name__ == '__main__':
    # utils.init_jieba(os.path.dirname(os.path.abspath(os.getcwd())) + "\\data_helpers\\dics")
    ModelCache()
    app.run(debug=True, use_reloader=False)