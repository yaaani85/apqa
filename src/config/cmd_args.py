from configparser import ConfigParser
import os

class LearningSetting(object):

    def __init__(self):

                
        parser = ConfigParser()
        thisfolder = os.path.dirname(os.path.abspath(__file__))
        parser.read(os.path.join(thisfolder, 'default.ini'))
        self.data_path= parser.get('DATA', 'data_path')
        self.embedding_model_path = parser.get('EMBEDDING_MODEL', 'embedding_model_path')
        self.edb_config_path = parser.get('EDB', 'edb_config_path')
        self.top_k = parser.get("TOPK", 'top_k')



cmd_args = LearningSetting()