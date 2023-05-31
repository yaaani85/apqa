import os
import pickle
import pickle
import os
from utils import  get_index_to_entity_mappings

class Dataset():
    # TODO Load Queries in DataLoader 
    def __init__(self, data_path):
        assert os.path.exists(data_path), "Please specify an existsing path to the Dataset"

        self._index_to_entities, self._index_to_relation = get_index_to_entity_mappings(data_path)
        # self._name_to_entities, self._entitites_to_name = get_name_to_entity_mappings(data_path)
        self._data_directory = data_path

    # TODO to utils
        # try:
        #     # with open(f'{self.data_directory}/entity_idx.pickle', 'rb') as handle:
        #     #     self._entity_dict = pickle.load(handle)

        # except IOError:
        self._entity_dict = None

        # except Exception as e:
        #     raise e
    
    @property
    def name(self):
        # change hardcode
        return "FB15K-237"

    @property
    def query_types(self):
        return ['1_2']
        # return ['1_2','1_3','2_2','2_3','4_3','3_3','2_2_disj','4_3_disj']

    @property
    def entity_dict(self):
        return self._entity_dict

    @property
    def datalog_queries_path(self):
        return self._data_directory + "/datalog"
    
    @property
    def queries(self):
        return self._queries
    
    @property
    def answers(self):
        return self._test_answers
    
    @property
    def answers_hard(self):
        return self._test_answers_hard
    @property
    def index_to_entities(self):
        return self._index_to_entities
    
    @property 
    def entities_to_index(self):
        return {entity: idx for idx, entity in self._index_to_entities.items()}
        

    @property
    def index_to_relation(self):
        return self._index_to_relation
    
    @property
    def data_directory(self):
        return self._data_directory
    
    # @property
    # def name_to_entities(self):
    #     return self._name_to_entities
    
    # @property
    # def entities_to_name(self):
    #     return self._entitites_to_name
    

    @property
    def num_entities(self):
        return len(self._index_to_entities)
    

   
    def get_query_path(self, query_type, query_id):
        query_path = os.path.join(query_type, query_id)
        query_path = os.path.join(self.datalog_queries_path, query_path)
        query_path = os.path.join(query_path, "datalog.txt")
        assert os.path.isfile(query_path), f"FILE NOT FOUND{query_path}"
        return query_path
    
        