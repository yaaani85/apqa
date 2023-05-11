import os
import pickle
import pickle
import os
from utils import get_name_to_entity_mappings, get_index_to_entity_mappings

class Dataset():
    def __init__(self, data_path):
        assert os.path.exists(data_path), "Please specify an existsing path to the Dataset"

        self._index_to_entities, self._index_to_relation = get_index_to_entity_mappings(data_path)
        self._name_to_entities, self._entitites_to_name = get_name_to_entity_mappings(data_path)
        self._data_directory = data_path

        with open(f"{self._data_directory}/datalog/test_queries.txt", "r") as datalog_txt:
            datalog_queries = [line.rstrip() for line in datalog_txt]
            query_types = ["2p"] * len(datalog_queries)
            self._datalog_queries =  zip(datalog_queries, query_types)
    # 

        with open(f'{self._data_directory}/test_answers_fbk15.pickle', 'rb') as handle:
            self._test_answers = pickle.load(handle)

        with open(f'{self._data_directory}/test_answers_hard_fbk15.pickle', 'rb') as handle:
            self._test_answers_hard = pickle.load(handle)

        with open(f'{self._data_directory}/test_queries_fbk15.pickle', 'rb') as handle:
            self._queries = pickle.load(handle)


        with open(f'{self._data_directory}/entity_idx.pickle', 'rb') as handle:
            self._entity_dict = pickle.load(handle)

    @property
    def entity_dict(self):
        return self._entity_dict

    @property
    def datalog_queries(self):
        return self._datalog_queries
    
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
    
    @property
    def name_to_entities(self):
        return self._name_to_entities
    
    @property
    def entities_to_name(self):
        return self._entitites_to_name
    

    @property
    def num_entities(self):
        return len(self._index_to_entities)