import glog
import json
from rule_parser import RuleParser
from kge.model import KgeModel
from kge.config import Config
from kge.dataset import Dataset
from kge.util.io import load_checkpoint
from embtopk.embtopktable import EmbTopKEDBTable
from wmc.sdd_wmc import WMCSolver
from utils import  get_terms_from_body, parse_program, get_query_id_form_query, get_py_format, print_proof
import logging
import torch
import os
from tqdm import tqdm
from collections import defaultdict

class Query():
    """Abstract representation of the (datalog) Queries specific to CQD/QTO (Hamilton)"""
    def __init__(self, datalog_file, query_type):
        try:
            # to parser
            self._query_type = query_type
            self._datalog_program, query, rule_1, rule_2, rule_3 = parse_program(datalog_file, query_type)

            self._qid = get_query_id_form_query(query)
            self._predicates_in_query, self._subjects_in_query = get_terms_from_body(query)
            self._predicates_in_rule_1, self._subjects_in_rule_1 = get_terms_from_body(rule_1)
            self._predicates_in_rule_2, self._subjects_in_rule_2 = get_terms_from_body(rule_2)
            self._predicates_in_rule_3, self._subjects_in_rule_3 = get_terms_from_body(rule_3)

        except Exception as e:
            raise ValueError(f"Incorrect Datalog rule or query type format, error: {e}")
        
    @property
    def datalog_program(self):
        return self._datalog_program
    
    @property
    def type(self):
        return self._query_type

    @property
    def qid(self):
        return self._qid

    @property
    def first_predicate_query(self):
        return self._predicates_in_query[0]
    
    @property
    def second_predicate_query(self):
        return self._predicates_in_query[1]
    @property
    def third_predicate_query(self):
        return self._predicates_in_query[2]
    
    @property
    def second_predicate_rule_1(self):
        return self._predicates_in_rule_1[1]

    
    @property
    def first_predicate_rule_1(self):
        return self._predicates_in_rule_1[0]
    
    @property
    def first_predicate_rule_2(self):
        return self._predicates_in_rule_2[0]
        
    @property
    def first_predicate_rule_3(self):
        return self._predicates_in_rule_3[0]

    @property
    def first_subject_query(self):
        return self._subjects_in_query[0]

    @property
    def first_subject_rule_1(self):
        return self._subjects_in_rule_1[0]
    
    @property
    def first_subject_rule_2(self):
        return self._subjects_in_rule_2[0]

    @property
    def first_subject_rule_3(self):
        return self._subjects_in_rule_3[0]



class Engine():
    """Wraps a glog-engine for KGE tasks"""
    def __init__(self, edb, program):

        self._reasoner = glog.Reasoner("probtgchase", edb, program, typeProv="FULLPROV", edbCheck=False, queryCont=False, delProofs=False)
        self._reasoner.create_model(0)
        self._tg = self._reasoner.get_TG()
        self._querier = glog.Querier(self._tg)

    @property
    def reasoner(self):
        return self._reasoner
    @property
    def querier(self):
        return self._querier

    def _is_determenistic_fact(self, leaf):
        if (leaf[0] == 1): return True
        return False

      
    def _decode_term(self, term):
        return self._querier.get_term_name(term)

    def _decode_determnistic_fact(self, atom):

        _, subject_id, predicate_id, object_id = atom 
        return ( self._decode_term(predicate_id), self._decode_term(subject_id), self._decode_term(object_id))

    def _decode_probabilistic_fact(self, atom):
        predicate_id, subject_id, object_id, probability = atom 

        return (self.querier.get_predicate_name(predicate_id), self._decode_term(subject_id), self._decode_term(object_id)), float(self._decode_term(probability))

    def get_lineage(self, query, dataset):
        probabilities ={}
        contains_deteremnistic_answer = defaultdict(list)
        answers = defaultdict(list)
        variables_per_answer = defaultdict(set)
        nodes = self._querier.get_node_details_predicate(query.qid)
        if nodes == "{}":
            return query.qid, ({}, {}, {}), {}
        tuples = self._querier.get_facts_coordinates_with_predicate(query.qid)

        for current_answer, (node, fact_id) in tuples:
            
            decoded_answer = self._decode_term(current_answer[0])
            leaves = self._querier.get_leaves(node, fact_id)
            # print("CA",decoded_answer, dataset.entitites_to_name[decoded_answer.strip("<>")], leaves)
  

            for l in leaves:
                proof = list()
                
                proof_is_determenistic = True
                for leaf in l:
                    if self._is_determenistic_fact(leaf):
                        fact = self._decode_determnistic_fact( leaf)
                    else:
                        fact, probability = self._decode_probabilistic_fact( leaf)
                        probabilities[fact] = probability
                        proof_is_determenistic = False
                    
                    proof.append(fact)

        
                answers[decoded_answer].append(proof)
                contains_deteremnistic_answer[decoded_answer].append(proof_is_determenistic)
                variables_per_answer[decoded_answer].update(proof)
        return query.qid, (answers, variables_per_answer, contains_deteremnistic_answer), probabilities


class ApproximateProbabilisticQueryAnswerer():
    """Complex Logical Query Answering 
        Probabilistic reasoning over KGE
        Currently supports queries of type: 1p, 2p, ....
        requires: pretrained embedding model (libkge) & datalog engine (glog)

        CQD-Beam 
    
    """

    def __init__(self, embedding_model_path, edb_config_path, k, threshold, max_queries, dataset):
        self._edb = glog.EDBLayer(edb_config_path)
        self._program = glog.Program(self._edb)
        self._rules = []
        self._k = k
        self._threshold = threshold
        self._max_queries = max_queries
        self._embedding_model = KgeModel.create_from(load_checkpoint(embedding_model_path))
        
        self._dataset = dataset

    @property
    def edb(self):
        return self._edb
    
    @property
    def embedding_model(self):
        return self._embedding_model

    @property
    def rules(self):
        return self._rules
    
    @property
    def dataset(self):
        return self._dataset

    @property
    def program(self):
        return self._program

    @property
    def k(self):
        return self._k
    
    @property
    def threshold(self):
        return self._threshold
    
    @property
    def max_queries(self):
        return self._max_queries
    
    @program.setter
    def program(self, edb):
        self._program = glog.Program(edb)



    def answer_queries(self, query_type):
        final_scores = None

        os.chdir(self.dataset.datalog_queries_path)
        query_list = os.listdir(query_type)
        query_list.sort(key=lambda x: int(x))

        for query_id in tqdm(query_list):
            query_scores = torch.zeros(self.dataset.num_entities)

            query_path = self.dataset.get_query_path(query_type, query_id)
            try:

                if query_id == str(self.max_queries):
                    return final_scores
           

                qid, answers, probabilities = self._answer_query(query_path, query_type)
                qid = qid.replace("Q", "")
                solver = WMCSolver(answers, probabilities)
                wmc, _ = solver.get_wmcs()

                entity_indexes = [(self.dataset.entities_to_index[entity.strip("<>")]) for entity in wmc.keys()]
                query_scores[entity_indexes] = torch.Tensor([score for score, _ in wmc.values()])
                query_scores = query_scores.unsqueeze(0)

                final_scores = query_scores if final_scores is None else torch.cat((final_scores, query_scores))

            except Exception as e:
                logging.error(e)
                raise e

        
        return final_scores
        
    
    def _answer_query(self, datalog_file, query_type):
        query = Query(datalog_file, query_type)
        
        processed_query = self._preprocess(query)
        self.program = self.edb
        self._add_deterministic_facts_to_knowledge_base()
        self._add_probabilistic_facts_to_knowledge_base()

        for rule in query.datalog_program:
            self.program.add_rule(rule.strip("\n"))

        engine = Engine(self.edb, self.program)
        return engine.get_lineage(processed_query, self.dataset)
    


    def _preprocess(self, query):
        
        if query.type == "1_2":
            return self._handle_2p(query)

        elif query.type == "1_3":
            return self._handle_3p(query)

        elif query.type == "2_2":
            return self._handle_2i(query)
        
        elif query.type == "3_3":
            return self._handle_ci(query)
    
        elif query.type == "2_3":
            return self._handle_3i(query)
    
        elif query.type == "4_3":
            return self._handle_ic(query)
        
        elif query.type == "2_2_disj":
            return self._handle_2u(query)
    
        elif query.type == "4_3_disj":
            return self._handle_up(query)
        else:
            raise Exception(f"Query type {query.type} not supported!")


    def _add_emb_source(self, predicate):
        print("adding source", predicate)
        
        E = EmbTopKEDBTable(predicate, predicate, self.k, self.edb, self.embedding_model, self.dataset.entity_dict, score_threshold=self.threshold)
        self.edb.add_source(predicate, E)        

        return E


    def _handle_conjunction(self, first_predicate, first_subject, second_predicate):
        
        self.edb.add_source(first_predicate, first_predicate_embd)       
        first_predicate_embd = self._add_emb_source(first_predicate)

        self._add_rules(first_predicate,  first_subject)
        subject = get_py_format(first_subject)
        intermediate_variables = first_predicate_embd.get_iterator(subject, last_step=False)
        
        self._add_emb_source(second_predicate)
        
        for intermediate_result in intermediate_variables.data:
            # second_predicate_embd = EmbTopKEDBTable(second_predicate, second_predicate, 14505, self.edb, self.embedding_model, self.dataset.entity_dict, score_threshold=self.threshold)
            # self.edb.add_source(second_predicate, second_predicate_embd) 
            _, intermediate_object, probabilty = intermediate_result
            # print("intermediate", self.dataset.entitites_to_name[intermediate_object.strip("<>")], probapPPbilty)

            self._add_rules(second_predicate, intermediate_object)
        

    def _handle_2p(self, query):
        self._handle_conjunction(query.first_predicate_query, query.first_subject_query , query.second_predicate_query)
        return query

    def _handle_3p(self, query):
        first_predicate = self._add_emb_source(query.first_predicate_query)
        self._add_rules(query.first_predicate_query,  query.first_subject_query)
        first_subject = get_py_format(query.first_subject_query)
        intermediate_variables = first_predicate.get_iterator(first_subject, last_step=False)
        second_predicate = self._add_emb_source(query.second_predicate_query)
        for intermediate_result in intermediate_variables.data:
            _, intermediate_object, probabilty = intermediate_result
            self._add_rules(query.second_predicate_query, intermediate_object)

            intermediate_py_object = get_py_format(intermediate_object)
            intermediate_variables_2 = second_predicate.get_iterator(intermediate_py_object, last_step=False)
            for intermediate_result in intermediate_variables_2.data:
                _, intermediate_object, probabilty = intermediate_result
                self._add_rules(query.second_predicate_query, intermediate_object)


        return query



    def _handle_2i(self, query):

        self._add_emb_source(query.first_predicate_rule_1)
        self._add_rules(query.first_predicate_rule_1, query.first_subject_rule_1)

        self._add_emb_source(query.first_predicate_rule_2)
        self._add_rules(query.first_predicate_rule_2, query.first_subject_rule_2)

    
        return query

    def _handle_3i(self, query):
        self._add_emb_source(query.first_predicate_rule_1)
        self._add_rules(query.first_predicate_rule_1,   query.first_subject_rule_1)

        self._add_emb_source(query.first_predicate_rule_2)
        self._add_rules(query.first_predicate_rule_2,   query.first_subject_rule_2)

        self._add_emb_source(query.first_predicate_rule_3)
        self._add_rules(query.first_predicate_rule_3,   query.first_subject_rule_3)
        return query
    
    def _handle_ci(self, query):
        self._handle_conjunction(query.first_predicate_rule_1, query.first_subject_rule_1, query.second_predicate_rule_1)
        
        self._add_emb_source(query.first_predicate_rule_2)
        self._add_rules(query.first_predicate_rule_2,   query.first_subject_rule_2)

        return query
        
    def _handle_ic(self, query):
        # same as ci
        self._handle_ci(query)
        return query

    def _handle_2u(self, query):
        # same as 2i
        self._handle_2i(query)
        return query
    
    def _handle_up(self, query):
        # same as 2i
        self._handle_2i(query)
        return query
    

    def _add_probabilistic_facts_to_knowledge_base(self):
        for rule in self.rules:
            self.program.add_rule(rule)


    def _add_deterministic_facts_to_knowledge_base(self):
        self.program.add_rule("PKB(X, Y, Z) :- TE(X, Y, Z)")
     

    def _add_rules(self, predicate, subject):
        self.rules.append(f"{predicate}_{subject}(X, S) :- {predicate}({subject},X,S)")
        self.rules.append(f"PKB({subject},{predicate}, X) :- {predicate}_{subject}(X,S)")
