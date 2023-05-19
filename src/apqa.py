import glog
import json
from rule_parser import RuleParser
from kge.model import KgeModel
from kge.util.io import load_checkpoint
from embtopk.embtopktable import EmbTopKEDBTable
from wmc import WMC
from utils import  get_terms_from_body, parse_program
import logging
import torch
import os
from tqdm import tqdm
class Query():
    """Abstract representation of the (datalog) Queries specific to CQD/QTO (Hamilton)"""
    #TODO REWRITE THIS CLASS
    def __init__(self, datalog_file, query_type):
        try:
            # to parser
            self._datalog_program, query, rule_1, rule_2, rule_3 = parse_program(datalog_file)
            self._predicates_in_query_body, self._subjects_in_query_body = get_terms_from_body(query)
            self._predicates_in_rule_1_body, self._subjects_in_rule_1_body = get_terms_from_body(rule_1)
            self._predicates_in_rule_2_body, self._subjects_in_rule_1_body = get_terms_from_body(rule_2)
            self._predicates_in_rule_3_body, self._subjects_in_rule_1_body = get_terms_from_body(rule_3)

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
        return self._predicates_in_body[0]
    
    @property
    def second_predicate_query(self):
        return self._predicates_in_body[1]
    @property
    def third_predicate_query(self):
        return self._predicates_in_body[2]
    
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
    def first_constant_query(self):
        return self._constants_in_body[0]

    @property
    def first_constant_rule_1(self):
        return self._constants_in_rule_1[0]
    
    @property
    def first_constant_rule_2(self):
        return self._constants_in_rule_2[0]

    @property
    def first_constant_rule_3(self):
        return self._constants_in_rule_3[0]



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

    def get_facts_in_predicate(self, predicate):
        results = []
        
        nodes = self._querier.get_node_details_predicate(f"{predicate}")
        nodes = json.loads(nodes)
        
        for node in nodes:
            facts = self._querier.get_facts_in_TG_node(int(node['id']))
            facts = json.loads(facts)
            for fact in facts:
                try:
                    name, prob = fact
                    print(prob)
                    results.append(name)                     
                except ValueError:
                    results.append(fact)
        
        return results



    # Rewrite this one
    def get_lineage(self, query):
        probabilities ={}

        answers = {}
        variables_per_answer = {}
        nodes = self._querier.get_node_details_predicate(query.qid)
        if nodes == "{}":
            return query.qid, ({}, {}), {}
        
        tuples = self._querier.get_facts_coordinates_with_predicate(query.qid)
        for current_answer, coordinates in tuples:
            current_answer = self._decode_term(current_answer[0])
            node, factid = coordinates
            leaves = self._querier.get_leaves(node, factid)
            
            answers[current_answer] = list()
            variables_per_answer[current_answer] = set()

            for l in leaves:
                proof = list()
                for leaf in l:
                    if self._is_determenistic_fact(leaf):
                        fact = self._decode_determnistic_fact( leaf)
                    else:
                        fact, probability = self._decode_probabilistic_fact( leaf)
                        probabilities[fact] = probability

                    proof.append(fact)
                answers[current_answer].append(proof)
                variables_per_answer[current_answer].update(proof)
        return query.qid, (answers, variables_per_answer), probabilities



class ApproximateProbabilisticQueryAnswerer():
    """Complex Logical Query Answering 
        Probabilistic reasoning over KGE
        Currently supports queries of type: 1p, 2p, ....
        requires: pretrained embedding model (libkge) & datalog engine (glog)

        CQD-Beam 
    
    """

    def __init__(self, embedding_model_path, edb_config_path, k, dataset, log_level):
        self._edb = glog.EDBLayer(edb_config_path)
        self._program = glog.Program(self._edb)
        self._rules = []
        self._k = k
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
    
    @program.setter
    def program(self, edb):
        self._program = glog.Program(edb)



    def answer_queries(self, query_type):
        solver = WMC()
        results = {}
        query_scores = torch.zeros(self.dataset.num_entities)
        final_scores = None

        os.chdir(self.dataset.datalog_queries_path)

        for query_id in tqdm(os.listdir(query_type)):
            query_path = self.dataset.get_query_path(query_type, query_id)
           
            try:
                qid, answers, probabilities = self._answer_query(query_path, query_type)
                qid = qid.replace("Q", "")
                wmc = solver.solve(answers, probabilities)
                results[qid] = wmc
                entity_indexes = [self.dataset.entities_to_index[entity.strip("<>")] for entity in wmc.keys()]
                query_scores[entity_indexes] = torch.Tensor(list(wmc.values()))
                final_scores = query_scores if final_scores is None else torch.column_stack([final_scores,query_scores])

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
        return engine.get_lineage(processed_query)
    


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


    def _handle_atom(self, first_predicate, first_constant):
        self._add_top_k_source(first_predicate, first_constant)
    

    def _handle_conjunction(self, first_predicate, first_constant, second_predicate):
        self._add_top_k_source(first_predicate, first_constant)
        intermediate_variable_assignments = self._collect_top_k_intermediate_variable_assignments(first_predicate, first_constant)
        for possible_assignment in intermediate_variable_assignments:
            self._add_top_k_source(second_predicate, possible_assignment)
        
        
    def _handle_2p(self, query):

        self._handle_conjunction(query.first_predicate_query, query.first_constant_query, query.second_predicate_query)
        return query

    def _handle_3p(self, query):
        self._add_top_k_source(query.first_predicate_query, query.first_constant_query)
        intermediate_variable_assignments = self._collect_top_k_intermediate_variable_assignments(query.first_predicate_query, query.first_constant_query)
        for possible_assignment in intermediate_variable_assignments:
            self._add_top_k_source(query.second_predicate_query, possible_assignment)
            intermediate_variable_assignments_2 = self._collect_top_k_intermediate_variable_assignments(query.first_predicate_query, query.first_constant_query)
            for possible_assignment in intermediate_variable_assignments_2:
                self._add_top_k_source(query.third_predicate_query, possible_assignment)

        return query


    def _handle_2i(self, query):
        self._handle_atom(query.first_predicate_rule_1, query.first_constant_rule_1)
        self._handle_atom(query.first_predicate_rule_2, query.first_constant_rule_2)
        return query

    def _handle_3i(self, query):
        self._handle_atom(query.first_predicate_rule_1, query.first_constant_rule_1)
        self._handle_atom(query.first_predicate_rule_2, query.first_constant_rule_2)
        self._handle_atom(query.first_predicate_rule_3, query.first_constant_rule_3)
        return query
    
    def _handle_ci(self, query):
        self._handle_conjunction(query.first_predicate_rule_1, query.first_constant_rule_1, query.second_predicate_rule_1)
        self._handle_atom(query.first_predicate_rule_2, query.first_constant_rule_2)
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
     
   
    def _add_top_k_source(self, predicate, subject):
        E = EmbTopKEDBTable(predicate, predicate, self.k, self.edb, self.embedding_model, self.dataset.entity_dict)

        # if source exists skip
        predicates_in_edb = self.edb.get_predicates()
        if not predicate in predicates_in_edb:
        
            self.edb.add_source(predicate, E)

        self.rules.append(f"{predicate}_{subject}(X, S) :- {predicate}({subject},X,S)")
        self.rules.append(f"PKB({subject},{predicate}, X) :- {predicate}_{subject}(X,S)")
        

    def _collect_top_k_intermediate_variable_assignments(self, predicate,subject):
        temp_predicate_name = f"{predicate}_{subject}"
        # update the program with the new edb source
        self.program = self.edb
        self._add_probabilistic_facts_to_knowledge_base()
        # create a (temp) engine just to get the intermediate variables (CQ-BEAM style)
        engine = Engine(self.edb, self.program)
        intermediate_variables = engine.get_facts_in_predicate(temp_predicate_name)
        return intermediate_variables
      
    
