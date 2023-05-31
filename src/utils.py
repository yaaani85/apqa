import pickle
from rule_parser import RuleParser

# def get_name_to_entity_mappings(data_directory):
#     name_to_entities = {}
#     entitites_to_name = {}

#     with open('/home/yannick/phd/coding/pqa/data/FB15k/entity2text.txt') as f:
#         for line in f:
#             fb_id, name = line.strip().split('\t')
#             name_to_entities[name] = fb_id
#             entitites_to_name[fb_id] = name

#     return name_to_entities, entitites_to_name


def get_index_to_entity_mappings(data_directory):
    with open(f'{data_directory}/ind2ent.pkl', 'rb') as f:
        index_to_entity = pickle.load(f)

    with open(f'{data_directory}/ind2rel.pkl', 'rb') as f:
        index_to_relation = pickle.load(f)   

        return index_to_entity, index_to_relation
    

def get_answers(data_directory, query_type):
    with open(f'{data_directory}/eval/{query_type}/test_answers.pickle', 'rb') as f:
        test_answers = pickle.load(f)

    with open(f'{data_directory}/eval/{query_type}/test_answers_hard.pickle', 'rb') as f:
        test_answers_hard = pickle.load(f)

    with open(f'{data_directory}/eval/{query_type}/test_queries.pickle', 'rb') as f:
        queries = pickle.load(f)

  

    return (queries, test_answers, test_answers_hard)



def parse_query(datalog_query):
    if not datalog_query:
        return
    parser = RuleParser()
    parsed_rule = parser.parse_rule(datalog_query)
    parsed_rule_head = parsed_rule.get_head()
    rule_body = parsed_rule.get_body()
    parsed_rule_body = [tuple(literal.get_terms()) for literal in rule_body]

    return parsed_rule_head, parsed_rule_body



def get_query_id_form_query(query_datalog):
    head, _ = parse_query(query_datalog)
    
    return str(head[0].get_predicate_name())

def get_terms_from_body(query_datalog):
    if query_datalog is None:
        return None, None
    
    _, body = parse_query(query_datalog)

    if body is None:
        return None, None
    predicates = []
    subjects = []

    for literal in body:
        if len(literal) == 3:
            subject, predicate, _ = literal
            predicates.append(predicate)
            subjects.append(subject)

    return predicates, subjects


def parse_program(program, query_type):
    '''This method only applies to CQD/QTO (introduced by hamilton) dataset'''
    datalog_program = []
    datalog_query = None
    rule_1 = None
    rule_2 = None
    rule_3 = None

    with open(program, 'r') as file:
        datalog_program = []
        for line in file:
            if line.startswith("Q"):
                datalog_query = line
                datalog_program.append(line.strip())
            elif line.startswith("Rule_1") and not rule_1:
                rule_1 = line
                datalog_program.append(line.strip())

            elif line.startswith("Rule_1") and rule_1:
                rule_2 = line
                datalog_program.append(line.strip())


            elif line.startswith("Rule_2"):
                rule_2 = line
                datalog_program.append(line.strip())


            elif line.startswith("Rule_3"):
                rule_3 = line
                datalog_program.append(line.strip())

    return datalog_program, datalog_query, rule_1, rule_2, rule_3
