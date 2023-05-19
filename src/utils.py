import pickle

def get_name_to_entity_mappings(data_directory):
    name_to_entities = {}
    entitites_to_name = {}

    with open('/home/yannick/phd/coding/pqa/data/FB15k/entity2text.txt') as f:
        for line in f:
            fb_id, name = line.strip().split('\t')
            name_to_entities[name] = fb_id
            entitites_to_name[fb_id] = name

    return name_to_entities, entitites_to_name


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
            
       
def get_terms_from_body(body):
    predicates = []
    subjects = []

    for literal in body:
        terms = literal.get_terms()
        if len(terms) == 3:
            subject, predicate, _ = terms
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
                query_datalog = line
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
