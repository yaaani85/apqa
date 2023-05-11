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