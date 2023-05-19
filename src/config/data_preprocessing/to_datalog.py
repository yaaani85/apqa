import os
from kbc.utils import QuerDAG


def to_datalog(env, query_type, queries):

    print("TO DATALOG")
    if not os.path.exists(f'fb15k-237/{query_type}'):
        os.makedirs(f'fb15k-237/{query_type}')
        print("MADE DIR")
    
    print("query type", type(query_type))
    query_type = query_type.strip("\n")
    if query_type == "1_2":
        datalog = handle_2p(env, query_type, queries)

    elif query_type == "1_3":
        datalog = handle_3p(env, query_type, queries)

    elif query_type == "2_2":
        datalog = handle_2i(env, query_type, queries)

    elif query_type == "3_3":
        datalog = handle_ci(env, query_type, queries)

    elif query_type == "2_3":
        datalog = handle_3i(env, query_type, queries)

    elif query_type == "4_3":
        datalog = handle_ic(env, query_type, queries)

    elif query_type == "2_2_disj":
        datalog = handle_2u(env, query_type, queries)

    elif query_type == "4_3_disj":
        datalog = handle_uc(env, query_type, queries)

    else:
        raise Exception(f"Invalid query type {query_type}")

    


def handle_2p(env, query_type, queries):
    datalog = []
    for query_id, query in enumerate(queries):

        # print("QUERY", query)
        anchor, rel1, x1, x2, rel2, x3 = query.split('_')

        anchor = env.ent_id2fb[int(anchor)]
        rel1 = env.rel_id2fb[int(rel1)]
        rel2 = env.rel_id2fb[int(rel2)]

        query = [f"Q{query_id}(A2) :- PKB(<{anchor}>,<{rel1}>, A1), PKB(A1,<{rel2}>, A2)"]

        if not os.path.exists(f'fb15k-237/{query_type}/{query_id}'):
            os.makedirs(f'fb15k-237/{query_type}/{query_id}')

        with open(f'fb15k-237/{query_type}/{query_id}/datalog.txt', 'w+') as f:
            f.writelines([rule + '\n' for rule in query])
        
    return datalog


def handle_3p(env, query_type, queries):
    datalog = []

    for query_id, query in enumerate(queries):

        anchor, rel1, x1, x2, rel2, x3, x4, rel3, x5 = query.split('_')

        anchor = env.ent_id2fb[int(anchor)]
        rel1 = env.rel_id2fb[int(rel1)]
        rel2 = env.rel_id2fb[int(rel2)]
        rel3 = env.rel_id2fb[int(rel3)]

        query = [f"Q{query_id}(A3) :- PKB(<{anchor}>,<{rel1}>, A1), PKB(A1,<{rel2}>, A2), PKB(A2, <{rel3}>, A3)"]

        if not os.path.exists(f'fb15k-237/{query_type}/{query_id}'):
            os.makedirs(f'fb15k-237/{query_type}/{query_id}')
        with open(f'fb15k-237/{query_type}/{query_id}/datalog.txt', 'w+') as f:
            f.writelines([rule + '\n' for rule in query])

    return datalog

def handle_2i(env, query_type, queries):
    datalog = []

    for query_id, query in enumerate(queries):

        anchor1, rel1, x2, anchor2, rel2, x5 = query.split('_')

        anchor1 = env.ent_id2fb[int(anchor1)]
        anchor2 = env.ent_id2fb[int(anchor2)]

        rel1 = env.rel_id2fb[int(rel1)]
        rel2 = env.rel_id2fb[int(rel2)]

        query = [f"Q{query_id}(V0) :- Rule_1(V0), Rule_2(V0)",
                   f"Rule_1(A1) :- PKB(<{anchor1}>,<{rel1}>,A1)",
                   f"Rule_2(A2) :- PKB(<{anchor2}>,<{rel2}>,A2)"]

        if not os.path.exists(f'fb15k-237/{query_type}/{query_id}'):
            os.makedirs(f'fb15k-237/{query_type}/{query_id}')
        with open(f'fb15k-237/{query_type}/{query_id}/datalog.txt', 'w+') as f:
            f.writelines([rule + '\n' for rule in query])
    return datalog

def handle_ci(env, query_type, queries):
    datalog = []

    for query_id, query in enumerate(queries):

        anchor1, rel1, x2, x3, rel2, x4, anchor3, rel3, x5 = query.split('_')

        anchor1 = env.ent_id2fb[int(anchor1)]
        anchor3 = env.ent_id2fb[int(anchor3)]

        rel1 = env.rel_id2fb[int(rel1)]
        rel2 = env.rel_id2fb[int(rel2)]
        rel3 = env.rel_id2fb[int(rel3)]

        query = [f"Q{query_id}(V0) :- Rule_1(V0), Rule_2(V0)",
                   f"Rule_1(A2) :- PKB(<{anchor1}>,<{rel1}>, A1), PKB(A1,<{rel2}>, A2)",
                   f"Rule_2(A3) :- PKB(<{anchor3}>,<{rel3}>,A3)"]

        if not os.path.exists(f'fb15k-237/{query_type}/{query_id}'):
            os.makedirs(f'fb15k-237/{query_type}/{query_id}')
        with open(f'fb15k-237/{query_type}/{query_id}/datalog.txt', 'w+') as f:
            f.writelines([rule + '\n' for rule in query])

    return datalog

def handle_3i(env, query_type, queries):
    datalog = []

    for query_id, query in enumerate(queries):

        anchor1, rel1, x2, anchor2, rel2, x5, anchor3, rel3, x6 = query.split('_')

        anchor1 = env.ent_id2fb[int(anchor1)]
        anchor2 = env.ent_id2fb[int(anchor2)]
        anchor3 = env.ent_id2fb[int(anchor3)]

        rel1 = env.rel_id2fb[int(rel1)]
        rel2 = env.rel_id2fb[int(rel2)]
        rel3 = env.rel_id2fb[int(rel3)]

        query = [f"Q{query_id}(V0) :- Rule_1(V0), Rule_2(V0), Rule_3(V0)",
                   f"Rule_1(A1) :- PKB(<{anchor1}>,<{rel1}>, A1))",
                   f"Rule_2(A2) :- PKB(<{anchor2}>,<{rel2}>,A2)",
                   f"Rule_3(A3) :- PKB(<{anchor3}>,<{rel3}>,A3)"]
        if not os.path.exists(f'fb15k-237/{query_type}/{query_id}'):
            os.makedirs(f'fb15k-237/{query_type}/{query_id}')
        with open(f'fb15k-237/{query_type}/{query_id}/datalog.txt', 'w+') as f:
            f.writelines([rule + '\n' for rule in query])
    return datalog

def handle_ic(env, query_type, queries):
    datalog = []

    for query_id, query in enumerate(queries):

        anchor1, rel1, x1, x2, rel2, x3, anchor3, rel3, x6 = query.split('_')

        anchor1 = env.ent_id2fb[int(anchor1)]
        anchor3 = env.ent_id2fb[int(anchor3)]

        rel1 = env.rel_id2fb[int(rel1)]
        rel2 = env.rel_id2fb[int(rel2)]
        rel3 = env.rel_id2fb[int(rel3)]

        query = [f"Q{query_id}(A1) :- PKB(V0,<{rel3}>, A1)",
                   f"QIntersection(V0) :- Rule_1(V0), Rule_2(V0)",
                   f"Rule_1(A2) :- PKB(<{anchor1}>,<{rel1}>, A1), PKB(A1,<{rel2}>, A2)",
                   f"Rule_2(A3) :- PKB(<{anchor3}>,<{rel3}>,A3)"]

        if not os.path.exists(f'fb15k-237/{query_type}/{query_id}'):
            os.makedirs(f'fb15k-237/{query_type}/{query_id}')

        with open(f'fb15k-237/{query_type}/{query_id}/datalog.txt', 'w+') as f:
            f.writelines([rule + '\n' for rule in query])

    return datalog

def handle_2u(env, query_type, queries):
    datalog = []

    for query_id, query in enumerate(queries):

        anchor1, rel1, x1, anchor2, rel2, x2 = query.split('_')

        anchor1 = env.ent_id2fb[int(anchor1)]
        anchor2 = env.ent_id2fb[int(anchor2)]

        rel1 = env.rel_id2fb[int(rel1)]
        rel2 = env.rel_id2fb[int(rel2)]

        query = [f"Q{query_id}(V0) :- Rule_1(V0)",
                   f"Rule_1(A1) :- PKB(<{anchor1}>,<{rel1}>,A1)",
                   f"Rule_1(A2) :- PKB(<{anchor2}>,<{rel2}>,A2)"]
        if not os.path.exists(f'fb15k-237/{query_type}/{query_id}'):
            os.makedirs(f'fb15k-237/{query_type}/{query_id}')
        with open(f'fb15k-237/{query_type}/{query_id}/datalog.txt', 'w+') as f:
            f.writelines([rule + '\n' for rule in query])
    return datalog

def handle_uc(env, query_type, queries):
    datalog = []

    for query_id, query in enumerate(queries):
        anchor1, rel1, x1, anchor2, rel2, x2, x3, rel3, x4 = query.split('_')

        anchor1 = env.ent_id2fb[int(anchor1)]
        anchor2 = env.ent_id2fb[int(anchor2)]

        rel1 = env.rel_id2fb[int(rel1)]
        rel2 = env.rel_id2fb[int(rel2)]
        rel3 = env.rel_id2fb[int(rel3)]

        query = [f"Q{query_id}(A1) :- PKB(V0,<{rel3}>, A1)",
                   f"QUnion(V0) :- Rule_1(V0)",
                   f"Rule_1(A2) :- PKB(<{anchor1}>,<{rel1}>,A2)",
                   f"Rule_1(A3) :- PKB(<{anchor2}>,<{rel2}>,A3)"]
        
        if not os.path.exists(f'fb15k-237/{query_type}/{query_id}'):
            os.makedirs(f'fb15k-237/{query_type}/{query_id}')
        with open(f'fb15k-237/{query_type}/{query_id}/datalog.txt', 'w+') as f:
            f.writelines([rule + '\n' for rule in query])
    
    
    return datalog 
