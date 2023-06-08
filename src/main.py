from config.cmd_args import cmd_args
from apqa import ApproximateProbabilisticQueryAnswerer as APQA
from metrics import evaluation
from dataset import Dataset
from utils import get_answers
import json
LOG_LEVEL = 0

def run(embedding_model_path, edb, top_k, threshold, dataset, rank_embedding_model, max_queries):
   
    apqa = APQA(embedding_model_path, edb, k=top_k, threshold=threshold, max_queries=max_queries, dataset=dataset)
    
    for query_type in dataset.query_types:
        scores = apqa.answer_queries(query_type)
        answers = get_answers(dataset.data_directory, query_type)
        metrics = evaluation(scores, answers, dataset, max_queries)
        # with open(f'topk_d={dataset.name}_e={query_type}_rank={rank_embedding_model}_k={top_k}.json', 'w+') as fp:
        #     json.dump(metrics, fp)

        print("metrics:", metrics)
    # TODO add logger file results


if __name__ == "__main__":
    

    embedding_model_path = cmd_args.embedding_model_path
    edb_config = cmd_args.edb_config_path
    top_k = int(cmd_args.top_k)
    threshold = float(cmd_args.threshold)
    max_queries = int(cmd_args.max_queries)

    rank_embedding_model = int(cmd_args.rank)
    data_path = cmd_args.data_path

    dataset = Dataset(data_path)


    run(embedding_model_path=embedding_model_path, edb=edb_config, top_k=top_k, threshold=threshold, dataset=dataset, rank_embedding_model=rank_embedding_model, max_queries=max_queries)
