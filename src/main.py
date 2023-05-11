from config.cmd_args import cmd_args
from apqa import ApproximateProbabilisticQueryAnswerer as APQA
from metrics import evaluation
from dataset import Dataset


def run(embedding_model, edb, top_k, dataset):
   
    apqa = APQA(embedding_model, edb, k=top_k, dataset=dataset)
    scores = apqa.answer_queries()
    metrics = evaluation(scores, dataset)
    print("metrics", metrics)
    

    # TODO add logger file results


if __name__ == "__main__":
    

    embedding_model = cmd_args.embedding_model_path
    edb_config = cmd_args.edb_config_path
    top_k = int(cmd_args.top_k)
    data_path = cmd_args.data_path

    dataset = Dataset(data_path)




    run(embedding_model=embedding_model, edb=edb_config, top_k=top_k, dataset=dataset)
