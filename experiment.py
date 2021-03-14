from simpletransformers.classification import (ClassificationModel, ClassificationArgs)
import utils
import torch
import logging
import pandas as pd
import sklearn
import wandb
import os.path as path_util

cuda_available = torch.cuda.is_available()
data_code = '2021031401'
wandb_project_name = "scc" + data_code
output_dir = "outputs"


# todo 加载和切割数据
def load_data(data_path):
    data = utils.read_data(data_path)
    df = pd.DataFrame(data)
    df.columns = ["text_a", "text_b", "labels"]
    return df


train_data_path = "/home/chenruiguo/code/smart_cite_con/data/data" + data_code + "/train_set.csv"
eval_data_path = "/home/chenruiguo/code/smart_cite_con/data/data" + data_code + "/eval_set.csv"
test_data_path = "/home/chenruiguo/code/smart_cite_con/data/data" + data_code + "/test_set.csv"


def early_stopping_setting(model_args: ClassificationArgs):
    model_args.use_early_stopping = True
    model_args.early_stopping_patience = 5
    model_args.early_stopping_delta = 0.01
    model_args.early_stopping_metric = "f1"
    model_args.early_stopping_metric_minimize = False
    model_args.evaluate_during_training = True
    model_args.evaluate_during_training_steps = 1000


def hyperparameter_setting(model_args: ClassificationArgs, learning_rate=5e-5, train_batch_size=32):
    model_args.num_train_epochs = 128
    model_args.learning_rate = learning_rate
    # weight_decay 1e-5 - 1e-2
    model_args.weight_decay = 0
    model_args.train_batch_size = train_batch_size
    # model_args.eval_batch_size = 32


def sweep_setting(model_args: ClassificationArgs):
    model_args.reprocess_input_data = True
    model_args.evaluate_during_training = True
    model_args.overwrite_output_dir = True
    model_args.use_multiprocessing = True
    model_args.manual_seed = 4


def other_setting(model_args: ClassificationArgs, output_dir="outputs"):
    model_args.wandb_project = wandb_project_name

    model_args.save_model_every_epoch = False
    model_args.save_eval_checkpoints = False

    model_args.overwrite_output_dir = True
    model_args.output_dir = output_dir
    model_args.save_best_model = True


def load_all_data():
    train_df = load_data(train_data_path)
    eval_df = load_data(eval_data_path)
    test_df = load_data(test_data_path)
    return train_df, eval_df, test_df


def init():
    train_df, eval_df, test_df = load_all_data()
    # Optional model configuration
    model_args = ClassificationArgs()
    hyperparameter_setting(model_args)
    other_setting(model_args)

    return train_df, eval_df, test_df, model_args


def output_wrong(wrong, path=output_dir):
    list = []
    path = path_util.join(path, "wrong_predictions.csv")
    for i in wrong:
        list.append([i.text_a, i.text_b, i.label])
    utils.write_data(list, path)


def output_eval_result(result, wrong=None):
    print("mcc: %.2f %%" % (result['mcc'] * 100))
    print("auroc: %.2f %%" % (result['auroc'] * 100))
    print("auprc: %.2f %%" % (result['auprc'] * 100))
    print("eval_loss: %.4f" % (result['eval_loss']))
    print("precision: %.2f %%" % (result['precision'] * 100))
    print("recall: %.2f %%" % (result['recall'] * 100))
    print("f1: %.2f %%" % (result['f1'] * 100))
    print(result["report"])
    print(result["confusion"])
    if wrong:
        output_wrong(wrong)


def train(train_df, eval_df, test_df, model_args):
    # Create a ClassificationModel
    # bert, bert-large-uncased bert-base-uncased bert-base-cased-finetuned-mrpc
    # xlnet，xlnet-base-cased xlnet-large-cased
    #
    model = ClassificationModel("bert", "bert-base-cased", use_cuda=cuda_available, args=model_args)

    # Train the model
    model.train_model(train_df, eval_df=eval_df, f1=sklearn.metrics.f1_score,
                      precision=sklearn.metrics.precision_score,
                      recall=sklearn.metrics.recall_score)

    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(test_df, f1=sklearn.metrics.f1_score,
                                                                confusion=sklearn.metrics.confusion_matrix,
                                                                report=sklearn.metrics.classification_report,
                                                                precision=sklearn.metrics.precision_score,
                                                                recall=sklearn.metrics.recall_score)
    return result, wrong_predictions


def raw_train():
    train_df, eval_df, test_df, model_args = init()
    return train(train_df, eval_df, test_df, model_args)


def train_with_early_stopping():
    train_df, eval_df, test_df, model_args = init()
    early_stopping_setting(model_args)
    return train(train_df, eval_df, test_df, model_args)


sweep_config = {
    "method": "grid",  # grid, random ,bayes
    "metric": {"name": "f1", "goal": "maximum"},
    "parameters": {
        # "num_train_epochs": {"values": [2, 3, 5]},
        "learning_rate": {"values": [1e-5, 2e-5, 3e-5, 4e-5, 5e-5]},
    },
}

sweep_id = wandb.sweep(sweep_config, project=wandb_project_name)


def train_with_sweep():
    wandb.init()
    train_df, eval_df, test_df, model_args = init()
    sweep_setting(model_args)
    train(train_df, eval_df, test_df, model_args)
    wandb.join()


train_batch_sizes = [8, 16, 32, 64, 128]
learning_rates = [1e-5, 2e-5, 3e-5, 4e-5, 5e-5]
output_dirs = ["output1", "outputs2", "outputs3", "outputs4", "outputs5"]


def batch_testing():
    train_df, eval_df, test_df = load_all_data()
    output_dirs.reverse()
    model_args = ClassificationArgs()
    early_stopping_setting(model_args)
    # for train_batch_size in train_batch_sizes:
    #     output_dir1 = output_dirs.pop()
    #     other_setting(model_args, output_dir1)
    #     hyperparameter_setting(model_args, train_batch_size=train_batch_size)
    #     _, wrong = train(train_df, eval_df, test_df, model_args)
    #     output_wrong(wrong, output_dir1)

    for learning_rate in learning_rates:
        output_dir1 = output_dirs.pop()
        other_setting(model_args, output_dir1)
        hyperparameter_setting(model_args, learning_rate=learning_rate)
        train(train_df, eval_df, test_df, model_args)
        _, wrong = train(train_df, eval_df, test_df, model_args)
        output_wrong(wrong, output_dir1)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    # 仅训练和评估
    # result, wrong_predictions = raw_train()

    # 使用early stopping
    # result = train_with_early_stopping()
    # output_eval_result(result, wrong_predictions)

    batch_testing()
    # 使用sweep
    # wandb.agent(sweep_id, train_with_sweep())
