from simpletransformers.classification import (ClassificationModel, ClassificationArgs)
import utils
import torch
import logging
import pandas as pd
import sklearn
import wandb

cuda_available = torch.cuda.is_available()


# todo 加载和切割数据
def load_data(data_path):
    data = utils.read_data(data_path)
    df = pd.DataFrame(data)
    df.columns = ["text_a", "text_b", "labels"]
    return df


eval_data_path = "/home/guochenrui/smart_cite_con/data2019/eval_data.csv"
train_data_path = "/home/guochenrui/smart_cite_con/data2019/training_data.csv"


def early_stopping_setting(model_args: ClassificationArgs):
    model_args.use_early_stopping = True
    model_args.early_stopping_patience = 5
    model_args.early_stopping_delta = 0.01
    model_args.early_stopping_metric = "f1"
    model_args.early_stopping_metric_minimize = False
    model_args.evaluate_during_training = True
    model_args.evaluate_during_training_steps = 1000


def hyperparameter_setting(model_args: ClassificationArgs):
    model_args.num_train_epochs = 64
    model_args.learning_rate = 2e-5
    # weight_decay 1e-5 - 1e-2
    model_args.weight_decay = 0
    model_args.train_batch_size = 8
    # model_args.eval_batch_size = 32


def sweep_setting(model_args: ClassificationArgs):
    model_args.reprocess_input_data = True
    model_args.evaluate_during_training = True
    model_args.overwrite_output_dir = True
    model_args.use_multiprocessing = True
    model_args.manual_seed = 4


def other_setting(model_args: ClassificationArgs):
    model_args.wandb_project = "scc"

    model_args.save_model_every_epoch = False
    model_args.save_eval_checkpoints = False

    model_args.overwrite_output_dir = True
    model_args.output_dir = "outputs"


def init():
    train_df = load_data(train_data_path)
    eval_df = load_data(eval_data_path)

    # Optional model configuration
    model_args = ClassificationArgs()
    hyperparameter_setting(model_args)
    other_setting(model_args)

    return train_df, eval_df, model_args


def output_eval_result(result):
    print("mcc: %.2f %%" % (result['mcc'] * 100))
    print("auroc: %.2f %%" % (result['auroc'] * 100))
    print("auprc: %.2f %%" % (result['auprc'] * 100))
    print("eval_loss: %.2f %%" % (result['eval_loss'] * 100))
    print("precision: %.2f %%" % (result['precision'] * 100))
    print("recall: %.2f %%" % (result['recall'] * 100))
    print("f1: %.2f %%" % (result['f1'] * 100))

def train(train_df, eval_df, model_args):
    # Create a ClassificationModel
    model = ClassificationModel("bert", "allenai/scibert_scivocab_cased", use_cuda=cuda_available, args=model_args)

    # Train the model
    model.train_model(train_df, eval_df=eval_df, f1=sklearn.metrics.f1_score, precision=sklearn.metrics.precision_score,
                      recall=sklearn.metrics.recall_score)

    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(eval_df, f1=sklearn.metrics.f1_score,
                                                                precision=sklearn.metrics.precision_score,
                                                                recall=sklearn.metrics.recall_score)
    return result


def raw_train():
    train_df, eval_df, model_args = init()
    return train(train_df, eval_df, model_args)


def train_with_early_stopping():
    train_df, eval_df, model_args = init()
    early_stopping_setting(model_args)
    return train(train_df, eval_df, model_args)


sweep_config = {
        "method": "bayes",  # grid, random
        "metric": {"name": "f1", "goal": "maximum"},
        "parameters": {
            "num_train_epochs": {"values": [2, 3, 5]},
            "learning_rate": {"min": 1e-5, "max": 5e-5},
        },
    }

sweep_id = wandb.sweep(sweep_config, project="scc")


def train_with_sweep():
    wandb.init()
    train_df, eval_df, model_args = init()
    sweep_setting(model_args)
    train(train_df, eval_df, model_args)
    wandb.join()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    # 仅训练和评估
    # result = raw_train()

    # 使用early stopping
    result = train_with_early_stopping()
    print(result)

    # 使用sweep
    wandb.agent(sweep_id, train_with_sweep())
