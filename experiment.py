from simpletransformers.classification import (ClassificationModel, ClassificationArgs)
import utils
import torch
import logging
import pandas as pd

cuda_available = torch.cuda.is_available()


# todo 加载和切割数据
def load_data(data_path):
    data = utils.read_data(data_path)
    df = pd.DataFrame(data)
    df.columns = ["text_a", "text_b", "labels"]
    return df


eval_data_path = "/home/guochenrui/smart_cite_con/data2019/eval_data.csv"
train_data_path = "/home/guochenrui/smart_cite_con/data2019/training_data.csv"

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    train_df = load_data(train_data_path)
    eval_df = load_data(eval_data_path)

    # Optional model configuration
    # weight_decay 1e-5 - 1e-2
    model_args = ClassificationArgs(num_train_epochs=20, learning_rate=2e-5)
    model_args.save_model_every_epoch = False
    model_args.save_eval_checkpoints = False

    model_args.use_early_stopping = True
    model_args.early_stopping_patience = 5
    model_args.weight_decay = 0
    model_args.wandb_project = "scc"
    model_args.output_dir = "outputs14"
    model_args.train_batch_size = 16


    # Create a ClassificationModel
    print("Is cuda available?: ", cuda_available)
    model = ClassificationModel("bert", "allenai/scibert_scivocab_cased", use_cuda=cuda_available, args=model_args)

    # Train the model
    model.train_model(train_df)

    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(eval_df)

    precision = result['tp']/(result['tp']+result['fp'])
    recall = result['tp']/(result['tp']+result['fn'])
    f1 = 2*precision*recall/(precision+recall)

    print("mcc: %.2f %%" % (result['mcc'] * 100))
    print("auroc: %.2f %%" % (result['auroc'] * 100))
    print("auprc: %.2f %%" % (result['auprc'] * 100))
    print("eval_loss: %.2f %%" % (result['eval_loss'] * 100))
    print("precision: %.2f %%" % (precision*100))
    print("recall: %.2f %%" % (recall*100))
    print("f1: %.2f %%" % (f1*100))
