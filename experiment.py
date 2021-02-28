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


eval_data_path = "E:\\code\\smart_cite_con\\data\\test_data.csv"
train_data_path = "E:\\code\\smart_cite_con\\data\\train_data.csv"

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    train_df = load_data(train_data_path)
    eval_df = load_data(eval_data_path)

    # Optional model configuration
    model_args = ClassificationArgs(num_train_epochs=1)

    # Create a ClassificationModel
    model = ClassificationModel("roberta", "roberta-base", use_cuda=cuda_available)

    # Train the model
    model.train_model(train_df)

    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(eval_df)