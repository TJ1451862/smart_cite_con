from simpletransformers.classification import (ClassificationModel, ClassificationArgs)
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# Preparing train data
train_data = [
    [
        "Aragorn was the heir of Isildur",
        "Gimli fought with a battle axe",
        1,
    ],
    [
        "Frodo was the heir of Isildur",
        "Legolas was an expert archer",
        0,
    ],
]

train_df = pd.DataFrame(train_data)
train_df.columns = ["text_a", "text_b", "labels"]

# Preparing eval data
eval_data = [
    [
        "Theoden was the king of Rohan",
        "Gimli's preferred weapon was a battle axe",
        1,
    ],
    [
        "Merry was the king of Rohan",
        "Legolas was taller than Gimli",
        0,
    ],
]
eval_df = pd.DataFrame(eval_data)
eval_df.columns = ["text_a", "text_b", "labels"]

# Optional model configuration
model_args = ClassificationArgs(num_train_epochs=1)

# Create a ClassificationModel
model = ClassificationModel("roberta", "roberta-base", use_cuda=False)

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)

# Make predictions with the model
predictions, raw_outputs = model.predict(
    [
        [
            "Legolas was an expert archer",
            "Legolas was taller than Gimli",
        ]
    ]
)
