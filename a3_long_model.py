import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import LongformerTokenizer, LongformerForSequenceClassification, Trainer, TrainingArguments

# Load the data
data = pd.read_csv('output.csv')
X = data.drop(["author", "test_train"], axis=1)
reduced_feature_columns = X.shape[1] - 1  # Subtract 1 to exclude the "email_body" column
feature_columns = X.columns[:reduced_feature_columns]
X = X.drop(columns=feature_columns)

y = data["author"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Encode the labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# Initialize the Longformer tokenizer
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")

# Tokenize the text
train_encodings = tokenizer(X_train.to_numpy(dtype=str).ravel().tolist(), truncation=True, padding=True)
test_encodings = tokenizer(X_test.to_numpy(dtype=str).ravel().tolist(), truncation=True, padding=True)

# Prepare the dataset
class AuthorDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create dataset
train_dataset = AuthorDataset(train_encodings, y_train_encoded)
test_dataset = AuthorDataset(test_encodings, y_test_encoded)

# Initialize the Longformer model
model = LongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096", num_labels=len(set(y_train_encoded)))

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=8,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Train the model
trainer.train()

# Evaluate the model
trainer.evaluate()


# Evaluate the model and print the evaluation metrics
eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)

# Get the predictions and true labels for the test dataset
predictions, true_labels, _ = trainer.predict(test_dataset)

# The predictions are in logits format; convert them to class labels
predicted_labels = np.argmax(predictions, axis=1)

# Decode the predicted and true labels back to the original author names
predicted_authors = le.inverse_transform(predicted_labels)
true_authors = le.inverse_transform(true_labels)

# Print and store the predicted and true author names
print("Predicted authors:", predicted_authors)
print("True authors:", true_authors)
