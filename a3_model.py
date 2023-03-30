import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

# The class of custom dataset
class DatasetAuthors(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

class AuthorClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, activation):
        super(AuthorClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_dim)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            self.activation = None



    def forward(self, x):
        x = self.layer1(x)
        if self.activation:
            x = self.activation(x)
        x = self.layer2(x)
        return torch.log_softmax(x, dim=1)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test a model on features.")
    parser.add_argument("featurefile", type=str, help="The file containing the table of instances and features.")
    parser.add_argument("outputfile", type=str, help="The file to write the output confusion matrix to.")
    parser.add_argument("--test", type=float, default=0.2, help="The proportion of the dataset to use as the test set.")
    parser.add_argument("--hidden_size", type=int, default=64, help="The size of the hidden layer.")
    parser.add_argument("--activation", type=str, default='none', choices=["relu", "sigmoid", "none"], help="The type of activation function to use.")


    args = parser.parse_args()

    print("Reading {}...".format(args.featurefile))

    # Read the data
    data = pd.read_csv("./output.csv")

    # Drop the 'test_train' column from the data
    data = data.drop("test_train", axis=1)

    # Convert the labels to numerical values
    label_encoder = LabelEncoder()
    data['author'] = label_encoder.fit_transform(data['author'])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data.drop("author", axis=1), data["author"], test_size=args.test)

    # Convert the X_train and X_test DataFrames to float32 data type
    X_train = X_train.astype(np.float32).to_numpy()
    X_test = X_test.astype(np.float32).to_numpy()

    # Create the train_dataset and test_dataset instances
    train_dataset = DatasetAuthors(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train.to_numpy(), dtype=torch.long))
    test_dataset = DatasetAuthors(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test.to_numpy(), dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    # Defining the model
    input_dim = X_train.shape[1]
    output_dim = len(np.unique(y_train))
    model = AuthorClassifier(input_dim, output_dim, args.hidden_size, args.activation)

    # Defining the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training the model
    epochs = 100
    for epoch in range(epochs):
        model.train()
        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    # Evaluating the model
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            outputs = model(batch_data)
            _, predictions = torch.max(outputs, 1)
            y_true.extend(batch_labels.tolist())
            y_pred.extend(predictions.tolist())

    # Computing confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix score:")
    print(cm)

    # Calculating accuracy
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    print(f"Accuracy score: {accuracy:.4f}")

    # Defining the model
    input_dim = X_train.shape[1]
    output_dim = len(np.unique(y_train))
    model = AuthorClassifier(input_dim, output_dim, args.hidden_size, args.activation)

    # Defining the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training the model
    epochs = 100
    for epoch in range(epochs):
        model.train()
        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    # Evaluating the model
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            outputs = model(batch_data)
            _, predictions = torch.max(outputs, 1)
            y_true.extend(batch_labels.tolist())
            y_pred.extend(predictions.tolist())

    # Computing confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix score:")
    print(cm)

    # Calculating accuracy
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    print(f"Accuracy score: {accuracy:.4f}")
