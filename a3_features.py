import os
import sys
import argparse
import numpy as np
import pandas as pd
import re
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder

def remove_headers_and_signatures(text):
    # Removing headers
    text = re.sub(r"(?i)^(From|To|Cc|Bcc|Subject|Sent|Date): .+\n?", "", text, flags=re.MULTILINE)

    # Removing signature lines
    text = re.sub(r"(?i)^(--|\.\.\.|\*\*\*)\n?.*?$", "", text, flags=re.MULTILINE)

    return text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert directories into table.")
    parser.add_argument("inputdir", type=str, help="The root of the author directories.")
    parser.add_argument("outputfile", type=str, help="The name of the output file containing the table of instances.")
    parser.add_argument("dims", type=int, help="The output feature dimensions.")
    parser.add_argument("--test", "-T", dest="testsize", type=int, default="20", help="The percentage (integer) of instances to label as test.")
    args = parser.parse_args()

    print("Reading {}...".format(args.inputdir))
    # Do what you need to read the documents here.

    documents = []
    authors = []
    for author in os.listdir(args.inputdir):
        author_path = os.path.join(args.inputdir, author)
        if os.path.isdir(author_path):
            # Iterate over each text file inside the author directory
            for text_file in os.listdir(author_path):
                with open(os.path.join(author_path, text_file), "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                    text = remove_headers_and_signatures(text)
                    documents.append(text)
                    authors.append(author)

    print("Constructing table with {} feature dimensions and {}% test instances...".format(args.dims, args.testsize))
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(documents)

    svd = TruncatedSVD(n_components=args.dims)
    X_reduced = svd.fit_transform(X)

    test_labels = ["test" if random.random() < args.testsize / 100 else "train" for _ in range(len(documents))]

    table = pd.DataFrame(X_reduced)
    table["email_body"] = documents
    table["author"] = authors
    table["test_train"] = test_labels
    
    # Removing the "email_body" column before saving the data
    table = table.drop("email_body", axis=1)

    # Loading the data
    data = pd.read_csv(args.outputfile)
    

    # Converting the labels to numerical values
    label_encoder = LabelEncoder()
    data['author'] = label_encoder.fit_transform(data['author'])

    print("Writing to {}...".format(args.outputfile))
    table.to_csv(args.outputfile, index=False)

    print("Done!")
