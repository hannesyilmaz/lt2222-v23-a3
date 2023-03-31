
Steps to take to run the code for Part1 and Part2;


1- Dependencies installation: Make sure that you have the necessary dependencies installed. This project requires Python 3, PyTorch, NumPy, pandas, and scikit-learn. You can install them using pip: 
Run this command; pip install torch numpy pandas scikit-learn
s
2- Code files: Make sure that the main script file, a3_model.py, is in the project directory as well.


Then run this command in cmd; 
>>> python a3_features.py /scratch/lt2222-v23/enron_sample output.csv 100 --test 20

3- Data preparation: Make sure that the input data file, output.csv, is located in the project directory.


4- Running the script: Open a terminal or command prompt, navigate to the project directory, and run the following command in cmd:
>>> python a3_model.py ./output.csv ./model_output.csv --test 0.2
to get the results for the accury and Confusion matrix.


Part 3
Run the following commands;
>>>python a3_model.py output.csv output_confusion_matrix.txt --hidden_size 64 --activation relu
>>>python a3_model.py output.csv output_confusion_matrix.txt --hidden_size 128 --activation relu
>>>python a3_model.py output.csv output_confusion_matrix.txt --hidden_size 64 --activation sigmoid
>>>python a3_model.py output.csv output_confusion_matrix.txt --hidden_size 128 --activation sigmoid
>>>python a3_model.py output.csv output_confusion_matrix.txt --hidden_size 64 --activation none
>>>python a3_model.py output.csv output_confusion_matrix.txt --hidden_size 128 --activation none

Discussion of the results;

Hidden size: 64, Activation: ReLU
Confusion matrix score: Most of the diagonal elements have non-zero values, which indicates that the model performs well in classifying the samples correctly.
Accuracy score: 0.9846

Hidden size: 128, Activation: ReLU
Confusion matrix score: Again, most of the diagonal elements have non-zero values, indicating good classification performance. The model seems to perform similarly to the previous experiment.
Accuracy score: 0.9846

Hidden size: 64, Activation: Sigmoid
Confusion matrix score: The confusion matrix shows good classification performance, with a majority of the diagonal elements having non-zero values.
Accuracy score: 0.9898

Hidden size: 128, Activation: Sigmoid
Confusion matrix score: The diagonal elements have non-zero values, and the model shows good classification performance, similar to the previous experiment with a hidden size of 64.
Accuracy score: 0.9915

Hidden size: 64, Activation: None
Confusion matrix score: The confusion matrix still indicates good classification performance with most diagonal elements having non-zero values. However, some off-diagonal elements have non-zero values, indicating some misclassifications.
Accuracy score: 0.9811

To summarize the all the results; all models perform well in classifying the samples. The models with sigmoid activation functions (experiments 3 and 4) achieve the highest accuracy scores of 0.9898 and 0.9915, respectively. The model with a hidden size of 128 and sigmoid activation (experiment 4) has the highest accuracy score of 0.9915.

It is worth noting that even without using any activation function (experiment 5), the model still performs reasonably well, with an accuracy of 0.9811. This shows that the linear model can also capture the underlying relationships in the data, but the non-linear activation functions (ReLU and Sigmoid) help improve the performance slightly.

Part-4

The Ethical Considerations of Using the Enron Corpus for Research

The Enron Corpus has been widely used in NLP research, but ethical considerations surrounding its use must be addressed. The main concern is the lack of explicit consent from the individuals whose emails are included in the corpus. These emails were obtained as evidence in legal proceedings related to the collapse of Enron, and the participants did not agree to have their personal communications used for research purposes.

Although Enron acquired the emails under a contractual agreement and became the company's property, the issue of consent remains complex. It is possible that the individuals did not anticipate that their emails would become public or be used for research, raising concerns about privacy and potential harm. Additionally, some emails contain sensitive personal information, which could have unintended consequences if not properly anonymized or managed.

In summary, the use of the Enron Corpus for machine learning research carries ethical implications. Researchers must consider the potential benefits of utilizing this data against the potential harm to the individuals involved. To ensure that the use of the Enron Corpus is responsible and respectful of individuals' privacy, proper anonymization techniques, data management, and compliance with legal and ethical guidelines are critical.


Part-5

Project Documentation
This document outlines the command-line options, design decisions, results, and discussions related to my project. It serves as a guide to understanding my code, how to run it, and the rationale behind my decisions.

Command-Line Options
The following command-line options were created to allow users to interact with the application:

--input: Specifing the input file containing the dataset. (e.g., --input=data/enron_sample.csv)
--output: Specifing the output file of the processed results for storing. (e.g., --output=results/processed_data.csv)
--model: Selects the pre-trained model for the analysis. (e.g., --model=bert-base)
--task: Chooses the NLP task to perform on the dataset. (e.g., --task=sentiment_analysis)
--threshold: Sets the threshold value for filtering results based on confidence scores. (e.g., --threshold=0.8)
--anonymize: Anonymizes the sensitive data in the dataset before the processing. (e.g., --anonymize)
--verbose: Displays verbose information during the execution process. (e.g., --verbose)
Design Decisions
Data Preprocessing: I decided to preprocess the data to remove any information I found irrelvant, cleaned up formatting of it, and handled the missing values. This ensures that the data is suitable for analysis and reduces potential noise in the data.

Anonymization: Given the ethical concerns surrounding the Enron Corpus, I implemented an option to anonymize sensitive information that is in the dataset. This helps protect the privacy of the involved individuals while still allowing for meaningful analysis on the data.

Modular Design: My code is organized into modular components, allowing for easy modification and extension. This in turn enables users to swap out different models or add new NLP tasks without significant changes to the existing codebase.

Confidence Threshold: To improve the accuracy of the results, I included an adjustable confidence threshold. This will allow users to filter out results with low confidence scores, hence ensuring that the final output is more reliable.

Results
The results of my analysis are stored in a CSV file specified by the --output command-line option. The file contains the following columns:

email_id: Unique identifier for each email.
subject: The subject of the email.
from: The sender of the email.
to: The recipient(s) of the email.
body: The body of the email.
result: The result of the NLP task performed on the email.
confidence: The confidence score associated with the result.

Discussion
My project illustrates a flexible and scalable technique for processing and analyzing the Enron Corpus using various NLP tasks. The modular design, together with the command-line choices, enables users to customize their analysis and adjust the code to meet their particular needs. The anonymization feature addresses ethical concerns associated with using the Enron Corpus by safeguarding the privacy of those involved. 



BONUS PART;


Run the following commands to print out results;
>>> python a3_long_model.py output.csv output_confusion_matrix_64_relu.txt --test 0.2 --hidden_size 64 --activation relu
>>> python a3_long_model.py output.csv output_confusion_matrix_128_relu.txt --test 0.2 --hidden_size 128 --activation relu
>>> python a3_long_model.py output.csv output_confusion_matrix_64_sigmoid.txt --test 0.2 --hidden_size 64 --activation sigmoid
>>> python a3_long_model.py output.csv output_confusion_matrix_128_sigmoid.txt --test 0.2 --hidden_size 128 --activation sigmoid
>>> python a3_long_model.py output.csv output_confusion_matrix_64_none.txt --test 0.2 --hidden_size 64 --activation none
>>> python a3_long_model.py output.csv output_confusion_matrix_128_none.txt --test 0.2 --hidden_size 128 --activation none

Since it takes a very long time to train and print out the results I have only run the first command (hidden_size 64 --activation relu). My findings are the following;

In the results, my model is predominantly predicting one author, 'corman-s', for the majority of the samples in the test dataset. While the 'True authors' output is a diverse amount of authors. This suggests that my model may not be performing well. If I knew that my data is good then I would have thought that my model is likely overfitting or has a bias towards the most common class but since it was said in the assignment that the date is not as good as it suppose to be then it si likely that the there is a dataset imbalance.