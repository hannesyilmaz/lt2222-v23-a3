# LT2222 V23 Assignment 3

- Running the a3_features.py script
To run the a3_features.py script, you'll need to provide the input directory containing the author folders, the name of the output file, and the output feature dimensions. Optionally, you can specify the percentage of instances to be labeled as test data.

- Prerequisites
Make sure you have Python installed on your system. The code was developed and tested with Python 3.10.

Install the required Python libraries: numpy, pandas, scikit-learn, and argparse. You can do this using pip:

Run;
pip install numpy pandas scikit-learn argparse

- Running the script
Navigate to the directory containing the a3_features.py script using the terminal or command prompt.

Run the script with the required arguments:

Use below command in the terminal;
'python a3_features.py [inputdir] [outputfile] [dims] [--test TESTSIZE]'
Replace the placeholders with the appropriate values:

[inputdir]: The root directory of the author folders.
[outputfile]: The name of the output file containing the table of instances.
[dims]: The output feature dimensions.
[--test TESTSIZE]: (Optional) The percentage (integer) of instances to label as test data. Defaults to 20.
For example:

Lastly use the following command in the terminal;
python a3_features.py /scratch/lt2222-v23/enron_sample output.csv 100 --test 20
The script will read the input directory, process the text files, and create a feature table that is saved to the specified output file.