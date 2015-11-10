Naive Bayes Classification
==========================
usage:
-----
python naive_bayes.py training_file <mode>

modes:
-----
--test test_file [-c c10 c01] --> Classifies the testing data, cost values c10 and c01 defailt to 1
--odds vocab_file -> Identifies the ten most positively and negatively associated words in the training data