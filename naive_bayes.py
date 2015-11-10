from __future__ import division
from collections import defaultdict
import sys
import math


def train(data):
    pos_counts = defaultdict(int)
    neg_counts = defaultdict(int)
    example_counts = {'neg': 0, 'pos': 0, 'total': 0}

    vocab_size = 11389

    for line in data:
        example = line.split(' ')
        if int(example[0]) > 0:
            examples = pos_counts
            example_counts['pos'] += 1
        else:
            examples = neg_counts
            example_counts['neg'] += 1
        example_counts['total'] += 1

        for attribute in example[1:]:
            word, count = [int(x) for x in attribute.split(":")]

            examples['total'] += count
            examples[word] += count

            if word > vocab_size:
                vocab_size = word

    return pos_counts, neg_counts, example_counts, vocab_size


def test(data, c10, c01):

    false_pos = 0
    false_neg = 0
    total_tests = 0
    for line in data:
        total_tests += 1
        example = line.split(' ')
        pos_prior = example_counts['pos'] / example_counts['total']
        neg_prior = example_counts['neg'] / example_counts['total']

        pos_total = 0
        for attribute in example[1:]:
            word, count = [int(x) for x in attribute.split(":")]
            occurances = pos_counts[word]
            conditional = (occurances + 1) / (pos_counts['total'] + vocab_size)
            pos_total += math.log(conditional) * count

        neg_total = 0
        for attribute in example[1:]:
            word, count = [int(x) for x in attribute.split(":")]
            occurances = neg_counts[word]
            conditional = (occurances + 1) / (neg_counts['total'] + vocab_size)
            neg_total += math.log(conditional) * count

        pos_class = pos_total + math.log(pos_prior)
        neg_class = neg_total + math.log(neg_prior)

        if (int(example[0]) > 0) != (pos_class >= neg_class):
            if math.log(c10) + pos_class >= math.log(c01) + neg_class:
                false_pos += 1
            else:
                false_neg += 1
    return false_pos, false_neg, total_tests


def find_weights(vocab):
    large = [(0, 0) for x in xrange(10)]
    small = [(0, float('inf')) for x in xrange(10)]
    for word in set(pos_counts.keys() + neg_counts.keys()):
        pos_occur = pos_counts[word]
        neg_occur = neg_counts[word]
        pos_condit = (pos_occur + 1) / (pos_counts['total'] + vocab_size)
        neg_condit = (neg_occur + 1) / (neg_counts['total'] + vocab_size)
        odds = math.log(pos_condit) - math.log(neg_condit)
        i = 9
        while odds > large[i][1] and i >= 0:
            i -= 1
        if i < 9:
            large.pop()
            large.insert(i + 1, (vocab[word], odds))

        i = 9
        while odds < small[i][1] and i >= 0:
            i -= 1
        if i < 9:
            small.pop()
            small.insert(i + 1, (vocab[word], odds))
    return large, small


if __name__ == '__main__':

    if len(sys.argv) < 3:
        print "Not enough arguments\n"
        print "\tusage: python naive_bayes.py training_file <mode>\n"
        print "Modes:"
        print "\t--test test_file [-c c10 c01]\t-> Classifies the testing data"
        print "\t\t\t\t\t   with the [optional] given cost values"
        print "\t--odds vocab_file\t\t-> Identifies the ten most positively and"
        print "\t\t\t\t\t   negatively associated words in the"
        print "\t\t\t\t\t   training data"
    else:
        train_path = sys.argv[1]
        training_data = open(train_path)

        pos_counts, neg_counts, example_counts, vocab_size = train(training_data)

        training_data.close()

        print "Training successful on %d examples\n" % (example_counts['total'])

        if sys.argv[2] == '--test':
            test_path = sys.argv[3]
            testing_data = open(test_path)
            if len(sys.argv) > 4 and sys.argv[4] == '-c':
                c10, c01 = [float(x) for x in sys.argv[5:7]]
            else:
                c10 = c01 = 1

            print "Using cost values c10 = %d and c01 = %d" % (c10, c01)

            false_pos, false_neg, total_tests = test(testing_data, c10, c01)

            print "Testing successful on %d examples\n" % (total_tests)
            print "False Positives: %d" % (false_pos)
            print "False Negatives: %d" % (false_neg)
            error = (false_neg + false_pos) / total_tests
            print "Accuracy: %f" % (1 - error)

        if sys.argv[2] == '--odds':
            vocab_path = sys.argv[3]
            vocab_handle = open(vocab_path)
            vocab = []

            for line in vocab_handle:
                vocab.append(line.strip())

            large, small = find_weights(vocab)
            print "\nWords with largest log-odds:"
            for w in large:
                print "\t%s - %f" % (w[0], w[1])
            print "\nWords with smallest log-odds:"
            for w in small:
                print "\t%s - %f" % (w[0], w[1])
