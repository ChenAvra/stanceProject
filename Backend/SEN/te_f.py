def isnan(value):
  try:
      import math
      return math.isnan(float(value))
  except:
      return False

#matplotlib inline

import tensorflow as tf
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker
import urllib
import sys
import os
import csv
import zipfile
from tensorflow.python.framework import ops
import string
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer

tokenizer = PunktSentenceTokenizer()

# Constants setup
max_hypothesis_length, max_evidence_length = 27, 27
batch_size, vector_size, hidden_size = 128, 50, 64

lstm_size = hidden_size

weight_decay = 0.0001

learning_rate = 1

input_p, output_p = 0.5, 0.5

training_iterations_count = 100000

display_step = 10

def sent_process(sent):
    sent = sent.translate(str.maketrans('', '', string.punctuation))
    sent = [word for word in sent.split() if word.lower() not in stopwords.words('english')]
    return " ".join(sent)

def unzip_single_file(zip_file_name, output_file_name, output_file_path):
    """
        If the outFile is already created, don't recreate
        If the outFile does not exist, create it from the zipFile
    """
    if not os.path.isfile(output_file_name):
        with open(output_file_path, 'wb') as out_file:
            with zipfile.ZipFile(zip_file_name) as zipped:
                for info in zipped.infolist():
                    if output_file_name in info.filename:
                        with zipped.open(info) as requested_file:
                            out_file.write(requested_file.read())
                            return

def fixBadZipfile(zipFile):
    f = open(zipFile, 'r+b')
    data = f.read()
    pos = data.find('\x50\x4b\x05\x06') # End of central directory signature
    if (pos > 0):
         print("Trancating file at location " + str(pos + 22)+ ".")
         f.seek(pos + 22)   # size of 'ZIP end of central directory record'
         f.truncate()
         f.close()


def sentence2sequence(sentence,glove_wordmap):
    """

    - Turns an input sentence into an (n,d) matrix,
        where n is the number of tokens in the sentence
        and d is the number of dimensions each word vector has.

    """
    tokens = sentence.lower().split(" ")
    rows = []
    words = []
    # Greedy search for tokens
    for token in tokens:
        i = len(token)
        while len(token) > 0 and i > 0:
            word = token[:i]
            if word in glove_wordmap:
                rows.append(glove_wordmap[word])
                words.append(word)
                token = token[i:]
                i = len(token)
            else:
                i = i - 1
    return rows, words

def score_setup(row):
    convert_dict = {
        'entailment': 0,
        'neutral': 1,
        'contradiction': 2
    }
    score = np.zeros((3,))
    for x in range(1, 6):
        tag = row["label" + str(x)]
        if tag in convert_dict: score[convert_dict[tag]] += 1
    return score / (1.0 * np.sum(score))

def fit_to_size(matrix, shape):
    res = np.zeros(shape)
    slices = [slice(0, min(dim, shape[e])) for e, dim in enumerate(matrix.shape)]
    res[slices] = matrix[slices]
    return res

def split_data_into_scores(glove_wordmap):
    PROJECT_ROOT = os.path.abspath(__file__)
    BASE_DIR = os.path.dirname(PROJECT_ROOT)
    import csv
    with open(BASE_DIR+"\\snli_1.0_dev.txt", "r") as data:
        train = csv.DictReader(data, delimiter='\t')
        evi_sentences = []
        hyp_sentences = []
        labels = []
        scores = []
        for row in train:
            hyp_sentences.append(np.vstack(
                sentence2sequence(row["sentence1"].lower(),glove_wordmap)[0]))
            evi_sentences.append(np.vstack(
                sentence2sequence(row["sentence2"].lower(),glove_wordmap)[0]))
            labels.append(row["gold_label"])
            scores.append(score_setup(row))

        hyp_sentences = np.stack([fit_to_size(x, (max_hypothesis_length, vector_size))
                                  for x in hyp_sentences])
        evi_sentences = np.stack([fit_to_size(x, (max_evidence_length, vector_size))
                                  for x in evi_sentences])
        return (hyp_sentences, evi_sentences), labels, np.array(scores)

def training_model():
    PROJECT_ROOT = os.path.abspath(__file__)
    BASE_DIR = os.path.dirname(PROJECT_ROOT)
    glove_zip_file = BASE_DIR+"\\glove.6B.zip"
    glove_vectors_file = BASE_DIR+"\\glove.6B.50d.txt"
    glove_vectors_file_name = BASE_DIR+"\\glove.6B.50d.txt"

    snli_zip_file = BASE_DIR+"\\snli_1.0.zip"
    snli_dev_file = BASE_DIR+"\\snli_1.0_dev.txt"
    snli_dev_file_name = "snli_1.0_dev.txt"
    snli_full_dataset_file = "snli_1.0_train.txt"

    from six.moves.urllib.request import urlretrieve

    # large file - 862 MB
    if (not os.path.isfile(glove_zip_file) and
            not os.path.isfile(glove_vectors_file)):
        urlretrieve("http://nlp.stanford.edu/data/glove.6B.zip",
                    glove_zip_file)

    # # medium-sized file - 94.6 MB
    if (not os.path.isfile(snli_zip_file) and
            not os.path.isfile(snli_dev_file)):
        urlretrieve("https://nlp.stanford.edu/projects/snli/snli_1.0.zip",
                    snli_zip_file)

    unzip_single_file(glove_zip_file, glove_vectors_file_name, glove_vectors_file)
    unzip_single_file(snli_zip_file,snli_dev_file_name , snli_dev_file)

    glove_wordmap = {}
    with open(glove_vectors_file, encoding="utf8") as glove:
        for line in glove:
            name, vector = tuple(line.split(" ", 1))
            glove_wordmap[name] = np.fromstring(vector, sep=" ")

    rnn_size = 64
    rnn = tf.keras.layers.SimpleRNNCell(rnn_size)

    data_feature_list, correct_values, correct_scores = split_data_into_scores(glove_wordmap)

    l_h, l_e = max_hypothesis_length, max_evidence_length
    N, D, H = batch_size, vector_size, hidden_size
    l_seq = l_h + l_e
    ops.reset_default_graph()
    # lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    lstm = tf.keras.layers.LSTMCell(lstm_size)
    lstm_drop = tf.compat.v1.nn.rnn_cell.DropoutWrapper(lstm, input_p, output_p)

    # N: The number of elements in each of our batches,
    #   which we use to train subsets of data for efficiency's sake.
    # l_h: The maximum length of a hypothesis, or the second sentence.  This is
    #   used because training an RNN is extraordinarily difficult without
    #   rolling it out to a fixed length.
    # l_e: The maximum length of evidence, the first sentence.  This is used
    #   because training an RNN is extraordinarily difficult without
    #   rolling it out to a fixed length.
    # D: The size of our used GloVe or other vectors.
    tf.compat.v1.disable_eager_execution()
    hyp = tf.compat.v1.placeholder(tf.float32, [N, l_h, D], 'hypothesis')
    evi = tf.compat.v1.placeholder(tf.float32, [N, l_e, D], 'evidence')
    y = tf.compat.v1.placeholder(tf.float32, [N, 3], 'label')
    # hyp: Where the hypotheses will be stored during training.
    # evi: Where the evidences will be stored during training.
    # y: Where correct scores will be stored during training.

    # lstm_size: the size of the gates in the LSTM,
    #    as in the first LSTM layer's initialization.
    # lstm_back = tf.contrib.rnn.BasicLSTMCell(lstm_size)

    lstm_back = tf.compat.v1.nn.rnn_cell.LSTMCell(lstm_size)
    # lstm_back:  The LSTM used for looking backwards
    #   through the sentences, similar to lstm.

    # input_p: the probability that inputs to the LSTM will be retained at each
    #   iteration of dropout.
    # output_p: the probability that outputs from the LSTM will be retained at
    #   each iteration of dropout.
    lstm_drop_back = tf.compat.v1.nn.rnn_cell.DropoutWrapper(lstm_back, input_p, output_p)
    # lstm_drop_back:  A dropout wrapper for lstm_back, like lstm_drop.

    fc_initializer = tf.random_normal_initializer(stddev=0.1)
    # fc_initializer: initial values for the fully connected layer's weights.
    # hidden_size: the size of the outputs from each lstm layer.
    #   Multiplied by 2 to account for the two LSTMs.
    fc_weight = tf.compat.v1.get_variable('fc_weight', [2 * hidden_size, 3],
                                          initializer=fc_initializer)
    # fc_weight: Storage for the fully connected layer's weights.
    fc_bias = tf.compat.v1.get_variable('bias', [3])
    # fc_bias: Storage for the fully connected layer's bias.

    # tf.GraphKeys.REGULARIZATION_LOSSES:  A key to a collection in the graph
    #   designated for losses due to regularization.
    #   In this case, this portion of loss is regularization on the weights
    #   for the fully connected layer.
    tf.compat.v1.add_to_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES,
                                   tf.nn.l2_loss(fc_weight))

    x = tf.concat([hyp, evi], 1)  # N, (Lh+Le), d
    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])  # (Le+Lh), N, d
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, vector_size])  # (Le+Lh)*N, d
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(x, l_seq, )

    # x: the inputs to the bidirectional_rnn

    # tf.contrib.rnn.static_bidirectional_rnn: Runs the input through
    #   two recurrent networks, one that runs the inputs forward and one
    #   that runs the inputs in reversed order, combining the outputs.
    rnn_outputs, _, _ = tf.compat.v1.nn.static_bidirectional_rnn(lstm, lstm_back, x, dtype=tf.float32)
    # rnn_outputs: the list of LSTM outputs, as a list.
    #   What we want is the latest output, rnn_outputs[-1]

    classification_scores = tf.matmul(rnn_outputs[-1], fc_weight) + fc_bias
    # The scores are relative certainties for how likely the output matches
    #   a certain entailment:
    #     0: Positive entailment
    #     1: Neutral entailment
    #     2: Negative entailment

    with tf.compat.v1.variable_scope('Accuracy'):
        predicts = tf.cast(tf.argmax(classification_scores, 1), 'int32')
        y_label = tf.cast(tf.argmax(y, 1), 'int32')
        corrects = tf.equal(predicts, y_label)
        num_corrects = tf.reduce_sum(tf.cast(corrects, tf.float32))
        accuracy = tf.reduce_mean(tf.cast(corrects, tf.float32))

    with tf.compat.v1.variable_scope("loss"):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits=classification_scores, labels=y)
        loss = tf.reduce_mean(cross_entropy)
        total_loss = loss + weight_decay * tf.add_n(
            tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES))

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)

    opt_op = optimizer.minimize(total_loss)

    # Initialize variables
    init = tf.compat.v1.global_variables_initializer()

    # Use TQDM if installed
    tqdm_installed = False
    try:
        from tqdm import tqdm
        tqdm_installed = True
    except:
        pass

    # Launch the Tensorflow session
    sess = tf.compat.v1.Session()
    sess.run(init)

    # training_iterations_count: The number of data pieces to train on in total
    # batch_size: The number of data pieces per batch
    training_iterations = range(0, training_iterations_count, batch_size)
    if tqdm_installed:
        # Add a progress bar if TQDM is installed
        training_iterations = tqdm(training_iterations)

    for i in training_iterations:

        # Select indices for a random data subset
        batch = np.random.randint(data_feature_list[0].shape[0], size=batch_size)

        # Use the selected subset indices to initialize the graph's
        #   placeholder values
        hyps, evis, ys = (data_feature_list[0][batch, :],
                          data_feature_list[1][batch, :],
                          correct_scores[batch])

        # Run the optimization with these initialized values
        sess.run([opt_op], feed_dict={hyp: hyps, evi: evis, y: ys})
        # display_step: how often the accuracy and loss should
        #   be tested and displayed.
        if (i / batch_size) % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={hyp: hyps, evi: evis, y: ys})
            # Calculate batch loss
            tmp_loss = sess.run(loss, feed_dict={hyp: hyps, evi: evis, y: ys})
            # Display results
            print("Iter " + str(i / batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(tmp_loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
    return sess, glove_wordmap,classification_scores, hyp, evi,N, y

def ent_feature_extraction(sess, glove_wordmap, classification_scores, hyp, evi, N, y, csvFilePath, target, outputFile):
    Features_pmh = pd.read_csv(csvFilePath, header=0)

    length_features = len(Features_pmh)
    result = []
    pred = []

    text = Features_pmh['Sentence'].copy()

    Features_pmh['Sentence'] = text.apply(sent_process)
    file = open(outputFile, 'w',encoding='utf-8')
    fields = ('Text', 'hypotheses', 'result', 'pos_scr', 'neg_scr', 'nut_scr')
    wr = csv.DictWriter(file, fieldnames=fields, lineterminator='\n')
    wr.writeheader()

    for i in range(length_features):
        if (isnan(Features_pmh['Sentence'][i]) == False):
            evidences = [Features_pmh['Sentence'][i]]
        else:
            evidences = [Features_pmh['Sentence'][i]]
        if(evidences[0]!=''):
            hypotheses = [Features_pmh['Claim'][i]]
            # hypotheses = [target]

            sentence2 = [fit_to_size(np.vstack(sentence2sequence(evidence,glove_wordmap)[0]), (max_evidence_length, vector_size)) for
                         evidence in evidences]

            sentence1 = [fit_to_size(np.vstack(sentence2sequence(hypothesis,glove_wordmap)[0]), (max_hypothesis_length, vector_size)) for
                         hypothesis in hypotheses]

            prediction = sess.run(classification_scores,
                                  feed_dict={hyp: (sentence1 * N), evi: (sentence2 * N), y: [[0, 0, 0]] * N})
            # print(["Positive", "Neutral", "Negative"][np.argmax(prediction[0])]+" entailment")
            result.append(["Positive", "Neutral", "Negative"][np.argmax(prediction[0])])
            pred.append(prediction[0])
            wr.writerow(
                {'Text': Features_pmh['Sentence'][i], 'hypotheses': hypotheses, "result": result[i], 'pos_scr': pred[i][0],
                 'neg_scr': pred[i][2], 'nut_scr': pred[i][1], })
        else:
            print("adding zeros because the input doesn't fit the library, i is:", i)
            pred.append([0,0,0])
            result.append("Neutral")
            print(Features_pmh['Sentence'][i])
            print(Features_pmh['Claim'][i])
            wr.writerow(
                {'Text': Features_pmh['Sentence'][i], 'hypotheses': [Features_pmh['Claim'][i]], "result": "Neutral",
                 'pos_scr': '0',
                 'neg_scr': '0', 'nut_scr': '0', })



    file.close()
    return sess

def run_te_f_feature_extraction():
    sess, glove_wordmap,classification_scores, hyp, evi,N, y = training_model()
    import os
    PROJECT_ROOT = os.path.abspath(__file__)
    BASE_DIR = os.path.dirname(PROJECT_ROOT)
    arr = os.listdir(BASE_DIR+"\\topics")
    for topic in arr:
        for k in ["train","test"]:
            csvFileToRead= BASE_DIR + "\\topics\\" + topic + "\\" + k + ".csv"
            outputFileToWrite = BASE_DIR+"\\topics\\"+topic+"\\ent_feature_extraction_"+k+".csv"
            sess=ent_feature_extraction(sess, glove_wordmap, classification_scores, hyp, evi, N, y, csvFileToRead, topic, outputFileToWrite)
    sess.close()
