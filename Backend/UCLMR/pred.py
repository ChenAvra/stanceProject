# Copyright 2017 Benjamin Riedel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Import relevant packages and modules
from .util import *
import random
import tensorflow as tf
import tf_slim as slim
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix

tf.compat.v1.disable_eager_execution()


def Pred(df_train, df_test, l, num_of_l):

    # Prompt for mode
    # mode = input('mode (load / train)? ')
    # mode = 'train'
    num_of_labels = num_of_l

    labels = l


    #file_predictions = 'predictions_test.

    #create labels dictionary
    label_ref = {}
    label_ref_rev = {}

    counter = 0
    for t in labels:
        label_ref[t] = counter
        label_ref_rev[counter] = t
        counter += 1

    # Initialise hyperparameters
    r = random.Random()
    lim_unigram = 5000
    target_size = num_of_labels
    hidden_size = 100
    train_keep_prob = 0.6
    l2_alpha = 0.00001
    learn_rate = 0.01
    clip_ratio = 5
    batch_size_train = 500
    epochs = 90


    # Load data sets
    raw_train = FNCData(df_train)
    raw_test = FNCData(df_test)
    test_stances = df_test[Stance]
    test_stances_array = test_stances.to_numpy()
    n_train = len(raw_train.instances)


    # Process data sets
    train_set, train_stances, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = \
        pipeline_train(raw_train, raw_test, label_ref, lim_unigram=lim_unigram)
    feature_size = len(train_set[0])
    test_set = pipeline_test(raw_test, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer)


    # Define model

    # Create placeholders
    features_pl = tf.compat.v1.placeholder(tf.float32, [None, feature_size], 'features')
    stances_pl = tf.compat.v1.placeholder(tf.int64, [None], 'stances')
    keep_prob_pl = tf.compat.v1.placeholder(tf.float32)

    # Infer batch size
    batch_size = tf.shape(input=features_pl)[0]

    # Define multi-layer perceptron
    hidden_layer = tf.nn.dropout(tf.nn.relu(slim.fully_connected(features_pl, hidden_size)), rate=1 - (keep_prob_pl))
    logits_flat = tf.nn.dropout(slim.fully_connected(hidden_layer, target_size), rate=1 - (keep_prob_pl))
    logits = tf.reshape(logits_flat, [batch_size, target_size])

    # hidden_layer = tf.nn.dropout(tf.nn.relu(tf.contrib.layers.linear(features_pl, hidden_size)), rate=1 - (keep_prob_pl))
    # logits_flat = tf.nn.dropout(tf.contrib.layers.linear(hidden_layer, target_size), rate=1 - (keep_prob_pl))
    # logits = tf.reshape(logits_flat, [batch_size, target_size])

    # Define L2 loss
    tf_vars = tf.compat.v1.trainable_variables()
    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf_vars if 'bias' not in v.name]) * l2_alpha

    # Define overall loss
    loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=stances_pl) + l2_loss)

    # loss = tf.reduce_sum(input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(logits, stances_pl) + l2_loss)

    # Define prediction
    softmaxed_logits = tf.nn.softmax(logits)
    predict = tf.argmax(softmaxed_logits, 1)


    # Load model
    # if mode == 'load':
    #     with tf.compat.v1.Session() as sess:
    #         load_model(sess)
    #
    #
    #         # Predict
    #         test_feed_dict = {features_pl: test_set, keep_prob_pl: 1.0}
    #         test_pred = sess.run(predict, feed_dict=test_feed_dict)


    # Train model
    #if mode == 'train':

    # Define optimiser
    opt_func = tf.compat.v1.train.AdamOptimizer(learn_rate)
    grads, _ = tf.clip_by_global_norm(tf.gradients(ys=loss, xs=tf_vars), clip_ratio)
    opt_op = opt_func.apply_gradients(zip(grads, tf_vars))

    # Perform training
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        for epoch in range(epochs):
            total_loss = 0
            indices = list(range(n_train))
            r.shuffle(indices)

            for i in range(n_train // batch_size_train):
                batch_indices = indices[i * batch_size_train: (i + 1) * batch_size_train]
                batch_features = [train_set[i] for i in batch_indices]
                batch_stances = [train_stances[i] for i in batch_indices]

                batch_feed_dict = {features_pl: batch_features, stances_pl: batch_stances, keep_prob_pl: train_keep_prob}
                _, current_loss = sess.run([opt_op, loss], feed_dict=batch_feed_dict)
                total_loss += current_loss


        # Predict
        test_feed_dict = {features_pl: test_set, keep_prob_pl: 1.0}
        test_pred = sess.run(predict, feed_dict=test_feed_dict)

    test_pred_categorial = list()
    for i in range(len(test_pred)):
         val = label_ref_rev[test_pred[i]]
         test_pred_categorial.append(val)

    y_true = list(test_stances_array)
    y_pred = list(test_pred_categorial)


    # cm = confusion_matrix(y_true, y_pred)

    # plot_confusion_matrix(cm, target_names=labels, title="Confusion Matrix", normalize=False)


    # cmd = ConfusionMatrixDisplay(cm, display_labels=['agree', 'disagree', 'discuss', 'unrelated'])
    # cmd.show()
    # print(cmd)
    # ax= plt.subplot()
    # sns.heatmap(cm, annot=True, ax = ax) #annot=True to annotate cells
    #
    # # labels, title and ticks
    # ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')
    # ax.set_title('Confusion Matrix')
    # ax.xaxis.set_ticklabels(['agree', 'disagree', 'discuss', 'unrelated']); ax.yaxis.set_ticklabels(['unrelated', 'discuss', 'disagree', 'agree'], )

    # print(accuracy_score(y_true, y_pred))
    # print(metrics.classification_report(y_true, y_pred))

    return y_true, y_pred
    # Save predictions
    #save_predictions(test_pred, file_predictions)
