# coding=utf-8 
# copyright by speechflow  2023/03/17

import argparse
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
# tf.config.set_visible_devices(gpus[0:1], 'GPU')
import datetime
import time
import os
from shutil import copyfile
import matplotlib.pyplot as plt
from vocab.vocab import Vocab
from configs.config import Config
from models.model import MulSpeechLR as Model
from termcolor import colored
from featurizers.speech_featurizers import NumpySpeechFeaturizer
from dataset import create_dataset
import tensorflow_addons as tfa
from sklearn.metrics import f1_score, recall_score, precision_score
mirrored_strategy = tf.distribute.MirroredStrategy()


def train(config_file):
    config = Config(config_file)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_log_root = "./saved_weights/"
    if not os.path.exists(dir_log_root):
        os.mkdir(dir_log_root)
    dir_current = dir_log_root + current_time
    if not os.path.isdir(dir_log_root):
        os.mkdir(dir_log_root)
    if not os.path.isdir(dir_current):
        os.mkdir(dir_current)
    copyfile(config_file, dir_current + '/config.yml')
    log_file = open(dir_current + '/log.txt', 'w')
    copyfile(config.dataset_config['vocabulary'], dir_current + '/vocab.txt')
    
   
    config.print()
    log_file.write(config.toString())
    # vocab_file.write(config.toString())
    log_file.flush()

    vocab = Vocab(config.dataset_config['vocabulary'])
    batch_size = config.running_config['batch_size']
    global_batch_size = batch_size * mirrored_strategy.num_replicas_in_sync
    speech_featurizer = NumpySpeechFeaturizer(config.speech_config)
    model = Model(**config.model_config, vocab_size=len(vocab.token_list))
    if config.running_config['load_weights'] is not None:
        model.load_weights(config.running_config['load_weights'])
    model.add_featurizers(speech_featurizer)
    model.init_build([None, config.speech_config['num_feature_bins']])
    model.summary()

    train_dataset = create_dataset(batch_size=global_batch_size,
                                load_type=config.dataset_config['load_type'],
                                data_type=config.dataset_config['train'],
                                speech_featurizer=speech_featurizer,
                                config = config,
                                vocab = vocab)
    eval_dataset = create_dataset(batch_size=global_batch_size,
                                load_type=config.dataset_config['load_type'],
                                data_type=config.dataset_config['dev'],
                                speech_featurizer=speech_featurizer,
                                config = config,
                                vocab = vocab)
    test_dataset = create_dataset(batch_size=global_batch_size,
                                load_type=config.dataset_config['load_type'],
                                data_type=config.dataset_config['test'],
                                speech_featurizer=speech_featurizer,
                                config = config,
                                vocab = vocab)
    train_dist_batch = mirrored_strategy.experimental_distribute_dataset(train_dataset)
    dev_dist_batch = mirrored_strategy.experimental_distribute_dataset(eval_dataset)
    test_dist_batch = mirrored_strategy.experimental_distribute_dataset(test_dataset)
    dev_loss = tf.keras.metrics.Mean(name='dev_loss')
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    dev_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
    init_steps = config.optimizer_config['init_steps']
    step = tf.Variable(init_steps)
    
    optimizer = tf.keras.optimizers.Adam(lr=config.optimizer_config['max_lr'])
    ckpt = tf.train.Checkpoint(step=step, optimizer=optimizer, model=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, dir_current + '/ckpt', max_to_keep=5)
    loss_object = tfa.losses.SigmoidFocalCrossEntropy(
            from_logits = True,
            alpha = 0.25,
            gamma  = 0,
        reduction  = tf.keras.losses.Reduction.NONE)
    loss_object_label_smooth = tf.keras.losses.CategoricalCrossentropy(
    from_logits=True, label_smoothing=0.1, reduction=tf.keras.losses.Reduction.NONE)

    def compute_loss(real, pred, smooth=False):
        if smooth:
            loss_ = loss_object_label_smooth(tf.one_hot(real, len(vocab.token_list)), pred)
        else:
            real = tf.one_hot(real, len(vocab.token_list))
            loss_ = loss_object(real, pred)
        return tf.nn.compute_average_loss(loss_, global_batch_size=global_batch_size)

    def accuracy_function(real, pred):
        pred = tf.cast(pred, dtype=tf.int32)
        accuracies = tf.equal(real, pred)

        mask = tf.math.logical_not(tf.math.equal(real, 0))
        accuracies = tf.math.logical_and(mask, accuracies)

        accuracies = tf.cast(accuracies, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)

        return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

    @tf.function
    def train_step(input, input_length, target):
        with tf.GradientTape() as tape:
            predictions = model([input, input_length], training=True)
            loss = compute_loss(target, predictions, smooth=True)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    @tf.function
    def dev_step(input, input_length, target):
        predictions = model([input, input_length], training=False)
        t_loss = compute_loss(target, predictions, smooth=True)
        
        return t_loss, predictions

    @tf.function
    def test_step(input, input_length, target):
        predictions = model([input, input_length], training=False)
        return predictions, target

    @tf.function(experimental_relax_shapes=True)
    def distributed_train_step(x, x_len, y):
        per_replica_losses = mirrored_strategy.run(train_step, args=(x, x_len, y))
        mean_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
        return mean_loss

    @tf.function(experimental_relax_shapes=True)
    def distributed_dev_step(x, x_len, y):
        per_replica_losses, per_replica_preds = mirrored_strategy.run(dev_step, args=(x, x_len, y))
        mean_loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
        return mean_loss, per_replica_preds
        

    @tf.function(experimental_relax_shapes=True)
    def distributed_test_step(x, x_len, y):
        return mirrored_strategy.run(test_step, args=(x, x_len, y))

    plot_train_loss = []
    plot_dev_loss = []
    plot_acc, plot_precision = [], []
    best_acc= 0
    train_iter = iter(train_dist_batch)
    dev_iter = iter(dev_dist_batch)
    test_iter = iter(test_dist_batch)

    for epoch in range(1, config.running_config['num_epochs'] + 1):
        if config.dataset_config['load_type']=='txt':
            train_iter = iter(train_dist_batch)
            dev_iter = iter(dev_dist_batch)
            test_iter = iter(test_dist_batch)
        start = time.time()
        # training loop
        train_loss = 0.0
        dev_loss = 0.0
        for train_batches in range(config.running_config['train_steps']):
            inp, inp_len, target = next(train_iter)
            train_loss += distributed_train_step(inp, inp_len, target)
            template = '\rEpoch {} Step {} Loss {:.4f}'
            print(colored(template.format(
                epoch, train_batches + 1, train_loss / (train_batches + 1),
                ), 'green'), end='', flush=True)
            step.assign_add(1)

        # validation loop
        pred_all = tf.zeros([1], dtype=tf.int32)
        true_all = tf.zeros([1], dtype=tf.int32)
        for dev_batches in range(config.running_config['dev_steps']):
            inp, inp_len, target = next(dev_iter)
            loss, predicted_result = distributed_dev_step(inp, inp_len, target)
            dev_loss += loss
            if mirrored_strategy.num_replicas_in_sync == 1:
                prediction = tf.nn.softmax(predicted_result)
                y_pred = tf.argmax(prediction, axis=-1)
                y_pred = tf.cast(y_pred, dtype=tf.int32)
                pred_all = tf.concat([pred_all, y_pred], axis=0)
                true_all = tf.concat([true_all, target], axis=0)
            else:
                for i in range(mirrored_strategy.num_replicas_in_sync):
                    predicted_result_per_replica = predicted_result.values[i]
                    y_true = target.values[i]
                    y_pred = tf.argmax(predicted_result_per_replica, axis=-1)
                    y_pred = tf.cast(y_pred, dtype=tf.int32)
                    pred_all = tf.concat([pred_all, y_pred], axis=0)
                    true_all = tf.concat([true_all, y_true], axis=0)
        dev_accuracy = accuracy_function(true_all, pred_all)

        pred_all = tf.zeros([1], dtype=tf.int32)
        true_all = tf.zeros([1], dtype=tf.int32)
        for test_batches in range(config.running_config['test_steps']):
            inp, inp_len, target = next(test_iter)
            predicted_result, target_result = distributed_test_step(inp, inp_len, target)
            if mirrored_strategy.num_replicas_in_sync == 1:
                prediction = tf.nn.softmax(predicted_result)
                y_pred =tf.argmax(prediction, axis=-1)
                y_pred = tf.cast(y_pred, dtype=tf.int32)
                pred_all = tf.concat([pred_all, y_pred], axis=0)
                true_all = tf.concat([true_all, target], axis=0)
            else:
                for replica in range(mirrored_strategy.num_replicas_in_sync):
                    predicted_result_per_replica = predicted_result.values[i]
                    y_true = target.values[i]
                    y_pred = tf.argmax(predicted_result_per_replica, axis=-1)
                    y_pred = tf.cast(y_pred, dtype=tf.int32)
                    pred_all = tf.concat([pred_all, y_pred], axis=0)
                    true_all = tf.concat([true_all, y_true], axis=0)
        
        test_acc =  accuracy_function(real=true_all, pred=pred_all)
       
        test_f1 = f1_score(y_true=true_all, y_pred=pred_all, average='macro')
        precision = precision_score(y_true=true_all, y_pred=pred_all, average='macro', zero_division=1)
        recall = recall_score(y_true=true_all, y_pred=pred_all, average='macro')
        if precision > best_acc:
            best_acc = precision
            model.save_weights(dir_current + '/best/' + 'model')
        model.save_weights(dir_current + '/last/' + 'model')
        template = ("\rEpoch {}, Loss: {:.4f}, Val Loss: {:.4f}, "
                    "Val Acc: {:.4f}, test ACC: {:.4f},F1: {:.4f}, precision: {:.4f}, recall: {:.4f}, Time Cost: {:.2f} sec")
        text = template.format(epoch, train_loss / config.running_config['train_steps'],
                               dev_loss/ config.running_config['dev_steps'], dev_accuracy *100,
                               test_acc*100, test_f1*100, precision*100, recall*100, time.time() - start)
        print(colored(text, 'cyan'))
        log_file.write(text)
        log_file.flush()
        plot_train_loss.append(train_loss / config.running_config['train_steps'])
        plot_dev_loss.append(dev_loss / config.running_config['dev_steps'])
        plot_acc.append(test_acc)
        plot_precision.append(precision)
        ckpt_manager.save()

        plt.plot(plot_train_loss, '-r', label='train_loss')
        plt.title('Train Loss')
        plt.xlabel('Epochs')
        plt.savefig(dir_current + '/loss.png')
        #plot dev
        plt.clf()
        plt.plot(plot_dev_loss, '-g', label='dev_loss')
        plt.title('dev Loss')
        plt.xlabel('Epochs')
        plt.savefig(dir_current + '/dev_loss.png')

        # plot acc curve
        plt.clf()
        plt.plot(plot_acc, 'b-', label='acc')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.savefig(dir_current + '/acc.png')
        # plot f1 curve
        plt.clf()
        plt.plot(plot_precision, 'y-', label='f1-score')
        plt.title('F1')
        plt.xlabel('Epochs')
        plt.savefig(dir_current + '/f1-score.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spoken_language_identification Model training")
    parser.add_argument("--config_file", type=str, default='./configs/config.yml', help="Config File Path")
    args = parser.parse_args()
    kwargs = vars(args)
    with mirrored_strategy.scope():
        train(**kwargs)