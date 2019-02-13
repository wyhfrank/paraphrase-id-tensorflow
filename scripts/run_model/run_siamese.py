# import argparse
import configargparse
import sys
import logging
import math
import numpy as np
import os
import pandas as pd
import pickle
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from duplicate_questions.data.data_manager import DataManager
from duplicate_questions.data.embedding_manager import EmbeddingManager
from duplicate_questions.data.instances.sts_instance import STSInstance
from duplicate_questions.data.instances.code_instance import CodeInstance
from duplicate_questions.models.siamese_bilstm.siamese_bilstm import SiameseBiLSTM

from scripts.data.visualize_result import plot_pairs

logger = logging.getLogger(__name__)


def main():
    default_run_id = "01"
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # Parse config arguments
    argparser = configargparse.ArgumentParser(default_config_files=['../../config/default.yml'],
        description=("Run a baseline Siamese BiLSTM model "
                     "for paraphrase identification."))
    argparser.add_argument("mode", type=str,
                           choices=["train", "predict"],
                           help=("One of {train|predict}, to "
                                 "indicate what you want the model to do. "
                                 "If you pick \"predict\", then you must also "
                                 "supply the path to a pretrained model and "
                                 "DataIndexer to load."))
    argparser.add_argument("--config_file", is_config_file_arg=True,
                           help="The path to a config file.")
    argparser.add_argument("--data_file_dir", type=str,
                           default=os.path.join(project_dir,
                                                "data/processed/"),
                           help="Path of the dir to the (train, val, test).csv files.")
    argparser.add_argument("--train_filename", type=str,
                           default=os.path.join("train.csv"),
                           help="Basename of the train file.")
    argparser.add_argument("--val_filename", type=str,
                           default=os.path.join("train.csv"),
                           help="Basename of the validation file.")
    argparser.add_argument("--test_filename", type=str,
                           default=os.path.join("test.csv"),
                           help="Basename of the test file.")
    argparser.add_argument("--batch_size", type=int, default=128,
                           help="Number of instances per batch.")
    argparser.add_argument("--num_epochs", type=int, default=10,
                           help=("Number of epochs to perform in "
                                 "training."))
    argparser.add_argument("--early_stopping_patience", type=int, default=0,
                           help=("number of epochs with no validation "
                                 "accuracy improvement after which training "
                                 "will be stopped"))
    argparser.add_argument("--num_sentence_words", type=int, default=30,
                           help=("The maximum length of a sentence. Longer "
                                 "sentences will be truncated, and shorter "
                                 "ones will be padded."))
    argparser.add_argument("--embedding_file_path_template", type=str,
                           help="Path to a file with pretrained embeddings.",
                           default=os.path.join(project_dir,
                                                "data/external/bcb",
                                                "{name}.{dim}d.txt"))
    argparser.add_argument("--embedding_file_name", type=str,
                           help="Name of the embedding file.")
    argparser.add_argument("--word_embedding_dim", type=int, default=8,
                           help="Dimensionality of the word embedding layer")
    argparser.add_argument("--fine_tune_embeddings", action="store_true",
                           help=("Whether to train the embedding layer "
                                 "(if True), or keep it fixed (False)."))
    argparser.add_argument("--rnn_hidden_size", type=int, default=256,
                           help=("The output dimension of the RNN."))
    argparser.add_argument("--share_encoder_weights", action="store_true",
                           help=("Whether to use the same encoder on both "
                                 "input sentences (thus sharing weights), "
                                 "or a different one for each sentence"))
    argparser.add_argument("--rnn_output_mode", type=str, default="last",
                           choices=["mean_pool", "last"],
                           help=("How to calculate the final sentence "
                                 "representation from the RNN outputs. "
                                 "\"mean_pool\" indicates that the outputs "
                                 "will be averaged (with respect to padding), "
                                 "and \"last\" indicates that the last "
                                 "relevant output will be used as the "
                                 "sentence representation."))
    argparser.add_argument("--output_keep_prob", type=float, default=1.0,
                           help=("The proportion of RNN outputs to keep, "
                                 "where the rest are dropped out."))
    argparser.add_argument("--log_period", type=int, default=10,
                           help=("Number of steps between each summary "
                                 "op evaluation."))
    argparser.add_argument("--val_period", type=int, default=250,
                           help=("Number of steps between each evaluation of "
                                 "validation performance."))
    argparser.add_argument("--log_dir", type=str,
                           default=os.path.join(project_dir,
                                                "logs/"),
                           help=("Directory to save logs to."))
    argparser.add_argument("--save_period", type=int, default=250,
                           help=("Number of steps between each "
                                 "model checkpoint"))
    argparser.add_argument("--model_save_root", type=str,
                           default=os.path.join(project_dir,
                                                "models/"),
                           help=("Directory to save model checkpoints to."))
    argparser.add_argument("--token_file_dir", type=str,
                           help=("Directory to token files."))
    argparser.add_argument("--token_file_ext", type=str,
                           help=("File extentions of token files."))
    argparser.add_argument("--run_id", type=str, default=default_run_id,
                           help=("Identifying run ID for this run. If "
                                 "predicting, you probably want this "
                                 "to be the same as the train run_id"))
    argparser.add_argument("--model_name", type=str,
                           help=("Identifying model name for this run. If"
                                 "predicting, you probably want this "
                                 "to be the same as the train run_id"))
    argparser.add_argument("--reweight_predictions_for_kaggle", action="store_true",
                           help=("Only relevant when predicting. Whether to "
                                 "reweight the prediction probabilities to "
                                 "account for class proportion discrepancy "
                                 "between train and test."))

    config = argparser.parse_args()
    # logger.info(config)

    model_name = config.model_name
    run_id = config.run_id.zfill(2)
    mode = config.mode
    batch_size = config.batch_size

    paths = construct_paths(model_name, run_id, config.data_file_dir, config.train_filename,
                            config.val_filename, config.test_filename, config.model_save_root,
                            config.log_dir, config.embedding_file_path_template, config.embedding_file_name,
                            config.word_embedding_dim)
    model_save_file_path = paths['model_save_file_path']
    model_save_dir = paths['model_save_dir']
    data_manager_pickle_file_path = paths['data_manager_pickle_file_path']
    CodeInstance.set_token_file(config.token_file_dir, config.token_file_ext)

    if mode == "train":
        # Read the train data from a file, and use it to index the validation data

        # TODO: determine from config
        # data_manager = DataManager(STSInstance)
        data_manager = DataManager(CodeInstance)
        num_sentence_words = config.num_sentence_words
        get_train_data_gen, train_data_size = data_manager.get_train_data_from_file(
            [paths['train_file_path']],
            max_lengths={"num_sentence_words": num_sentence_words})
        get_val_data_gen, val_data_size = data_manager.get_validation_data_from_file(
            [paths['val_file_path']], max_lengths={"num_sentence_words": num_sentence_words})
    else:
        # Load the fitted DataManager, and use it to index the test data
        logger.info("Loading pickled DataManager "
                    "from {}".format(data_manager_pickle_file_path))
        data_manager = pickle.load(open(data_manager_pickle_file_path, "rb"))
        test_data_gen, test_data_size = data_manager.get_test_data_from_file(
            [paths['test_file_path']])

    vars(config)["word_vocab_size"] = data_manager.data_indexer.get_vocab_size()

    # Log the run parameters.
    log_path = paths['log_file_path']
    logger.info("Writing logs to {}".format(log_path))
    if not os.path.exists(log_path):
        logger.info("log path {} does not exist, "
                    "creating it".format(log_path))
        os.makedirs(log_path)

    params_path = os.path.join(log_path, mode + "params.json")
    logger.info("Writing params to {}".format(params_path))
    with open(params_path, 'w') as params_file:
        json.dump(vars(config), params_file, indent=4)

    # Get the embeddings.
    embedding_manager = EmbeddingManager(data_manager.data_indexer)
    embedding_matrix = embedding_manager.get_embedding_matrix(
        config.word_embedding_dim,
        paths['embedding_file_path'])
    vars(config)["word_embedding_matrix"] = embedding_matrix

    # Initialize the model.
    model = SiameseBiLSTM(vars(config))
    model.build_graph()

    if mode == "train":
        # Train the model.
        num_epochs = config.num_epochs
        num_train_steps_per_epoch = int(math.ceil(train_data_size / batch_size))
        num_val_steps = int(math.ceil(val_data_size / batch_size))
        log_period = config.log_period
        val_period = config.val_period

        save_period = config.save_period

        logger.info("Checkpoints will be written to {}".format(model_save_dir))
        if not os.path.exists(model_save_dir):
            logger.info("save path {} does not exist, "
                        "creating it".format(model_save_dir))
            os.makedirs(model_save_dir)

        logger.info("Saving fitted DataManager to {}".format(model_save_dir))
        pickle.dump(data_manager, open(data_manager_pickle_file_path, "wb"))

        patience = config.early_stopping_patience
        model.train(get_train_instance_generator=get_train_data_gen,
                    get_val_instance_generator=get_val_data_gen,
                    batch_size=batch_size,
                    num_train_steps_per_epoch=num_train_steps_per_epoch,
                    num_epochs=num_epochs,
                    num_val_steps=num_val_steps,
                    save_path=model_save_file_path,
                    log_path=log_path,
                    log_period=log_period,
                    val_period=val_period,
                    save_period=save_period,
                    patience=patience)
    else:
        # Predict with the model
        num_test_steps = int(math.ceil(test_data_size / batch_size))
        # Numpy array of shape (num_test_examples, 2)
        raw_predictions, encodings = model.predict(get_test_instance_generator=test_data_gen,
                                        model_load_dir=model_save_dir,
                                        batch_size=batch_size,
                                        num_test_steps=num_test_steps)
        # Remove the first column, so we're left with just the probabilities
        # that a question is a duplicate.
        is_duplicate_probabilities = np.delete(raw_predictions, 0, 1)

        # The class balance between kaggle train and test seems different.
        # This edits prediction probability to account for the discrepancy.
        # See: https://www.kaggle.com/c/quora-question-pairs/discussion/31179
        if config.reweight_predictions_for_kaggle:
            positive_weight = 0.165 / 0.37
            negative_weight = (1 - 0.165) / (1 - 0.37)
            is_duplicate_probabilities = ((positive_weight * is_duplicate_probabilities) /
                                          (positive_weight * is_duplicate_probabilities +
                                           negative_weight *
                                           (1 - is_duplicate_probabilities)))

        # Write the predictions to an output submission file
        predictions_file_path = paths['predictions_file_path']
        logger.info("Writing predictions to {}".format(predictions_file_path))
        is_duplicate_df = pd.DataFrame(is_duplicate_probabilities)
        # is_duplicate_df.to_csv(predictions_file_path, index_label="test_id",
        #                        header=["is_duplicate"])
        # is_duplicate_df.to_csv(predictions_file_path, index=False, header=False)

        encodings_df = pd.DataFrame(encodings)
        pair_info_df = pd.read_csv(paths['test_file_path'], header=None)

        # print(pair_info_df.shape, is_duplicate_df.shape, encodings_df.shape)

        # result = pd.DataFrame(np.hstack((pair_info_df, is_duplicate_df, encodings_df)))
        result = pd.DataFrame(np.hstack((pair_info_df, is_duplicate_df)))

        result.to_csv(predictions_file_path, index=False, header=False)
        plot_pairs(predictions_file_path)


def construct_paths(model_name, run_id, data_file_dir, train_filename='train.csv',
                    val_filename='val.csv', test_filename='test.csv', model_save_root='../../models/',
                    log_dir='../../logs/', embedding_file_path_template='',
                    embedding_filename='', embedding_size=8):
    model_save_dir = os.path.join(model_save_root, model_name, run_id + "/")

    data_manager_pickle_file_path = os.path.join(model_save_dir,
                                                 "{}-{}-DataManager.pkl".format(model_name,
                                                                                run_id))

    train_file_path = os.path.join(data_file_dir, train_filename)
    val_file_path = os.path.join(data_file_dir, val_filename)
    test_file_path = os.path.join(data_file_dir, test_filename)

    log_file_path = os.path.join(log_dir, model_name, run_id)

    predictions_file_path = test_file_path + ".output_predictions.csv"
    model_save_file_path = os.path.join(model_save_dir, model_name + "-" + run_id)

    embedding_file_path = embedding_file_path_template.format(name=embedding_filename,
                                                              dim=embedding_size)

    paths = {
        "model_save_dir": model_save_dir,
        "data_manager_pickle_file_path": data_manager_pickle_file_path,
        "train_file_path": train_file_path,
        "val_file_path": val_file_path,
        "test_file_path": test_file_path,
        "predictions_file_path": predictions_file_path,
        "model_save_file_path": model_save_file_path,
        "log_file_path": log_file_path,
        "embedding_file_path": embedding_file_path,
    }
    return paths


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(levelname)s "
                               "- %(name)s - %(message)s",
                        level=logging.INFO)
    main()
