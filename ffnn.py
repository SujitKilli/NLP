import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
import string
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import  word_tokenize
from nltk.corpus import stopwords
from argparse import ArgumentParser
from collections import defaultdict

unk = '<UNK>'
nltk.download('punkt')
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class FFNN(nn.Module):
    def __init__(self, input_dim, h):
        super(FFNN, self).__init__()
        self.h = h
        self.W1 = nn.Linear(input_dim, h)
        self.activation = nn.ReLU() # The rectified linear unit; one valid choice of activation function
        self.output_dim = 5
        self.W2 = nn.Linear(h, self.output_dim)

        self.softmax = nn.LogSoftmax() # The softmax function that converts vectors into probability distributions; computes log probabilities for computational benefits
        self.loss = nn.NLLLoss() # The cross-entropy/negative log likelihood loss taught in class

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        # obtain first hidden layer representation
        first_layer_multiplication = self.W1(input_vector)
        first_layer_output = self.activation(first_layer_multiplication)

        # obtain output layer representation
        final_raw_output = self.W2(first_layer_output)

        # obtain probability dist.
        probability_distribution = self.softmax(final_raw_output)

        return probability_distribution


# Returns:
# vocab = A set of strings corresponding to the vocabulary
def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab


# Returns:
# vocab = A set of strings corresponding to the vocabulary including <UNK>
# word2index = A dictionary mapping word/token to its index (a number in 0, ..., V - 1)
# index2word = A dictionary inverting the mapping of word2index
def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index
        index2word[index] = word
    vocab.add(unk)
    return vocab, word2index, index2word


# Returns:
# vectorized_data = A list of pairs (vector representation of input, y)
def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index))
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data



def load_data(train_data, val_data, test_data = None):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    train_dict = defaultdict(int)
    val_dict = defaultdict(int)

    tra = []
    val = []
    for elt in training:
        sentence = elt["text"].translate(str.maketrans('', '', string.punctuation)).lower()
        tokens = word_tokenize(sentence)
        tokens = [word for word in tokens if word not in stop_words]
        tra.append((tokens,int(elt["stars"]-1)))
        train_dict[elt["stars"]]+=1
    for elt in validation:
        sentence = elt["text"].translate(str.maketrans('', '', string.punctuation)).lower()
        tokens = word_tokenize(sentence)
        tokens = [word for word in tokens if word not in stop_words]
        val.append((tokens,int(elt["stars"]-1)))
        val_dict[elt["stars"]]+=1

    print("training data",train_dict)
    print("validation data",val_dict)
    test = None
    if test_data != None:
        test_dict = defaultdict(int)
        with open(test_data) as test_f:
            testing = json.load(test_f)
        test = []
        for elt in testing:
            sentence = elt["text"].translate(str.maketrans('', '', string.punctuation)).lower()
            tokens = word_tokenize(sentence)
            tokens = [word for word in tokens if word not in stop_words]
            test.append((tokens,int(elt["stars"]-1)))
            test_dict[elt["stars"]]+=1
        print("testing data",test_dict)
    return tra, val, test


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required = True, help = "hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required = True, help = "num of epochs to train")
    parser.add_argument("--train_data", required = True, help = "path to training data")
    parser.add_argument("--val_data", required = True, help = "path to validation data")
    parser.add_argument("--test_data", default = None, help = "path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    # fix random seeds
    random.seed(42)
    torch.manual_seed(42)

    # load data
    print("========== Loading data ==========")
    train_data, valid_data, test_data = load_data(args.train_data, args.val_data, args.test_data) # X_data is a list of pairs (document, y); y in {0,1,2,3,4}
    print("Length of Training data", len(train_data))
    print("Length of Validation data", len(valid_data))
    if test_data!=None:
        print("Length of Test data", len(test_data))
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)

    print("========== Vectorizing data ==========")
    train_data = convert_to_vector_representation(train_data, word2index)
    valid_data = convert_to_vector_representation(valid_data, word2index)
    if test_data!=None:
        test_data = convert_to_vector_representation(test_data, word2index)


    model = FFNN(input_dim = len(vocab), h = args.hidden_dim)
    optimizer = optim.SGD(model.parameters(),lr=0.01, momentum=0.9)
    print("========== Training for {} epochs ==========".format(args.epochs))
    training_correct,training_total = 0,0
    val_correct,val_total = 0,0
    train_list,val_list,loss_list = [],[],[]
    for epoch in range(args.epochs):
        loss_total = 0
        loss_count = 0
        model.train()
        optimizer.zero_grad()
        loss = None
        training_correct = 0
        training_total = 0
        start_time = time.time()
        print("Training started for epoch {}".format(epoch + 1))
        random.shuffle(train_data) # Good practice to shuffle order of training data
        minibatch_size = 16
        N = len(train_data)
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                training_correct += int(predicted_label == gold_label)
                training_total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / minibatch_size
            loss_total += loss.data
            loss_count += 1
            loss.backward()
            optimizer.step()
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, training_correct / training_total))
        print("Training time for this epoch: {}".format(time.time() - start_time))
        loss_list.append(loss_total/loss_count)


        loss = None
        val_correct,val_total = 0,0
        start_time = time.time()
        print("Validation started for epoch {}".format(epoch + 1))
        minibatch_size = 16
        N = len(valid_data)
        model.eval()
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = valid_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                val_correct += int(predicted_label == gold_label)
                val_total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / minibatch_size
        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, val_correct / val_total))
        print("Validation time for this epoch: {}".format(time.time() - start_time))
        train_list.append(training_correct / training_total)
        val_list.append(val_correct / val_total)

        if test_data != None:
            loss = None
            test_correct,test_total = 0,0
            start_time = time.time()
            minibatch_size = 16
            N = len(test_data)
            for minibatch_index in tqdm(range(N // minibatch_size)):
                optimizer.zero_grad()
                loss = None
                for example_index in range(minibatch_size):
                    input_vector, gold_label = test_data[minibatch_index * minibatch_size + example_index]
                    predicted_vector = model(input_vector)
                    predicted_label = torch.argmax(predicted_vector)
                    test_correct += int(predicted_label == gold_label)
                    test_total += 1
                    example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                    if loss is None:
                        loss = example_loss
                    else:
                        loss += example_loss
                loss = loss / minibatch_size
            print(test_correct,test_total)
            print("Testing accuracy {}".format(test_correct / test_total))
            print("Testing time {}".format(time.time() - start_time))

    # write out to results/test.out
    # Ensure the 'results' directory exists
    os.makedirs('./results', exist_ok=True)
    with open('./results/test.out', 'w') as output_file:
        output_file.write(f"Final Training Accuracy: {training_correct / training_total}\n")
        output_file.write(f"Final Validation Accuracy: {val_correct / val_total}\n")
        if test_data != None:
            loss = None
            test_correct,test_total = 0,0
            start_time = time.time()
            minibatch_size = 16
            N = len(test_data)
            for minibatch_index in tqdm(range(N // minibatch_size)):
                optimizer.zero_grad()
                loss = None
                for example_index in range(minibatch_size):
                    input_vector, gold_label = test_data[minibatch_index * minibatch_size + example_index]
                    predicted_vector = model(input_vector)
                    predicted_label = torch.argmax(predicted_vector)
                    test_correct += int(predicted_label == gold_label)
                    test_total += 1
                    example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                    if loss is None:
                        loss = example_loss
                    else:
                        loss += example_loss
                loss = loss / minibatch_size
            print(test_correct,test_total)
            print("Testing accuracy {}".format(test_correct / test_total))
            print("Testing time {}".format(time.time() - start_time))
            output_file.write(f"Test Accuracy: {test_correct / test_total}\n")

    x = [i+1 for i in range(args.epochs)]

    plt.plot(x, train_list, label='training')
    plt.plot(x, val_list, label='validation')

    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()

    plt.show()

    plt.plot(x, loss_list, label='training loss')
    plt.plot(x, val_list, label='devset accuracy')

    plt.xlabel('epoch')
    plt.ylabel('Metrics')
    plt.legend()

    plt.show()