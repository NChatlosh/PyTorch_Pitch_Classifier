import wavfile
import numpy as np
import glob
import torch
import torch.nn as nn
from torch.autograd import Variable

def FindFiles(path): return glob.glob(path)

alpha = 0.001

numExamples = 96
training_examples = []
train_notes =[]
batch_size = 882
num_batches = 25

input_size = 22050
hidden_1_size = 32
hidden_2_size = 16
output_size = 12

notes_lookup = ["cn", "cs", "dn", "ds", "en", "fn", "fs", "gn", "gs", "an",  "as", "bn"]
indices = np.arange(12)

def FindGuess(X):
    guess = 0
    guess_index = 0
    for i in range(len(X)):
        val = np.abs(X[i])
        if(val > guess):
            guess = val
            guess_index = i
    guess_value = notes_lookup[guess_index]
    return guess_value

for fileName in FindFiles('samples/*.wav'):
        data = wavfile.read(fileName)
        data_norm = data[:][1]/(2**24)
        training_examples.append(np.array(data_norm).astype(np.float64))
        train_notes.append(fileName[8]+fileName[9])

encoded_notes = []

for note in train_notes:
    encoded = np.zeros(12)
    index = notes_lookup.index(note)
    encoded[index] = 1
    encoded_notes.append(encoded)

test_training_examples = []
test_train_notes =[]
test_encoded_notes = []

for fileName in FindFiles('test/*.wav'):
    data = wavfile.read(fileName)
    data_norm = data[:][1]/(2**24)
    test_training_examples.append(np.array(data_norm))
    test_train_notes.append(fileName[5]+fileName[6])

for note in test_train_notes:
    encoded = np.zeros(12)
    index = notes_lookup.index(note)
    encoded[index] = 1
    test_encoded_notes.append(encoded)


model = torch.nn.Sequential(
    torch.nn.Linear(input_size, hidden_1_size),
    torch.nn.Tanh(),
    torch.nn.Linear(hidden_1_size, hidden_2_size),
    torch.nn.Tanh(),
    torch.nn.Linear(hidden_2_size, output_size)
)

loss_fn = torch.nn.CrossEntropyLoss()

for i in range(100):
        for j in range(numExamples):
            in_np = training_examples[j]
            in_np = np.reshape(in_np, (1, input_size))
            true = encoded_notes[j]
            index = true.tolist().index(1)
            targetArr = np.array(indices[index])
            targetArr = np.reshape(targetArr, (1, 1))
            target = torch.from_numpy(targetArr).long()
            targ = Variable(target, requires_grad=False)
            input = torch.from_numpy(in_np).float()
            x = Variable(input, requires_grad=False)
            y_pred = model(x)
            loss = loss_fn(y_pred, targ[0])
            print(j, loss.data[0])
            print("guess: " + FindGuess(y_pred.data[0]) + " actual: " + FindGuess(true) + " ypred data[0]: " + str([index]))
            model.zero_grad()
            loss.backward()
            
            for param in model.parameters():
                param.data -= alpha * param.grad.data

print("////////////////////////////////////////")
print("TESTING TIME")
print("////////////////////////////////////////")

rand_indices = np.arange(12)
np.random.shuffle(rand_indices)

for i in range(12):
    test_index = rand_indices[i]
    inp = test_training_examples[test_index]
    inp = np.reshape(inp, (1, input_size))
    true = test_encoded_notes[test_index]
    input = torch.from_numpy(inp).float()
    index = true.tolist().index(1)
    targetArr = np.array(indices[index])
    targetArr = np.reshape(targetArr, (1, 1))
    target = torch.from_numpy(targetArr).long()
    targ = Variable(target, requires_grad=False)
    x = Variable(input, requires_grad=False)
    y_pred = model(x)
    loss = loss_fn(y_pred, targ[0])
    print(i, loss.data[0])
    print("guess: " + FindGuess(y_pred.data[0]) + " actual: " + FindGuess(true))
            