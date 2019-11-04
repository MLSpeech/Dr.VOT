import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

__author__ = 'YosiShrem'


class VOT_Seg(nn.Module):
    def __init__(self, input_size=63, hidden_size=200, num_layers=2, dropout=0.5, is_cuda=True):
        super(VOT_Seg, self).__init__()
        self.input_dim = input_size
        self.hidden_dim = hidden_size
        self.num_layers = num_layers

        self.hidden = None
        self.biLSTM = nn.LSTM(input_size, hidden_size // 2, num_layers=num_layers, bidirectional=True,dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.device = torch.device("cuda" if torch.cuda.is_available() and is_cuda else "cpu")

        print("Model runs on : {}, num_of_layers {}".format(self.device,self.num_layers))
        # self.init_hidden()
        self.to(self.device)

    def init_hidden(self):
        # if is_cuda:
        self.hidden = (
            torch.zeros(2 * self.num_layers, 1, self.hidden_dim // 2).to(self.device),
            torch.zeros(2 * self.num_layers, 1, self.hidden_dim // 2).to(self.device))
        # else:
        #     self.hidden = (
        #         torch.zeros(2, 1, self.hidden_dim // 2),
        #         torch.zeros(2, 1, self.hidden_dim // 2))

    def forward(self, sequence):
        """
        run the lstm
        :param sequence: sound features sequence
        :return: phi for every time frame
        """
        self.init_hidden()
        x = sequence.view(-1, 1, self.input_dim)  # seq_len, batch, input_size
        lstm_out, self.hidden = self.biLSTM(x, self.hidden)
        lstm_out = self.dropout(lstm_out)
        return lstm_out


class VOT_tagger(nn.Module):
    def __init__(self, input_size=200, hidden_layer=10, output_size=2, dropout=0.5, is_cuda=True):
        super(VOT_tagger, self).__init__()
        # tagger
        self.input_size = input_size
        # self.bn1 = nn.BatchNorm1d(input_size)
        self.fc1 = nn.Linear(input_size, hidden_layer)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_layer, output_size)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

        self.device = torch.device("cuda" if torch.cuda.is_available() and is_cuda else "cpu")

        print("Tagger runs on : {}".format(self.device))
        self.to(self.device)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        probs = self.logsoftmax(self.fc2(x))
        # print (probs)
        return probs


class Adversarial_Classifier(nn.Module):
    def __init__(self, input_size=200, hidden_layer=(124,32), output_size=3, dropout=0.5, is_cuda=True):
        super(Adversarial_Classifier, self).__init__()
        self.input_size = input_size
        # self.hidden_layer = hidden_layer
        self.reverse = False
        self.lambd = 0

        self.grad_reverse = GradReverse(self.lambd)
        self.fc1 = nn.Linear(input_size, hidden_layer[0])
        self.fc2 = nn.Linear(hidden_layer[0], hidden_layer[1])
        self.fc3 = nn.Linear(hidden_layer[1], output_size)
        self.dropout = nn.Dropout(dropout)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

        self.device = torch.device("cuda" if torch.cuda.is_available() and is_cuda else "cpu")

        print("Adversarial  runs on : {}, hidden_layer [{}]".format(self.device, hidden_layer))
        if self.reverse:
            print("Adversarial using reverse gradient")
        print("Adversarial lambda [{}]\n".format(self.lambd))
        self.to(self.device)

    def set_lambda(self, new_lambd):
        print("set_new_lambd : {}\n".format(new_lambd))
        self.lambd = new_lambd
        self.grad_reverse = GradReverse(new_lambd)
        self.reverse = True

    def set_no_adv(self):
        print("set_no_adversarial\n")
        self.reverse = False
        self.lambd = 0

    def forward(self, x):
        if self.reverse:
            x = self.grad_reverse(x)
        x = x.view(-1, self.input_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        probs = self.logsoftmax(self.fc3(x))
        return probs


class GradReverse(torch.autograd.Function):

    def __init__(self, lambd):
        super(GradReverse, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return -self.lambd * grad_output
