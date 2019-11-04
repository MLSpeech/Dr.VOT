import torch.nn as nn
import torch
from data_utils import POS,NEG
# POS=1
# NEG=0


__author__ = 'YosiShrem'


class structured_layer(nn.Module):
    def __init__(self, dim=200,is_cuda=True,no_tagging=False):
        super(structured_layer, self).__init__()
        self.dim = dim
        self.w_pos = nn.Parameter(torch.zeros(1, dim, requires_grad=True))
        self.w_neg = nn.Parameter(torch.zeros(1, dim, requires_grad=True))
        # self.w1 = nn.Parameter(torch.zeros(1, dim, requires_grad=True))
        self.MIN_GAP = 1
        self.MIN_SIZE = 5
        self.MAX_SIZE =200
        self.support_tagging = not no_tagging
        self.device = torch.device("cuda" if torch.cuda.is_available() and is_cuda else "cpu")

        print("Structured runs on : {}".format(self.device))
        self.to(self.device)


    def forward(self, x,scores):
        x = x.view(-1, self.dim)
        scores = scores.view(-1)
        if self.support_tagging and scores[POS]>scores[NEG]: # pos
            x = torch.mm(self.w_pos , x.t())
        else:
            x = torch.mm(self.w_neg, x.t())
        return x

    def predict(self, input):
        """
        go over all possible segmentations and choose the one with the highest score
        :param input: the score for each time frame. w*phi(x,y_i)
        :return: score, onset,offset
        """
        input=input.view(-1).cpu()

        onset = 1
        offset = onset+self.MIN_SIZE
        score = input[onset] + input[offset]
        for i in range(self.MIN_GAP, input.shape[0]):
            max_length = min(i+self.MAX_SIZE , input.shape[0])
            for j in range(i + self.MIN_SIZE, max_length):
                tmp = input[i] + input[j]
                if tmp > score:
                    score = tmp
                    onset = i
                    offset = j

        #return score,vot,onset,ofset
        return score, offset-onset , onset, offset

"""

    #on lua, StructLayer:updateOutput is the Forward
    #on lua, the backward is composed of 2 functions:
        the updateGradInput -
            Computing the gradient of the module with respect to its own input
        and  accGradParameters -
            Computing the gradient of the module with respect to its own parameters
"""
