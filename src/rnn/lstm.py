import numpy as np

class LSTM:
    def sigmoid(self, x_array):
        result = np.zeros(x_array.shape)
        for i in range(x_array.shape):
            result[i] = 1 / (1 + np.exp(-x_array[i]))
        return result
    
    def __init__(self, prev_ct, prev_ht, input):
        combination = np.concatenate((prev_ht, input), axis=0) 
        ft = self.sigmoid(combination)
        it = np.tanh(combination)
        self.ct = ft * prev_ct + ft * it
        self.ht = np.tanh(self.ct) * ft
        
    def get(self):
        return self.ct, self.ht 
    

class LSTM_Network():
    def __init__(self, unit_number, input, bidirectional=False):
        self.unit_number = unit_number
        self.input = input
        self.bidirectional = bidirectional
        
        
    def forward(self):
        for i in range(self.unit_number):
            if i == 0:
                self.forward.append(LSTM(np.zeros(self.input.shape), np.zeros(self.input.shape), self.input))
            else:
                self.forward.append(LSTM(self.forward[i-1].ct, self.forward[i-1].ht, self.input))