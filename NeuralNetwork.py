import numpy as np
# import scipy.special
import math

def sigmoid(x):
  return 1/(1+np.exp(-x))

class NeuralNetwork:
    def __init__(self, input_node, hidden_node, output_node, initial_wih, initial_who, learn_rate):
        self.i_nodes = input_node
        self.h_nodes = hidden_node
        self.o_nodes = output_node

        self.wih = initial_wih
        self.who = initial_who

        self.lr = learn_rate
        # self.sigmoid = lambda x: scipy.special.expit(x)

        self.inputs = []
        self.hidden_outputs = []
        self.final_outputs = []

    def train(self, has_won):
        who_update = np.zeros_like(self.who)
        wih_update = np.zeros_like(self.wih)
        print("train" + str(len(self.inputs)))
        for k in range(len(self.inputs)):
            move = np.argmax(self.final_outputs[k])

            target_array = 0.5 * np.ones((self.o_nodes, 1))
            target_array[move] = 0.99 if has_won else 0.01

            print(".........................")
            print(self.final_outputs[k])
            print(target_array)
            print(".........................")

            output_errors = target_array - np.array(self.final_outputs[k])
            hidden_errors = np.dot(self.who.T, output_errors)

            who_update += (output_errors * self.final_outputs[k] *
                           (1.0 - self.final_outputs[k])) * np.transpose(self.hidden_outputs[k])
            wih_update += np.dot((hidden_errors * self.hidden_outputs[k] * (1.0 - self.hidden_outputs[k])),
                                 np.transpose(self.inputs[k]))
        self.who += self.lr * who_update
        self.wih += self.lr * wih_update

    # get AI decision
    def query(self, input_list):
        input_list = np.array(input_list, ndmin=2).T
        self.inputs.append(input_list)

        hidden_inputs = np.dot(self.wih, input_list)
        hidden_outputs = sigmoid(hidden_inputs)
        self.hidden_outputs.append(hidden_outputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = sigmoid(final_inputs)
        self.final_outputs.append(final_outputs)

        # print(final_outputs)
        return final_outputs

    def return_wih(self):
        return self.wih

    def return_who(self):
        return self.who

    def reset(self):
        self.inputs = []
        self.hidden_outputs = []
        self.final_outputs = []

    def save(self, text):
        np.savetxt('ai_catch_who_' + text + '.csv', self.who, delimiter=',')
        np.savetxt('ai_catch_wih_' + text + '.csv', self.wih, delimiter=',')
