import sys
import os
import random
import math

import numpy as np
import operator

# Outputs a random integer, according to a multinomial
# distribution specified by probs.
def rand_multinomial(probs):
    # Make sure probs sum to 1
    assert(abs(sum(probs) - 1.0) < 1e-5)
    rand = random.random()
    for index, prob in enumerate(probs):
        if rand < prob:
            return index
        else:
            rand -= prob
    return 0

# Outputs a random key, according to a (key,prob)
# iterator. For a probability dictionary
# d = {"A": 0.9, "C": 0.1}
# call using rand_multinomial_iter(d.items())
def rand_multinomial_iter(iterator):
    rand = random.random()
    for key, prob in iterator:
        if rand < prob:
            return key
        else:
            rand -= prob
    return 0


class HMM():

    def __init__(self):
        self.num_states = 2
        self.prior      = np.array([0.5, 0.5])
        self.transition = np.array([[0.999, 0.001], [0.01, 0.99]])
        self.emission   = np.array([{"A": 0.291, "T": 0.291, "C": 0.209, "G": 0.209},
                                    {"A": 0.169, "T": 0.169, "C": 0.331, "G": 0.331}])

    # Generates a sequence of states and characters from
    # the HMM model.
    # - length: Length of output sequence
    def sample(self, length):
        sequence = []
        states = []
        rand = random.random()
        cur_state = rand_multinomial(self.prior)
        for i in range(length):
            states.append(cur_state)
            char = rand_multinomial_iter(self.emission[cur_state].items())
            sequence.append(char)
            cur_state = rand_multinomial(self.transition[cur_state])
        return sequence, states

    # Generates a emission sequence given a sequence of states
    def generate_sequence(self, states):
        sequence = []
        for state in states:
            char = rand_multinomial_iter(self.emission[state].items())
            sequence.append(char)
        return sequence

    def generate_forward_matrix(self, size, frw):
        prob=[]
        # Create the Forward Prob Matrix
        for iterator in range(1, size, 1):
            for j in range(0, len(self.prior), 1):
                for i in range(0, len(self.prior), 1):
                    temp_prob = math.log(self.transition[i][j]) + math.log(self.emission[j][sequence[iterator]]) + frw[iterator-1][i]
                    prob.append(temp_prob)
                frw[iterator][j] = self.log_sum(prob)
                prob=[]
        return frw
    
    def generate_backward_matrix(self, size, bck):
        prob = []
        for iterator in range(size-2,-1,-1):
            for j in range(self.num_states):
                for i in range(self.num_states):
                    temo_prob =  math.log(self.transition[j][i]) + math.log(self.emission[i][sequence[iterator+1]]) + bck[iterator+1][i]
                    prob.append(temo_prob)
                bck[iterator][j] = self.log_sum(prob)
                prob=[]
        return bck

    # Outputs the most likely sequence of states given an emission sequence        
    def viterbi(self, sequence):
        # print("my code viterbi")
        size = len(sequence)
        
        # Initialize to  emission probabilities 
        firstA, firstB, rowA, rowB  = self.emission[0][sequence[0]], self.emission[1][sequence[0]], np.empty([self.num_states, size]), np.empty([self.num_states, size])
        # Initialization of the Probabilities
        rowA[0, 0], rowA[1, 0], rowB[0, 0], rowB[1, 0] = math.log(self.prior[0]) + math.log(firstA), math.log(self.prior[1]) + math.log(firstB), 0, 0
        
        for i in range(1, size):
            for j in range(0, self.num_states):
                # Set previous, output (emission) and transition probabilities 
                previousSecond = rowA[1][i-1]
                previousFIrst = rowA[0][i-1]

                tempResult = self.emission[j][sequence[i]]
                transOne, transTwo  = self.transition[0][j], self.transition[1][j]

                temp_smaller, temp_larger = math.log(transOne) + previousFIrst, math.log(transTwo) + previousSecond
                # Set to max
                rowA[j, i] = max(temp_smaller, temp_larger) + math.log(tempResult)
                rowB[j, i] = np.argmax([temp_smaller, temp_larger])

        current_s = np.empty(size, int)        
        # Determines the state with the highest probability
        current_s[size - 1] = rowA[:, size - 1].argmax()
        for j in range(size -1, 0, -1):
            current_s[j-1] = rowB[current_s[j], j]
        final_list = current_s.tolist()
        
        return final_list

    def log_sum(self, factors):
        if abs(min(factors)) > abs(max(factors)):
            a = min(factors)
        else:
            a = max(factors)

        total = 0
        for x in factors:
            total += math.exp(x - a)
        return a + math.log(total)

    # - sequence: String with characters [A,C,T,G]
    # return: posterior distribution. shape should be (len(sequence), 2)
    # Please use log_sum() in posterior computations.
    def posterior(self, sequence):
        
        ###########################################
        # Start your code
        # print("My code here post")
        # End your code
        ###########################################
        size=len(sequence)
        frw, bck, final_result = [], [], []
        
        for i in range (0,size,1):
          frw.append([0,0]), bck.append([1,1]), final_result.append([0,0])
		
		## initial the first probility for forward matrix
        low_part = math.log(self.prior[0])+math.log(self.emission[0][sequence[0]])
        high_part = math.log(self.prior[1])+math.log(self.emission[1][sequence[0]])
        
        # Initialzie the Forward Prob
        frw[0][0] = self.log_sum([low_part])
        frw[0][1] = self.log_sum([high_part])
        # Initialzie the Backwad Prob
        bck[size-1][0] = self.log_sum([math.log(bck[size-1][0])])
        bck[size-1][1] = self.log_sum([math.log(bck[size-1][1])])

        # Create the Forward Prob Matrix
        frw = self.generate_forward_matrix(size, frw)
        # Create the Forward Prob Matrix
        bck = self.generate_backward_matrix(size, bck)
				
		# Using the frw and bck matrix, obtain the final result 
        for iterator in range(size):
            for j in range(self.num_states):
                multiple = frw[iterator][j] * bck[iterator][j]
                result = (1 / self.log_sum(frw[size-1]))
                final_result[iterator][j] = (result * multiple)

        return final_result



    # Output the most likely state for each symbol in an emmision sequence
    # - sequence: posterior probabilities received from posterior()
    # return: list of state indices, e.g. [0,0,0,1,1,0,0,...]
    def posterior_decode(self, sequence):
        nSamples  = len(sequence)
        post = self.posterior(sequence)
        best_path = np.zeros(nSamples)
        for t in range(nSamples):
            best_path[t], _ = max(enumerate(post[t]), key=operator.itemgetter(1))
        return list(best_path.astype(int))


def read_sequences(filename):
    inputs = []
    with open(filename, "r") as f:
        for line in f:
            inputs.append(line.strip())
    return inputs

def write_sequence(filename, sequence):
    with open(filename, "w") as f:
        f.write("".join(sequence))

def write_output(filename, viterbi, posterior):
    vit_file_name = filename[:-4]+'_viterbi_output.txt' 
    with open(vit_file_name, "a") as f:
        for state in range(2):
            f.write(str(viterbi.count(state)))
            f.write("\n")
        f.write(" ".join(map(str, viterbi)))
        f.write("\n")

    pos_file_name = filename[:-4]+'_posteri_output.txt' 
    with open(pos_file_name, "a") as f:
        for state in range(2):
            f.write(str(posterior.count(state)))
            f.write("\n")
        f.write(" ".join(map(str, posterior)))
        f.write("\n")


def truncate_files(filename):
    vit_file_name = file[:-4]+'_viterbi_output.txt'
    pos_file_name = file[:-4]+'_posteri_output.txt' 
    if os.path.isfile(vit_file_name):
        open(vit_file_name, 'w')
    if os.path.isfile(pos_file_name):
        open(pos_file_name, 'w')


if __name__ == '__main__':

    hmm = HMM()

    file = sys.argv[1]
    truncate_files(file)
    
    sequences  = read_sequences(file)
    for sequence in sequences:
        viterbi   = hmm.viterbi(sequence)
        posterior = hmm.posterior_decode(sequence)
        # posterior = []
        write_output(file, viterbi, posterior)


