import sys, getopt
import copy
import random
import time
import numpy as np
sys.setrecursionlimit(10000)

class SatInstance:
    def __init__(self):
        pass

    def from_file(self, inputfile):
        self.clauses = list()
        self.VARS = set()
        self.p = 0
        self.cnf = 0
        with open(inputfile, "r") as input_file:
            self.clauses.append(list())
            maxvar = 0
            for line in input_file:
                tokens = line.split()
                if len(tokens) != 0 and tokens[0] not in ("p", "c"):
                    for tok in tokens:
                        lit = int(tok)
                        maxvar = max(maxvar, abs(lit))
                        if lit == 0:
                            self.clauses.append(list())
                        else:
                            self.clauses[-1].append(lit)
                if tokens[0] == "p":
                    self.p = int(tokens[2])
                    self.cnf = int(tokens[3])
            assert len(self.clauses[-1]) == 0
            self.clauses.pop()
            if (maxvar > self.p):
                print("Non-standard CNF encoding!")
                sys.exit(5)
        # Variables are numbered from 1 to p
        for i in range(1, self.p + 1):
            self.VARS.add(i)

    def __str__(self):
        s = ""
        for clause in self.clauses:
            s += str(clause)
            s += "\n"
        return s


def main(argv):
    inputfile = ''
    verbosity = False
    inputflag = False
    try:
        opts, args = getopt.getopt(argv, "hi:v", ["ifile="])
    except getopt.GetoptError:
        print('DPLLsat.py -i <inputCNFfile> [-v] ')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('DPLLsat.py -i <inputCNFfile> [-v]')
            sys.exit()
        ##-v sets the verbosity of informational output
        ## (set to true for output veriable assignments, defaults to false)
        elif opt == '-v':
            verbosity = True
        elif opt in ("-i", "--ifile"):
            inputfile = arg
            inputflag = True
    if inputflag:
        instance = SatInstance()
        instance.from_file(inputfile)
        #start_time = time.time()
        solve_dpll(instance, verbosity)
        #print("--- %s seconds ---" % (time.time() - start_time))

    else:
        print("You must have an input file!")
        print('DPLLsat.py -i <inputCNFfile> [-v]')


# Finds a satisfying assignment to a SAT instance,
# using the DPLL algorithm.
# Input: a SAT instance and verbosity flag
# Output: print "UNSAT" or
#    "SAT"
#    list of true literals (if verbosity == True)
#
#  You will need to define your own
#  DPLLsat(), DPLL(), pure-elim(), propagate-units(), and
#  any other auxiliary functions
def solve_dpll(instance, verbosity):
    
    clauses = instance.clauses
    variables = instance.VARS
    
    solution=[]   ##create array
    # clauses=propagate_units(clauses.copy())
    checker = None
    #Begin the recursion
    checker=DPLL(variables.copy(),clauses.copy(),[]) 
    if checker==False:
        print("UNSAT")
    else:
        print("SAT")
        for iterator in checker:
            if iterator>0:
                solution.append(iterator)
        if(verbosity):
            solution.sort()
            # Print the Solution
            print(solution)        

def DPLL(variables,checkClause,model):
    checker=True
    latestClause=[]

    if checkClause==0:
        return False

    if checkClause==[]:
        return model
    else:
        maximum=0
        for iterator in checkClause:
            for each in iterator:
                if(abs(each)>maximum):
                    maximum=each
        if checker:
            # remove the value
            tempCluases = []
            if checker==False:
                maximum=0-maximum
            for i in checkClause:
                new_clause=[]
                if maximum in i:
                    continue
                if -maximum in i:
                    for j in i:
                        if(j!=-maximum):
                            new_clause.append(j)
                    if new_clause==[]:
                        return 0
                    tempCluases.append(new_clause)
                else:
                    tempCluases.append(i)
            latestClause = tempCluases
            # Recursive Call to DPLL
            checker=DPLL(maximum, latestClause, model+[maximum])
            if checker==False:
                # remove the value
                tempCluases = []
                if checker==False:
                    maximum=0-maximum
                for i in checkClause:
                    new_clause=[]
                    if maximum in i:
                        continue
                    if -maximum in i:
                        for j in i:
                            if(j!=-maximum):
                                new_clause.append(j)
                        if new_clause==[]:
                            return 0
                        tempCluases.append(new_clause)
                    else:
                        tempCluases.append(i)
                latestClause = tempCluases
                # Recursive Call to DPLL
                checker=DPLL(maximum, latestClause, model+[maximum])
        return checker

    ###########################################


if __name__ == "__main__":
    main(sys.argv[1:])