import pickle

with open("./new_init_list", "rb") as f:
    listA = pickle.load(f)

with open("./while_train_list", "rb") as f:
    listB = pickle.load(f)

setA = set(listA)
setB = set(listB)

overlap = setA & setB
universe = setA | setB

result = float(len(overlap)) / len(universe) * 100

print("% of overlap between results from both experiments is ", result)
