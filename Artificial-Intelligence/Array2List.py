from numpy import load
weights_input_to_hidden = load("D:/SP2(资料存储室)/复旦学习资料/大三下/人工智能/Final/piskvork/model/input2hidden.model.npy")
weights_hidden_to_output = load("D:/SP2(资料存储室)/复旦学习资料/大三下/人工智能/Final/piskvork/model/hidden2output.model.npy")
hiddenList = weights_input_to_hidden.tolist()
outputList = weights_hidden_to_output.tolist()
print(hiddenList)
print(outputList)
