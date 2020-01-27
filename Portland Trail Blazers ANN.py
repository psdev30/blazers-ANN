import numpy as np

# 4 categories: points per game (PPG), points allowed per game (PAPG), wins, playoffs or not
# Over/Under (O/U) for PPG = 113, O/U for PAPG = 110, O/U for wins (W) is avg # of W for current playoff teams
# U PPG = 0; O PAPG = 0; @/O PGG = 1; @/U PAPG = 1
# data is other 14 teams in Western Conference  

data = [[1, 1, 0, 1], # GSW
        [0, 0, 0, 0], # PHX
        [0, 0, 0, 0], # MIN   
        [1, 1, 1, 1], # OKC
        [1, 1, 0, 1], # LAC
        [0, 0, 0, 0], # LAL
        [1, 1, 0, 1], # HOU
        [1, 0, 0, 0], # SAC
        [0, 0, 1, 0], # DAL
        [0, 1, 1, 1], # DEN
        [1, 0, 0, 0], # NO
        [0, 1, 0, 1], # SAS
        [0, 1, 1, 1], # UTA
        [0, 0, 1, 0]] # MEM
  
team_x =  [1, 1, 0] # POR 

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

def training():
    w1 = np.random.randn()
    w2 = np.random.randn()
    w3 = np.random.randn()
    b = np.random.randn()
    
    iterations = 50000
    learning_rate = 0.3

    for i in range(iterations):
        rand_index = np.random.randint(len(data))
        point = data[rand_index]

        z = point[0] * w1 + point[1] * w2 + point[2] * w3 + b
        pred = sigmoid(z) # network's prediction
        target = point[3]
        cost = np.square((pred - target))

        dcost_dpred = 2 * (pred - target)
        dpred_dz = sigmoid_prime(z)
     
        dz_dw1 = point[0] 
        dz_dw2 = point[1]
        dz_dw3 = point[2]
        dz_db = 1

        dcost_dw1 = dcost_dpred * dpred_dz * dz_dw1
        dcost_dw2 = dcost_dpred * dpred_dz * dz_dw2
        dcost_dw3 = dcost_dpred * dpred_dz * dz_dw3
        dcost_db = dcost_dpred * dpred_dz * dz_db

        w1 = w1 - learning_rate * dcost_dw1
        w2 = w2 - learning_rate * dcost_dw2
        w3 = w3 - learning_rate * dcost_dw3
        b = b - learning_rate * dcost_db
    return w1, w2, w3, b

w1, w2, w3, b = training()
print(w1, w2, w3, b)

# viewing predictions

for i in range(len(data)):
    point = data[i]
    print("Point: ", point)
    z = point[0] * w1 + point[1] * w2 + point[2] * w3 + b
    pred = sigmoid(z)
    print("Prediction: ", pred)

# Portland Trail Blazers

z = w1 * team_x[0] + w2 * team_x[1] + w3 * team_x[2] + b
pred = sigmoid(z)

print(pred)