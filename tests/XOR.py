from Layer import * 
X = np.reshape([[0,0],[0,1],[1,0],[1,1]],(4,2,1))
Y = np.reshape([[0],[1],[1],[0]],(4,1,1))

Network = [ 
    Dense(2,3),
    Tanh(),
    Dense(3,1),
    Tanh()]

epochs = 10000
learning_rate = 0.1

# Training Phase
for e in range(epochs):
    error = 0
    for x,y in zip(X,Y):
        output = x
        for layer in Network:
            output = layer.forward(output)
        error += mse(y,output)
        grad = mse_prime(y,output)
        for layer in reversed(Network):
            grad = layer.backward(grad,learning_rate)
        error /= len(X)
    if e//1000:
        print("{}/{} , error={}".format(e+1,epochs,error))