import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

def ReLU(v):

    o = np.maximum(0, v)
    return o

def sigmoid(v):
    
    o = 1 / (1+np.exp(-v))
    return o

def get_output_MLP(w_ih, w_ho, inputs):
    
    #n_hidden = w_ih.shape[1]
    hidden_inp = np.dot(inputs, w_ih)
    hidden_act = sigmoid(hidden_inp)
    bias = np.ones([1,1])
    hidden_net = np.concatenate([hidden_act, bias], 1)
    
    output_inp = np.dot(hidden_net, w_ho)
    output_act = sigmoid(output_inp)
    
    return [hidden_net, output_act]

def draw_MLP(w_ih, w_ho, threshold = 0.5, min_x = -10, max_x = 10, step = 1):
    
    x1 = np.arange(min_x, max_x, step)
    x2 = np.arange(min_x, max_x, step)
    n = len(x1)
    y = np.zeros([n,n])
    inputs = np.ones([1,3])
    
    for i in range(len(x1)):
        for j in range(len(x2)):
            inputs[0,0] = x1[i]
            inputs[0,1] = x2[j]
            [h,o] = get_output_MLP(w_ih, w_ho, inputs)
            y[i,j] = o > threshold
    
    plt.figure()
    for i in range(len(x1)):
        for j in range(len(x2)):
            if(y[i,j]):
                plt.plot(x1[i], x2[j], 'g+')
            else:
                plt.plot(x1[i], x2[j], 'ro')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

def draw_linear_func(w1, w2, threshold, min_x = -10, max_x = 10, step = 1, fs = 15):
    
    
    x1 = np.arange(min_x, max_x, step)
    x2 = np.arange(min_x, max_x, step)
    n = len(x1)
    y = np.zeros([n,n])
    
    for i in range(len(x1)):
        for j in range(len(x2)):
            y[i,j] = (w1 * x1[i] + w2 * x2[j]) > threshold
    
    plt.figure()
    for i in range(len(x1)):
        for j in range(len(x2)):
            if(y[i,j]):
                plt.plot(x1[i], x2[j], 'g+')
            else:
                plt.plot(x1[i], x2[j], 'ro')
    plt.xlabel('$x_1$', fontsize=fs)
    plt.ylabel('$x_2$', fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.show()
    

def draw_simple_problem():
    draw_linear_func(3, -1, 2)
    
def get_XOR_data(dx, dy, min_x = -10, max_x = 10, n_samples = 1000):
    
    x1 = np.random.uniform(min_x, max_x, (n_samples,1))
    x2 = np.random.uniform(min_x, max_x, (n_samples,1))
    y = np.sign(x1+dx) * np.sign(x2+dy) > 0
    
    return [x1, x2, y]

def get_traditional_XOR_data():
    
    x1 = np.asarray([1, 1, -1, -1])
    x2 = np.asarray([1, -1, 1, -1])
    y = np.asarray([0, 1, 1, 0])
    
    return [x1, x2, y]
    
def get_difficult_data():
    [x1, x2, y] = get_XOR_data(0, 0, n_samples=1000) # 1,3 for dx, dy is even more difficult
    y = y.astype(int)
    return [x1, x2, y]
    
def get_simple_data(w1=-2, w2=-2, threshold=0):
    [x1, x2, y] = get_linear_data(w1, w2, threshold, n_samples=1000)
    y = y.astype(int)
    return [x1, x2, y]
    

def get_medium_data():
    [x1, x2, y] = get_quadratic_data(0.25, 0, -5, n_samples=1000)
    y = y.astype(int)
    return [x1, x2, y]
    

def plot_data_line(x1, x2, y, w1, w2):

    n = len(x1)
    for i in range(n):
        if(y[i]):
            plt.plot(x1[i], x2[i], 'g+')
        else:
            plt.plot(x1[i], x2[i], 'ro')
    plt.plot(x1, -(w1/w2)*x1, 'k--')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


def plot_data(x1, x2, y):

    n = len(x1)
    for i in range(n):
        if(y[i]):
            plt.plot(x1[i], x2[i], 'g+')
        else:
            plt.plot(x1[i], x2[i], 'ro')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()
    
    
def get_quadratic_data(a, b, c, min_x = -10, max_x = 10, n_samples=1000):
    
    x1 = np.random.uniform(min_x, max_x, (n_samples,1))
    x2 = np.random.uniform(min_x, max_x, (n_samples,1))
    f = a * x1**2 + b * x1 + c
    y = x2 >= f
    
    return [x1, x2, y]

def get_linear_data(w1, w2, threshold, min_x = -10, max_x = 10, step = 1, n_samples=1000):
    
    x1 = np.random.uniform(min_x, max_x, (n_samples,1))
    x2 = np.random.uniform(min_x, max_x, (n_samples,1))
    y = (w1 * x1 + w2 * x2) > threshold
    
    return [x1, x2, y]

def learn_MLP(x1, x2, t):
    
    # create the weights for an MLP:
    n_hidden_neurons = 20
    np.random.seed(1)
    bound_ih = 0.5
    bound_ho = 1 / n_hidden_neurons
    w_ih = bound_ih*2*(np.random.random([3,n_hidden_neurons])-0.5)
    w_ho = bound_ho*2*(np.random.random([n_hidden_neurons+1, 1])-0.5)
    
    inputs = np.ones([1,3])
    
    n_iterations = 1000
    n_samples = len(x1)
    batch_size = np.min([50, n_samples])
    learning_rate = 0.01
    sum_errs = np.zeros([n_iterations, 1])
    
    Weights = []
    
    for i in range(n_iterations):
        
        if(i % (n_iterations/10) == 0):
            print(f'Iteration {i}')
        
        sum_err = 0
        
        samples = np.random.choice(n_samples, size=batch_size, replace = False)
        # print(f'First sample = {samples[0]}')
        
        d_w_ho = 0
        d_w_ih = 0
        
        for j in samples:
            # calculate output and hidden neuron activations:
            inputs[0,0:2] = [x1[j], x2[j]]
            [h,o] = get_output_MLP(w_ih, w_ho, inputs)
            
            # calculate the gradients:
            error = o - t[j]
            gradient_ho = np.transpose(error * o * (1-o) * h)
            
            # How to adapt the weight of the bias? Shouldn' it be a different activation function?
            gradient_ih = error * o * (1-o) * w_ho * np.transpose(h) * (1-np.transpose(h))
            # are all values related to the right weights?
            gradient_ih = np.transpose(np.multiply(inputs, gradient_ih))
            gradient_ih = gradient_ih[:,:-1]
            
            # update weights:
            d_w_ho += learning_rate * gradient_ho
            d_w_ih += learning_rate * gradient_ih
            
            # # cross-checking:
            # w_ho -= learning_rate * gradient_ho
            # w_ih -= learning_rate * gradient_ih
            
            # [h_post, o_post] = get_output_MLP(w_ih, w_ho, inputs)
            # error_post = o_post - t[j] 
            
            # if(abs(error_post) >= abs(error)):
            #     print('Error got bigger?')
            # #print(f'Error before: {error}, error after: {error_post}.')
            
            # keep track of the loss:
            sum_err += error*error
        
        #print(d_w_ho)
        
        w_ho -= d_w_ho
        w_ih -= d_w_ih
        
        Weights.append((np.copy(w_ho), np.copy(w_ih)))
        
        sum_errs[i] = sum_err / batch_size
        print(f'Sum error = {sum_errs[i][0]}')
    
    plt.figure()
    plt.plot(sum_errs)
    plt.xlabel('Epochs')
    plt.ylabel('Cumulative error.')
        
    # draw the output:
    draw_MLP(w_ih, w_ho, threshold = 0.5)
    
    # draw how it should be:
    plt.figure()
    plot_data(x1, x2, t)
    
def learn_perceptron(x1, x2, t, make_video=False, n_iterations = 100):
    
    # starting parameters:
    w1 = 0
    w2 = 3
    threshold = 0 # the threshold is 0, so this parameter does not have to be learned
    
    # delta rule:
    n_samples = len(x1)
    learning_rate = 0.20 # 0.01
    
    Hist = np.zeros([n_iterations, 3])
    
    n_changes = 0

    for i in range(n_iterations):
        
        print(f'Iteration {i}')
        sum_err = 0

        for j in range(n_samples):

            # calculate output, gradient, and error:
            y = w1 * x1[j] + w2 * x2[j]
            y = sigmoid(y)
            y = y[0]
            
            gradient = y * (1 - y)
            
            error = y - t[j][0]
            
            # update weights:
            w1 -= learning_rate * gradient * error * x1[j][0]
            w2 -= learning_rate * gradient * error * x2[j][0]
            
            if(error != 0 and make_video and abs(gradient) > 0.05):
                h = plt.figure()
                plot_data_line(x1, x2, t, w1, w2)
                ax = plt.gca()
                ax.set_xlim([-10, 10])
                ax.set_ylim([-10, 10])
                plt.savefig(f'./images/image_{n_changes}.png')
                plt.close(h)
                n_changes += 1

            # keep track of the loss:
            sum_err += error*error

        print(f'Sum error = {sum_err}')
        Hist[i,:] = [w1, w2, sum_err]
    
    
    print(f'w1={w1}, w2  = {w2}, T = {threshold}')
    # plot the data and the separation line:
    plt.figure()
    plot_data_line(x1, x2, t, w1, w2)
    
    # plot trajectory on error surface:
    v = Hist[:,0:2].flatten()
    limit = np.max(np.abs(v)) 
    step = 2*limit / 10
    w1_range = np.arange(-limit, limit, step)
    w2_range = np.arange(-limit, limit, step)
    N = len(w1_range)
    E = np.zeros([N,N])
    X = np.zeros([N,N])
    Y = np.zeros([N,N])
    for i1, w1 in enumerate(w1_range):
        for i2, w2 in enumerate(w2_range):
            sum_error = 0
            for j in range(n_samples):
                # calculate output, gradient, and error:
                y = w1 * x1[j] + w2 * x2[j]
                y = sigmoid(y)
                sum_error += (y-t[j])*(y-t[j])
            E[i1, i2] = sum_error
            X[i1, i2] = w1
            Y[i1, i2] = w2

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, E, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf) # , shrink=0.5, aspect=5
    plt.plot(Hist[:,0], Hist[:,1], Hist[:,2] + 2, 'k-')
    plt.xlabel('w1')
    plt.ylabel('w2')
    ax.set_zlabel('Error', rotation = 0)
    plt.show()
    
    
    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, E, levels = 10, cmap=cm.coolwarm)
    ax.clabel(CS, inline=True, fontsize=10)
    ax.set_title('Learning trajectory')
    plt.plot(Hist[:,0], Hist[:,1], 'k-')
    plt.plot(Hist[0,0], Hist[0,1], 'r*')
    plt.plot(Hist[-1,0], Hist[-1,1], 'go')
    plt.xlabel('w1')
    plt.ylabel('w2')
    plt.show()

if __name__ == '__main__':  

    # try regression:
    
    # get the dataset:
    # [x1, x2, t] = get_difficult_data()
    # [x1, x2, t] = get_traditional_XOR_data()
    # [x1, x2, t] = get_medium_data()
    [x1, x2, t] = get_simple_data(-5, 3, 0)
    
    learn_perceptron(x1, x2, t)