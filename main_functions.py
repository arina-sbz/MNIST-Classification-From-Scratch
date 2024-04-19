import numpy as np
import matplotlib.pyplot as plt

def initialize_parameters(layers):
    """
	Initialize the parameters. Weights are initialized randomly to small values with a normal distribution which has a mean of 0 and a standard deviation of 0.01 to prevent gradients from exploding during training. Offsets are initialized to zero

	parameters:
    layers: the layer structure of the model (list) which consists of the number of neurons in each layer (input layer, hidden layers and output layer)

	returns: 
	parameters: dictionary of parameters which consists the weights and offsets of each layer
	"""
    parameters = {}
    for layer in range(1, len(layers)):
        parameters[f"w{layer}"] = np.random.randn( 
            layers[layer], layers[layer - 1]) * 0.01
        parameters[f"b{layer}"] = np.zeros((layers[layer],1))

    return parameters

def relu(z):
    """
    Implement the relu activation function.
    
    parameters:
    z: array containing the result of linear forward step (linear combination of input and weights plus offset) of the forward propagation

    returns:
    the result of relu function
    """
    return np.maximum(0, z)

def sigmoid(z):
    """
    Implement the sigmoid activation function

    parameters:
    z: array containing the result of linear forward step (linear combination of input and weights plus offset) of the forward propagation
    
    returns:
    the result of sigmoid function
    """
    return 1/(1 + np.exp(-z))

def linear_forward(prev, w, b):
    """
    Implement the linear part of a layer's forward propagation.
    
    parameters:
    prev: the activations of the previous layer
    w: the weights matrix of the current layer
    b: the offset vector of the current layer
    
    returns:
    z: the linear combination of the weights and the previous activations plus the offset (for first layer its the linear combination of the weights and the inputs plus offset)
    """
    z = np.dot(w, prev) + b  
    return z

def activation_forward(z, activation):
    """
    Apply the specified activation function to the input z

    parameters:
    z: the input to apply the activation function to
    activation: the activation type, relu or sigmoid

    returns:
    q: the output after applying the activation function
    """
    if activation == "sigmoid":
        q = sigmoid(z)
    elif activation == "relu":
        q = relu(z)
    return q

def softmax(q):
    """
    Compute the softmax function for the given input

    parameters:
    q: Input logits which is the result of the linear forward step of layer before output layer

    returns:
    p: Softmax probabilities
    """
    q = q - np.max(q, axis=0, keepdims=True) # subtracting the maximum value of the input before exponentiation for numerical stability
    p = np.exp(q) / np.sum(np.exp(q), axis=0, keepdims=True)
    return p

def model_forward(x, parameters, activation, num_layers):
    """
    Perform forward propagation through the model

    parameters:
        x: Input data
        parameters: dictionary containing the model parameters
        activation: the activation type, relu or sigmoid
        num_layers: number of layers in the model (input layer + number of hidden layers)

    returns:
        logits: Logits of the output layer (before applying softmax to get probabilities)
        model_state: List of dictionaries containing the model state at each layer
    """
    model_state = []
    prev_q = x.T
    model_state.append({"x_zero": prev_q}) # add the input layer to the model state
    for layer in range(1, num_layers + 1):
        z = linear_forward(
            prev_q, parameters[f"w{layer}"], parameters[f"b{layer}"]) # linear forward pass
        if layer < num_layers: # if not last layer, apply activation function
            q = activation_forward(z, activation)
            model_state.append(
                {"z": z, "q": q, "w": parameters[f"w{layer}"], "b": parameters[f"b{layer}"]}) # append parameters, z and q to model_state
            prev_q = q # replace prev_q with the q of current layer
        else: # if current layer is last layer
            logits = z 
            probabilities = softmax(logits) # get probabilities
            model_state.append(
                {"z": z, "q": probabilities, "w": parameters[f"w{layer}"], "b": parameters[f"b{layer}"]}) # append parameters, z and q (which for last layer is the probabilities because we use it in backward propagation) to model_state
            
    return logits, model_state

def compute_loss(y_true, logits):
    """
    Compute the cross-entropy loss using softmax probabilities according to the given formula

    parameters:
    y_true: the actual target values (one-hot encoded).
    logits: the output of the model before applying softmax

    returns:
    loss: the computed cross-entropy loss averaged over number of samples
    """
    probabilities = softmax(logits) # get probabilities
    loss = -np.mean(np.sum(y_true.T * np.log(probabilities), axis=0)) # compute loss
    return loss

def linear_backward(dz, prev):
    """
    Compute the gradients of the weights and offsets in a linear layer

    parameters:
    dz: the gradient of the cost with respect to the linear output of the layer
    prev: the activations of the previous layer

    returns:
    dw: the gradient of the cost with respect to the weights of the layer
    db: the gradient of the cost with respect to the offsets of the layer
    """
    n = prev.shape[1] 
    dw = np.dot(dz, prev.T) / n
    db = np.sum(dz, axis=1, keepdims=True) / n
    return dw, db

def sigmoid_backward(z):
    """
    Compute the derivative of the sigmoid function with respect to the input z

    parameters:
    z: the input array 

    returns:
    dz: the derivative of the sigmoid function with respect to z
    """
    z_sig = sigmoid(z)
    dz = z_sig * (1 - z_sig) # the derivative is calculated using the output of the sigmoid function itself
    return dz

def relu_backward(z):
    """
    Compute the derivative of the relu activation function.

    parameters:
    z: the input array

    returns:
    dz: the derivative of the relu activation function with respect to z
    """
    dz = np.zeros_like(z) # create an array of zeros with the same shape as z
    dz[z > 0] = 1 # set the derivative to 1 where z > 0
    return dz

def activation_backward(z, activation):
    """
    Compute the backward pass of the chosen activation function

    parameters:
    z: the input array
    activation: the activation type, relu or sigmoid

    returns:
    dz: the gradient of the activation function with respect to the input value
    """
    if activation == "relu":
        dz = relu_backward(z)
    elif activation == "sigmoid":
        dz = sigmoid_backward(z)
    return dz

def model_backward(y_true, model_state, activation, num_layers):
    """
    Perform backpropagation to compute the gradients of the model parameters

    parameters:
    y_true: the true labels of the input data
    model_state: dictionary containing the parameters, z and q computed during forward propagation
    activation: the activation type, relu or sigmoid
    num_layers: the number of layers

    returns:
    gradients: dictionary containing the gradients of the model parameters with respect to the loss function
    """
    gradients = {}
    
    dz = model_state[-1]["q"] - y_true.T # compute the gradient of the loss with respect to the probabilities of the output layer
    
    for layer in reversed(range(1, num_layers + 1)): # loop over the layers in reverse order to perform backpropagation
        layer_data = model_state[layer]
        prev_q = model_state[layer - 1]['x_zero'] if layer == 1 else model_state[layer - 1]['q'] # get the activations of the previous layer if hidden layer, otherwise get the input layer
        if layer < num_layers:  # only for hidden layers (output layer has no activation function)
            dprev = activation_backward(layer_data['z'], activation) # compute the gradient of the activation function
            dz = np.dot(model_state[layer+1]['w'].T, dz) * dprev # compute the gradient of the loss with respect to the linear output of the layer

        dw, db = linear_backward(dz, prev_q) # compute the gradients of the weights and offsets
        
        gradients[f'dw{layer}'] = dw 
        gradients[f'db{layer}'] = db

    return gradients

def update_parameters(learning_rate, gradients, parameters, num_layers):
    """
    Update the model's weights and offset based on the calculated gradients

    parameters:
    learning_rate: the step size at each iteration while moving toward a minimum of the cost function
    gradients: the gradients of the cost function with respect to the weights and offset
    parameters: the model's current weights and offset

    returns:
    parameters: updated weights and offset
    """
    for layer in range(1, num_layers + 1): # loop over the layers
        parameters[f"w{layer}"] -= learning_rate * gradients[f"dw{layer}"] # update the weights
        parameters[f"b{layer}"] -= learning_rate * gradients[f"db{layer}"] # update the offsets

    return parameters

def predict(x, y_true, parameters, activation, num_layers):
    """
    Predict the class labels for input data and calculate the accuracy

    parameters:
        x: the input
        y_true: the true labels
        parameters: Dictionary containing the learned parameters of the model
        activation: the activation type, relu or sigmoid
        num_layers: number of layers

    returns:
        predictions: predicted labels
        accuracy: accuracy of the predictions, calculated as the percentage of correct predictions
        logits: logits of the final layer of the model

    """
    n = y_true.shape[0]
    logits, model_state = model_forward(x, parameters, activation, num_layers)
    predictions = np.argmax(model_state[-1]["q"], axis=0)
    sum_correct_predictions = np.sum(predictions == np.argmax(y_true, axis=1))
    accuracy = sum_correct_predictions / n 
    return predictions, accuracy, logits

def random_mini_batches(x, y, batch_size):
    """
    Create a list of random mini-batches from the given data

    parameters:
    x: the input data 
    y: the target labels
    batch_size: the size of each mini-batch.

    returns:
    mini_batches: A list of mini-batches, where each mini-batch is a tuple containing x_mini and y_mini
    """
    n = x.shape[0]  # number of examples
    mini_batches = []
    indices = np.arange(n) # create an array of indices from 0 to n
    np.random.shuffle(indices)  # shuffle indices
    for k in range(0, n, batch_size):
        mini_batch_indices = indices[k:k+batch_size] # get the indices of the current mini-batch
        x_mini = x[mini_batch_indices, :]
        y_mini = y[mini_batch_indices, :]
        mini_batches.append((x_mini, y_mini))
    
    return mini_batches

def train_model(x_train, y_train, layers, iterations, learning_rate, batch_size, x_test, y_test, k, activation):
    """
    Train a neural network model using mini-batch gradient descent

    parameters:
        x_train: Training data inputs
        y_train: Training data labels
        layers: List of integers representing the number of units in each layer of the model
        iterations: Number of iterations to train the model
        learning_rate: Learning rate for gradient descent
        batch_size: Size of each mini-batch
        x_test: Test data inputs
        y_test: Test data labels
        k: Number of iterations between each evaluation and progress print

    returns:
        test_costs: List of test costs at each evaluation step
        test_accuracies: List of test accuracies at each evaluation step
        batch_costs: List of training costs at each mini-batch
        batch_accuracies: List of training accuracies at each mini-batch
        model_state: Dictionary containing the parameters, z and q of the layers
    """

    parameters = initialize_parameters(layers)
    num_layers = int(len(parameters) / 2)
    batch_costs = []
    batch_accuracies = []
    train_costs = []
    train_accuracies = []
    test_costs = []
    test_accuracies = []

    for iteration in range(iterations):
        mini_batches = random_mini_batches(x_train, y_train, batch_size)
        
        for mini_batch in mini_batches:
            x_mini, y_mini = mini_batch

            logits, model_state = model_forward(x_mini, parameters, activation, num_layers)

            loss = compute_loss(y_mini, logits)
            batch_costs.append(loss)

            gradients = model_backward(
                y_mini, model_state, activation, num_layers)

            parameters = update_parameters(
                learning_rate, gradients, parameters, num_layers)
            
            _ , batch_accuracy, _ = predict(x_mini, y_mini, parameters, activation, num_layers)
            batch_accuracies.append(batch_accuracy)
              
        # evaluate and print progress every k-th iteration
        if (iteration + 1) % k == 0:
            # get predictions and accuracy on whole training data to show every k-th iteration
            train_predictions, train_accuracy, train_logits = predict(
                x_train, y_train, parameters,activation, num_layers) 
            train_cost = compute_loss(y_train, train_logits)
            train_costs.append(train_cost)
            train_accuracies.append(train_accuracy)

            # get predictions and accuracy on whole test data to show every k-th iteration
            test_predictions, test_accuracy, test_logits = predict(
                    x_test, y_test, parameters, activation, num_layers)
            test_cost= compute_loss(y_test, test_logits)
            test_costs.append(test_cost)
            test_accuracies.append(test_accuracy)

            print(f"Iteration: {iteration + 1}")
            # print(f"Train Cost: {train_cost}")
            # print(f"Train Accuracy: {train_accuracy}")
            print(f"Training - Last Batch Cost: {batch_costs[-1]}")
            print(f"Training - Last Batch Accuracy: {batch_accuracies[-1]}")
            print(f"Test Cost: {test_cost}")
            print(f"Test Accuracy: {test_accuracy}")
            print("------------------------------------")

    return train_costs, train_accuracies, test_costs, test_accuracies, batch_costs, batch_accuracies, model_state

def show_weights(parameters):
    """
    Extracts and visualizes the weights of the model with no hidden layers
    
    parameters:
    parameters: dictionary containing the model parameters
    """
    w = parameters[1]["w"] # the weights of the model
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    
    for i, ax in enumerate(axes.flat):
        weight = w[i, :].reshape(28, 28) # reshape the weights to the shape of the input images (28,28)
        ax.imshow(weight, cmap='gray')
        ax.axis('off')
        ax.set_title(f"{i}")
    
    plt.suptitle("Weights")
    plt.tight_layout()
    plt.savefig("weights_plot.png",dpi=300) 
    plt.show()

def training_curve_plot(title, train_losses, test_losses, train_accuracy, test_accuracy,iterations):
    lg=13
    md=10
    sm=9
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=lg)
    x = range(1, iterations + 1, 10)
    axs[0].plot(x, train_losses, label=f'Final train loss: {train_losses[-1]:.4f}')
    axs[0].plot(x, test_losses, label=f'Final test loss: {test_losses[-1]:.4f}')
    axs[0].set_xlabel('Iteration', fontsize=md)
    axs[0].set_ylabel('Loss', fontsize=md)
    axs[0].legend(fontsize=sm)
    axs[0].tick_params(axis='both', labelsize=sm)
    axs[0].grid(True, which="both", linestyle='--', linewidth=0.5)
    axs[1].plot(x, train_accuracy, label=f'Final train accuracy: {train_accuracy[-1]:.4f}%')
    axs[1].plot(x, test_accuracy, label=f'Final test accuracy: {test_accuracy[-1]:.4f}%')
    axs[1].set_xlabel('Iteration', fontsize=md)
    axs[1].set_ylabel('Accuracy (%)', fontsize=sm)
    axs[1].legend(fontsize=sm)
    axs[1].tick_params(axis='both', labelsize=sm)
    axs[1].grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.savefig(f"{title}.png",dpi=300) 
    plt.show()
    