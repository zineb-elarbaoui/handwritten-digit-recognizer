import numpy as np
from tensorflow.keras.datasets import mnist # type: ignore
from sklearn.preprocessing import OneHotEncoder
import cv2 

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize inputs to [0,1] range
X_train = X_train.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

# Flatten images
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# One-hot encode labels
encoder = OneHotEncoder(sparse_output=False)
y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1))
y_test_onehot = encoder.transform(y_test.reshape(-1, 1))

# Activation function & derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)



#def xavier_normal_weight_initialization(structure):
    weights = []
    for i in range(len(structure)-1):
        # Xavier initialization for weights with normal distribution
        std_dev = 1 / np.sqrt(structure[i])  # Standard deviation = 1 / sqrt(n_inputs)
        weight_matrix = np.random.normal(0, std_dev, (structure[i], structure[i+1]))  # mean=0, std=std_dev
        weights.append(weight_matrix)
    return weights

#def xavier_normal_bias_initialization(structure):
    biases = []
    for i in range(len(structure)-1):
        # Xavier initialization for biases with normal distribution
        std_dev = 1 / np.sqrt(structure[i])  # Standard deviation = 1 / sqrt(n_inputs)
        bias_vector = np.random.normal(0, std_dev, (1, structure[i+1]))  # mean=0, std=std_dev
        biases.append(bias_vector)
    return biases

# Neural Network class with batch training
class NeuralNetwork:
    def __init__(self, structure):
        self.layers = len(structure)
        self.weights = [np.random.randn(structure[i], structure[i+1]) * 0.1 for i in range(len(structure)-1)]
        self.biases = [np.zeros((1, structure[i+1])) for i in range(len(structure)-1)]
        
    def forward(self, X):
        sorties = [X]
        for w, b in zip(self.weights, self.biases):
            X = sigmoid(np.dot(X, w) + b)
            sorties.append(X)
        return sorties

    def backward(self, sorties, y_true, learning_rate):
        deltas = [(sorties[-1] - y_true) * sigmoid_derivative(sorties[-1])]
        
        # Compute deltas for hidden layers
        for i in range(len(self.weights)-1, 0, -1):
            deltas.insert(0, deltas[0] @ self.weights[i].T * sigmoid_derivative(sorties[i]))
        
        # Update weights & biases
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * sorties[i].T @ deltas[i]
            self.biases[i] -= learning_rate * np.sum(deltas[i], axis=0, keepdims=True)

    def train(self, X, y, epochs=5, batch_size=32, learning_rate=0.1):
        for epoch in range(epochs):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X, y = X[indices], y[indices]
            
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                activations = self.forward(X_batch)
                self.backward(activations, y_batch, learning_rate)
            
            # Compute accuracy
            predictions = np.argmax(self.forward(X)[-1], axis=1)
            accuracy = np.mean(predictions == np.argmax(y, axis=1)) * 100
            print(f"Epoch {epoch+1}/{epochs}, Accuracy: {accuracy:.2f}%")
    # After training, save the weights and biases
    def save_model(self, filename):
        model_dict = {
            'weights': self.weights,
            'biases': self.biases
        }
        np.save(filename, model_dict)
        print(f"Model saved to {filename}.npy")

    # Load model weights and biases from a file
    def load_model(self, filename):
        model_dict = np.load(filename, allow_pickle=True).item()
        self.weights = model_dict['weights']
        self.biases = model_dict['biases']
        print(f"Model loaded from {filename}.npy")




# Define network structure and train
structure = [784, 64, 32, 10]
nn = NeuralNetwork(structure)
nn.train(X_train, y_train_onehot, epochs=8, batch_size=32, learning_rate=0.1)

# Save model after training

nn.save_model("my_trained_nn.npy")


# Load the trained model
nn.load_model("my_trained_nn.npy")



# OpenCV Drawing Setup
drawing = False  # True if mouse is pressed
ix, iy = -1, -1
canvas = np.ones((280, 280), dtype=np.uint8) * 255  # White canvas for drawing

# Mouse callback function to capture mouse events
def draw(event, x, y, flags, param):
    global ix, iy, drawing, canvas

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.line(canvas, (ix, iy), (x, y), (0, 0, 0), 15)  # Draw black line (thick)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(canvas, (ix, iy), (x, y), (0, 0, 0), 15)  # Final line

# Create window and bind mouse callback function
cv2.namedWindow("Digit Drawing")
cv2.setMouseCallback("Digit Drawing", draw)

while True:
    # Show the canvas and let the user draw
    cv2.imshow("Digit Drawing", canvas)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):  # Press 'q' to quit
        break
    elif key == ord('c'):  # Press 'c' to clear the canvas
        canvas.fill(255)

    elif key == ord('p'):  # Press 'p' to predict the digit
        # Preprocess the drawn image to fit the model input
        img_resized = cv2.resize(canvas, (28, 28))  # Resize to 28x28
        img_inverted = cv2.bitwise_not(img_resized)  # Invert colors (OpenCV uses white background)
        img_normalized = img_inverted.astype(np.float32) / 255.0  # Normalize
        img_flattened = img_normalized.reshape(1, -1)  # Flatten the image for the model

        # Predict using the trained neural network
        prediction = np.argmax(nn.forward(img_flattened)[-1], axis=1)
        print(f"Predicted Digit: {prediction[0]}")

        # Display predicted digit on the canvas
        cv2.putText(canvas, f"Predicted: {prediction[0]}", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        
cv2.destroyAllWindows()