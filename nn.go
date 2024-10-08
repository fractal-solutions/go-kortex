package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// NeuralNetwork represents a neural network structure.
type NeuralNetwork struct {
	inputSize      int
	hiddenLayers   []int
	outputSize     int
	activationType string
	weights        [][][]float64
	biases         [][]float64
	layerInputs    [][]float64
	layerOutputs   [][]float64
}

// NewNeuralNetwork initializes a new neural network with given parameters.
func NewNeuralNetwork(inputSize int, hiddenLayers []int, outputSize int, activationType string) *NeuralNetwork {
	nn := &NeuralNetwork{
		inputSize:      inputSize,
		hiddenLayers:   hiddenLayers,
		outputSize:     outputSize,
		activationType: activationType,
	}

	// Initialize weights and biases
	nn.weights = append(nn.weights, randomMatrix(inputSize, hiddenLayers[0]))
	nn.biases = append(nn.biases, make([]float64, hiddenLayers[0]))

	for i := 1; i < len(hiddenLayers); i++ {
		nn.weights = append(nn.weights, randomMatrix(hiddenLayers[i-1], hiddenLayers[i]))
		nn.biases = append(nn.biases, make([]float64, hiddenLayers[i]))
	}

	nn.weights = append(nn.weights, randomMatrix(hiddenLayers[len(hiddenLayers)-1], outputSize))
	nn.biases = append(nn.biases, make([]float64, outputSize))

	return nn
}

// randomMatrix generates a matrix with random values.
func randomMatrix(rows, cols int) [][]float64 {
	matrix := make([][]float64, rows)
	for i := range matrix {
		matrix[i] = make([]float64, cols)
		for j := range matrix[i] {
			// Initialize with small random values
			matrix[i][j] = rand.Float64() * math.Sqrt(2/float64(rows+cols))
		}
	}
	return matrix
}

// relu activation function
func (nn *NeuralNetwork) relu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0
}

// reluDerivative derivative of relu
func (nn *NeuralNetwork) reluDerivative(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

// sigmoid activation function
func (nn *NeuralNetwork) sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

// sigmoidDerivative derivative of sigmoid
func (nn *NeuralNetwork) sigmoidDerivative(x float64) float64 {
	sig := nn.sigmoid(x)
	return sig * (1 - sig)
}

// swish activation function
func (nn *NeuralNetwork) swish(x float64) float64 {
	return x / (1 + math.Exp(-x))
}

// swishDerivative derivative of swish
func (nn *NeuralNetwork) swishDerivative(x float64) float64 {
	sig := nn.sigmoid(x)
	return sig + x*sig*(1-sig)
}

// activate returns the activation function based on the type.
func (nn *NeuralNetwork) activate(x float64) float64 {
	switch nn.activationType {
	case "relu":
		return nn.relu(x)
	case "sigmoid":
		return nn.sigmoid(x)
	case "swish":
		return nn.swish(x)
	default:
		return x
	}
}

// activateDerivative returns the derivative of the activation function based on the type.
func (nn *NeuralNetwork) activateDerivative(x float64) float64 {
	switch nn.activationType {
	case "relu":
		return nn.reluDerivative(x)
	case "sigmoid":
		return nn.sigmoidDerivative(x)
	case "swish":
		return nn.swishDerivative(x)
	default:
		return 1
	}
}

// forward passes the input through the network.
func (nn *NeuralNetwork) forward(input []float64) []float64 {
	nn.layerInputs = [][]float64{}
	nn.layerOutputs = [][]float64{}

	layerInput := input
	for i := 0; i < len(nn.hiddenLayers); i++ {
		layerOutput := make([]float64, nn.hiddenLayers[i])
		for j := 0; j < nn.hiddenLayers[i]; j++ {
			neuron := nn.biases[i][j]
			for k := 0; k < len(layerInput); k++ {
				neuron += layerInput[k] * nn.weights[i][k][j]
			}
			layerOutput[j] = nn.activate(neuron)
		}
		nn.layerInputs = append(nn.layerInputs, layerInput)
		nn.layerOutputs = append(nn.layerOutputs, layerOutput)
		layerInput = layerOutput
	}

	// Output layer
	output := make([]float64, nn.outputSize)
	for i := 0; i < nn.outputSize; i++ {
		neuron := nn.biases[len(nn.biases)-1][i]
		for j := 0; j < len(layerInput); j++ {
			neuron += layerInput[j] * nn.weights[len(nn.weights)-1][j][i]
		}
		output[i] = neuron
	}
	nn.layerInputs = append(nn.layerInputs, layerInput)
	nn.layerOutputs = append(nn.layerOutputs, output)

	return output
}

// Mean Squared Error Loss
func (nn *NeuralNetwork) meanSquaredErrorLoss(target, output []float64) float64 {
	loss := 0.0
	for i := 0; i < len(target); i++ {
		loss += math.Pow(target[i]-output[i], 2)
	}
	return loss / float64(len(target))
}

// Huber Loss
func (nn *NeuralNetwork) huberLoss(target, output []float64) float64 {
	const delta = 1.0
	loss := 0.0
	for i := 0; i < len(target); i++ {
		error := target[i] - output[i]
		if math.Abs(error) <= delta {
			loss += 0.5 * math.Pow(error, 2) // quadratic loss
		} else {
			loss += delta*math.Abs(error) - 0.5*delta // linear loss
		}
	}
	return loss / float64(len(target))
}

// Backpropagation to update weights and biases
func (nn *NeuralNetwork) backward(input, target []float64, learningRate float64) {
	deltas := make([][]float64, len(nn.hiddenLayers)+1)
	output := nn.layerOutputs[len(nn.layerOutputs)-1]

	// Output layer delta
	outputDelta := make([]float64, nn.outputSize)
	for i := 0; i < nn.outputSize; i++ {
		outputDelta[i] = target[i] - output[i]
	}
	deltas[len(deltas)-1] = outputDelta

	// Hidden layers delta
	for i := len(nn.hiddenLayers) - 1; i >= 0; i-- {
		layerDelta := make([]float64, nn.hiddenLayers[i])
		for j := 0; j < nn.hiddenLayers[i]; j++ {
			error := 0.0
			for k := 0; k < len(deltas[i+1]); k++ {
				error += deltas[i+1][k] * nn.weights[i+1][j][k]
			}
			layerDelta[j] = error * nn.activateDerivative(nn.layerOutputs[i][j])
		}
		deltas[i] = layerDelta
	}

	// Update weights and biases
	for i := 0; i < len(nn.weights); i++ {
		for j := 0; j < len(nn.weights[i]); j++ {
			for k := 0; k < len(nn.weights[i][j]); k++ {
				nn.weights[i][j][k] += learningRate * deltas[i][k] * nn.layerInputs[i][j]
			}
		}
	}

	for i := 0; i < len(nn.biases); i++ {
		for j := 0; j < len(nn.biases[i]); j++ {
			nn.biases[i][j] += learningRate * deltas[i][j]
		}
	}
}

// Train the neural network using backpropagation
func (nn *NeuralNetwork) train(trainingData []map[string][]float64, learningRate float64, epochs int) {
	start := time.Now()
	fmt.Println("")
	fmt.Println("TRAINING")
	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0
		for _, data := range trainingData {
			input := data["input"]
			target := data["output"]
			output := nn.forward(input)
			loss := nn.huberLoss(target, output)
			totalLoss += loss
			nn.backward(input, target, learningRate)
		}

		if epoch%1000 == 0 {
			fmt.Printf("Epoch %d, Loss: %f\n", epoch, totalLoss/float64(len(trainingData)))
		}
	}
	elapsed := time.Since(start)
	fmt.Printf("Time elapsed: %s\n", elapsed)
	fmt.Println("")
}
