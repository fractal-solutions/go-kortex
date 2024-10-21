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
	clipThreshold  float64
}

// NewNeuralNetwork initializes a new neural network with given parameters.
func NewNeuralNetwork(inputSize int, hiddenLayers []int, outputSize int, activationType string) *NeuralNetwork {
	nn := &NeuralNetwork{
		inputSize:      inputSize,
		hiddenLayers:   hiddenLayers,
		outputSize:     outputSize,
		activationType: activationType,
		clipThreshold:  1.0, // 0.1 to 1 to 5 for gradient clipping
	}

	// Initialize weights and biases
	nn.weights = append(nn.weights, randomMatrix(inputSize, hiddenLayers[0], inputSize))
	nn.biases = append(nn.biases, make([]float64, hiddenLayers[0]))

	for i := 1; i < len(hiddenLayers); i++ {
		nn.weights = append(nn.weights, randomMatrix(hiddenLayers[i-1], hiddenLayers[i], inputSize))
		nn.biases = append(nn.biases, make([]float64, hiddenLayers[i]))
	}

	nn.weights = append(nn.weights, randomMatrix(hiddenLayers[len(hiddenLayers)-1], outputSize, inputSize))
	nn.biases = append(nn.biases, make([]float64, outputSize))

	return nn
}

// randomMatrix generates a matrix with random values.
func randomMatrix(rows, cols, inputSize int) [][]float64 {
	matrix := make([][]float64, rows)
	for i := range matrix {
		matrix[i] = make([]float64, cols)
		for j := range matrix[i] {
			// Initialize with small random values
			matrix[i][j] = rand.Float64() * math.Sqrt(2/float64(rows))
			//matrix[i][j] = rand.Float64() * math.Sqrt(2.0/float64(inputSize))

		}
	}
	return matrix
}

// tanh activation function
func (nn *NeuralNetwork) tanh(x float64) float64 {
	return math.Tanh(x)
}

// tanhDerivative derivative of tanh
func (nn *NeuralNetwork) tanhDerivative(x float64) float64 {
	return 1 - math.Pow(math.Tanh(x), 2)
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

// leakyReLU activation function
func (nn *NeuralNetwork) leakyReLU(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0.01 * x // 0.01 is the leak factor, you can adjust it
}

// leakyReLUDerivative derivative of leakyReLU
func (nn *NeuralNetwork) leakyReLUDerivative(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0.01 // the same leak factor as in the leakyReLU function
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

// mish activation function
func (nn *NeuralNetwork) mish(x float64) float64 {
	return x * math.Tanh(math.Log(1+math.Exp(x)))
}

// mishDerivative derivative of Mish
func (nn *NeuralNetwork) mishDerivative(x float64) float64 {
	sp := math.Exp(x)
	omega := 4*(x+1) + 4*sp + sp*sp
	delta := 2*sp + sp*sp + 2
	return math.Exp(x) * omega / (delta * delta)
}

// elu activation function
func (nn *NeuralNetwork) elu(x float64) float64 {
	if x > 0 {
		return x
	}
	return 1.0 * (math.Exp(x) - 1)
}

// eluDerivative derivative of ELU
func (nn *NeuralNetwork) eluDerivative(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 1.0 * math.Exp(x)
}

// gelu activation function
func (nn *NeuralNetwork) gelu(x float64) float64 {
	return 0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(x+0.044715*math.Pow(x, 3))))
}

// geluDerivative derivative of GELU
func (nn *NeuralNetwork) geluDerivative(x float64) float64 {
	cdf := 0.5 * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(x+0.044715*math.Pow(x, 3))))
	pdf := math.Exp(-0.5*x*x) / math.Sqrt(2*math.Pi)
	return cdf + x*pdf
}

// activate returns the activation function based on the type.
func (nn *NeuralNetwork) activate(x float64) float64 {
	switch nn.activationType {
	case "tanh":
		return nn.tanh(x)
	case "relu":
		return nn.relu(x)
	case "leaky":
		return nn.leakyReLU(x)
	case "sigmoid":
		return nn.sigmoid(x)
	case "swish":
		return nn.swish(x)
	case "mish":
		return nn.mish(x)
	case "elu":
		return nn.elu(x)
	case "gelu":
		return nn.gelu(x)
	default:
		return x
	}
}

// activateDerivative returns the derivative of the activation function based on the type.
func (nn *NeuralNetwork) activateDerivative(x float64) float64 {
	switch nn.activationType {
	case "tanh":
		return nn.tanhDerivative(x)
	case "relu":
		return nn.reluDerivative(x)
	case "leaky":
		return nn.leakyReLUDerivative(x)
	case "sigmoid":
		return nn.sigmoidDerivative(x)
	case "swish":
		return nn.swishDerivative(x)
	case "mish":
		return nn.mishDerivative(x)
	case "elu":
		return nn.eluDerivative(x)
	case "gelu":
		return nn.geluDerivative(x)
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
		output[i] = nn.activate(neuron) ///ACTIVATING OUTPUT LAYER< EXPERIMENTAL TWEAK
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
func (nn *NeuralNetwork) backward(target []float64, learningRate float64) {
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
	//fmt.Println("Weights: ", nn.weights)
	//fmt.Println("Gradients: ", deltas)

	// Update weights and biases
	for i := 0; i < len(nn.weights); i++ {
		for j := 0; j < len(nn.weights[i]); j++ {
			for k := 0; k < len(nn.weights[i][j]); k++ {
				gradient := learningRate * deltas[i][k] * nn.layerInputs[i][j]
				// Gradient clipping
				if gradient > nn.clipThreshold {
					gradient = nn.clipThreshold
				} else if gradient < -nn.clipThreshold {
					gradient = -nn.clipThreshold
				}
				nn.weights[i][j][k] += gradient
			}
		}
	}

	for i := 0; i < len(nn.biases); i++ {
		for j := 0; j < len(nn.biases[i]); j++ {
			gradient := learningRate * deltas[i][j]
			// Gradient clipping
			if gradient > nn.clipThreshold {
				gradient = nn.clipThreshold
			} else if gradient < -nn.clipThreshold {
				gradient = -nn.clipThreshold
			}
			nn.biases[i][j] += gradient
		}
	}
}

// Train the neural network using backpropagation
func (nn *NeuralNetwork) train(trainingData []map[string][]float64, learningRate float64, epochs int) {
	// Set a default value for clipThreshold if it's zero
	if nn.clipThreshold == 0 {
		nn.clipThreshold = 1.0 // Default value, you can adjust this as needed
	}
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
			nn.backward(target, learningRate)
		}

		if epoch%1000 == 0 {
			fmt.Printf("Epoch %d, Loss: %f\n", epoch, totalLoss/float64(len(trainingData)))
		}
	}
	elapsed := time.Since(start)
	fmt.Printf("Time elapsed: %s\n", elapsed)
	fmt.Println("")
}
