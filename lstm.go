package main

import (
	"fmt"
	"math"
)

type LongShortTermMemory struct {
	RecurrentNeuralNetwork
	inputGate     [][]float64 // Input gate activations
	forgetGate    [][]float64 // Forget gate activations
	outputGate    [][]float64 // Output gate activations
	cellState     [][]float64 // Cell states
	cellStatePrev [][]float64 // Previous cell states
}

// NewLongShortTermMemory initializes an LSTM with recurrent connections.
func NewLongShortTermMemory(inputSize int, hiddenLayers []int, outputSize int, activationType string, learningRate float64, dropoutRate float64) *LongShortTermMemory {
	lstm := &LongShortTermMemory{
		RecurrentNeuralNetwork: *NewRecurrentNeuralNetwork(inputSize, hiddenLayers, outputSize, activationType, learningRate, dropoutRate),
	}

	// Initialize gates and cell states
	for _, size := range hiddenLayers {
		lstm.inputGate = append(lstm.inputGate, make([]float64, size))
		lstm.forgetGate = append(lstm.forgetGate, make([]float64, size))
		lstm.outputGate = append(lstm.outputGate, make([]float64, size))
		lstm.cellState = append(lstm.cellState, make([]float64, size))
		lstm.cellStatePrev = append(lstm.cellStatePrev, make([]float64, size))
	}

	return lstm
}

// LSTM forward pass
func (lstm *LongShortTermMemory) forward(inputs [][]float64) []float64 {
	timeSteps := len(inputs)
	var finalOutput []float64

	// Reset the layer inputs and outputs to store each time step's values
	lstm.layerInputs = [][]float64{}
	lstm.layerOutputs = [][]float64{}

	// Iterate over each time step
	for t := 0; t < timeSteps; t++ {
		layerInput := inputs[t] // Input at the current time step

		// Ensure the input size matches the expected input size
		if len(layerInput) != lstm.inputSize {
			panic(fmt.Sprintf("Input size mismatch at time step %d: expected %d, got %d", t, lstm.inputSize, len(layerInput)))
		}

		// Store the input for this time step
		// lstm.layerInputs = append(lstm.layerInputs, layerInput)

		// Iterate over each hidden layer
		for i := 0; i < len(lstm.hiddenLayers); i++ {
			layerOutput := make([]float64, lstm.hiddenLayers[i])

			// Compute input, forget, and output gates
			for j := 0; j < lstm.hiddenLayers[i]; j++ {
				// Input gate
				inputGateVal := lstm.biases[i][j]
				for k := 0; k < len(layerInput); k++ {
					inputGateVal += layerInput[k] * lstm.weights[i][k][j]
				}
				for k := 0; k < len(lstm.hiddenStates[i]); k++ {
					inputGateVal += lstm.hiddenStates[i][k] * lstm.recurrentWeights[i][k][j]
				}
				lstm.inputGate[i][j] = sigmoid(inputGateVal)

				// Forget gate
				forgetGateVal := lstm.biases[i][j]
				for k := 0; k < len(layerInput); k++ {
					forgetGateVal += layerInput[k] * lstm.weights[i][k][j]
				}
				for k := 0; k < len(lstm.hiddenStates[i]); k++ {
					forgetGateVal += lstm.hiddenStates[i][k] * lstm.recurrentWeights[i][k][j]
				}
				lstm.forgetGate[i][j] = sigmoid(forgetGateVal)

				// Output gate
				outputGateVal := lstm.biases[i][j]
				for k := 0; k < len(layerInput); k++ {
					outputGateVal += layerInput[k] * lstm.weights[i][k][j]
				}
				for k := 0; k < len(lstm.hiddenStates[i]); k++ {
					outputGateVal += lstm.hiddenStates[i][k] * lstm.recurrentWeights[i][k][j]
				}
				lstm.outputGate[i][j] = sigmoid(outputGateVal)
			}

			// Update cell state
			for j := 0; j < lstm.hiddenLayers[i]; j++ {
				cellInput := tanh(lstm.biases[i][j]) // New cell input
				lstm.cellState[i][j] = lstm.forgetGate[i][j]*lstm.cellStatePrev[i][j] + lstm.inputGate[i][j]*cellInput
				layerOutput[j] = lstm.outputGate[i][j] * tanh(lstm.cellState[i][j])
			}

			// Update hidden states for the next time step
			lstm.hiddenStates[i] = layerOutput
			lstm.cellStatePrev[i] = lstm.cellState[i]

			// Apply dropout
			layerOutput = lstm.applyDropout(layerOutput, lstm.dropoutRate)

			// Store the layer output for this time step
			lstm.layerInputs = append(lstm.layerInputs, layerInput)
			lstm.layerOutputs = append(lstm.layerOutputs, layerOutput)

			layerInput = layerOutput
		}

		// Output layer (compute output at the current time step)
		finalOutput = make([]float64, lstm.outputSize)
		for i := 0; i < lstm.outputSize; i++ {
			neuron := lstm.biases[len(lstm.biases)-1][i]
			for j := 0; j < len(layerInput); j++ {
				neuron += layerInput[j] * lstm.weights[len(lstm.weights)-1][j][i]
			}
			finalOutput[i] = lstm.activate(neuron)
		}

		// Store the final output for this time step
		lstm.layerInputs = append(lstm.layerInputs, layerInput)
		lstm.layerOutputs = append(lstm.layerOutputs, finalOutput)
	}

	// Return the final output after the last time step
	return finalOutput
}

// Sigmoid activation function
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// Tanh activation function
func tanh(x float64) float64 {
	return math.Tanh(x)
}
