package main

import (
	"fmt"
	"math/rand"
	"time"
)

type RecurrentNeuralNetwork struct {
	NeuralNetwork
	hiddenStates     [][]float64   // Hidden states for each hidden layer
	recurrentWeights [][][]float64 // Recurrent weights for each hidden layer
	dropoutRate      float64       // Dropout rate
}

// NewRecurrentNeuralNetwork initializes an RNN with recurrent connections.
func NewRecurrentNeuralNetwork(inputSize int, hiddenLayers []int, outputSize int, activationType string, learningRate float64, dropoutRate float64) *RecurrentNeuralNetwork {
	rnn := &RecurrentNeuralNetwork{
		NeuralNetwork: *NewNeuralNetwork(inputSize, hiddenLayers, outputSize, activationType, learningRate),
	}

	// Initialize hidden states (start with zero)
	for _, size := range hiddenLayers {
		rnn.hiddenStates = append(rnn.hiddenStates, make([]float64, size))
	}

	// Initialize recurrent weights
	for i := range hiddenLayers {
		previousSize := hiddenLayers[i]
		rnn.recurrentWeights = append(rnn.recurrentWeights, randomMatrix(previousSize, hiddenLayers[i], hiddenLayers[i]))
	}

	rnn.dropoutRate = dropoutRate
	return rnn
}

// Apply dropout to the layer output.
func (rnn *RecurrentNeuralNetwork) applyDropout(layerOutput []float64, dropoutRate float64) []float64 {
	for i := range layerOutput {
		if rand.Float64() < dropoutRate {
			layerOutput[i] = 0.0 // Set neuron output to 0 with probability of dropoutRate
		}
	}
	return layerOutput
}

// forward passes the input through the recurrent network for multiple time steps.
func (rnn *RecurrentNeuralNetwork) forward(inputs [][]float64) []float64 {
	timeSteps := len(inputs)
	var finalOutput []float64

	// Reset the layer inputs and outputs to store each time step's values
	rnn.layerInputs = [][]float64{}
	rnn.layerOutputs = [][]float64{}

	// Iterate over each time step
	for t := 0; t < timeSteps; t++ {
		layerInput := inputs[t] // Input at the current time step

		// Ensure the input size matches the expected input size
		if len(layerInput) != rnn.inputSize {
			panic(fmt.Sprintf("Input size mismatch at time step %d: expected %d, got %d", t, rnn.inputSize, len(layerInput)))
		}

		// Store the input for this time step
		//rnn.layerInputs = append(rnn.layerInputs, layerInput)

		// Iterate over each hidden layer
		for i := 0; i < len(rnn.hiddenLayers); i++ {
			layerOutput := make([]float64, rnn.hiddenLayers[i])

			for j := 0; j < rnn.hiddenLayers[i]; j++ {
				neuron := rnn.biases[i][j]

				// Add contribution from the current input
				for k := 0; k < len(layerInput); k++ {
					neuron += layerInput[k] * rnn.weights[i][k][j]
				}

				// Add contribution from the previous hidden state
				for k := 0; k < len(rnn.hiddenStates[i]); k++ {
					neuron += rnn.hiddenStates[i][k] * rnn.recurrentWeights[i][k][j]
				}

				// Apply the activation function
				layerOutput[j] = rnn.activate(neuron)
			}

			// Apply dropout
			layerOutput = rnn.applyDropout(layerOutput, rnn.dropoutRate)

			// Update hidden states for the next time step
			rnn.hiddenStates[i] = layerOutput

			// Store the layer output for this time step
			rnn.layerInputs = append(rnn.layerInputs, layerInput)
			rnn.layerOutputs = append(rnn.layerOutputs, layerOutput)

			layerInput = layerOutput
		}

		// Output layer (compute output at the current time step)
		finalOutput = make([]float64, rnn.outputSize)
		for i := 0; i < rnn.outputSize; i++ {
			neuron := rnn.biases[len(rnn.biases)-1][i]
			for j := 0; j < len(layerInput); j++ {
				neuron += layerInput[j] * rnn.weights[len(rnn.weights)-1][j][i]
			}
			finalOutput[i] = rnn.activate(neuron)
		}

		// Store the final output for this time step
		rnn.layerInputs = append(rnn.layerInputs, layerInput)
		rnn.layerOutputs = append(rnn.layerOutputs, finalOutput)
	}

	// Return the final output after the last time step
	return finalOutput
}

func (rnn *RecurrentNeuralNetwork) resetHiddenStates() {
	for i := range rnn.hiddenStates {
		for j := range rnn.hiddenStates[i] {
			rnn.hiddenStates[i][j] = rand.Float64() * 0.01 // Small random values
		}
	}
}

// Backpropagation Through Time (BPTT) for RNN
func (rnn *RecurrentNeuralNetwork) backwardBPTT(targets []float64, learningRate float64) {
	timeSteps := len(targets)
	//fmt.Println(timeSteps, " number of timesteps bptt")
	if timeSteps > 1 {
		for t := timeSteps - 1; t >= 0; t-- {
			// Perform backpropagation for each time step
			rnn.backward(targets, rnn.learningRate)
		}
	} else {
		rnn.backward(targets, rnn.learningRate)
	}
}

// Train the RNN using time-stepped sequences.
func (rnn *RecurrentNeuralNetwork) train(trainingData [][][]float64, targets [][]float64, epochs int) {

	start := time.Now()
	fmt.Println("")
	fmt.Println("TRAINING")

	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0
		for i := 0; i < len(trainingData); i++ {
			rnn.resetHiddenStates()
			inputSequence := trainingData[i]
			//fmt.Println("input seq ", (inputSequence), " length ", (len(inputSequence)))
			target := targets[i]
			//fmt.Println("target seq ", (target), " length ", (len(target)))

			// Forward pass for the entire sequence
			output := rnn.forward(inputSequence)

			// Compute loss (e.g., Mean Squared Error)
			loss := rnn.huberLoss(target, output)
			totalLoss += loss

			// Backpropagation Through Time (BPTT)
			rnn.backwardBPTT(target, rnn.learningRate)
		}

		// Print the loss at regular intervals
		if epoch%10 == 0 {
			fmt.Printf("Epoch %d, Loss: %f\n", epoch, totalLoss/float64(len(trainingData)))
		}
	}

	elapsed := time.Since(start)
	fmt.Printf("Time elapsed: %s\n", elapsed)
	fmt.Println("")
}
