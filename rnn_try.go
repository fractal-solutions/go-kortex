package main

import (
	"fmt"
)

func main() {
	// Example training data: sequences of inputs
	trainingData := [][][]float64{
		{{0.1, 0.2, 0.3}}, // Sequence 1
		{{0.2, 0.3, 0.4}}, // Sequence 2
		{{0.3, 0.4, 0.5}}, // Sequence 3
	}

	// Corresponding targets for each sequence (e.g., predict the next value in the sequence)
	targets := [][]float64{
		{0.4}, // Target for sequence 1
		{0.5}, // Target for sequence 2
		{0.6}, // Target for sequence 3
	}

	// Create an RNN with 1 input, 2 hidden layers (with 4 and 3 neurons), and 1 output
	rnn := NewRecurrentNeuralNetwork(3, []int{2, 2}, 1, "tanh")

	// Train the RNN
	rnn.train(trainingData, targets, 0.001, 1001)

	fmt.Println("TESTING RNN...")
	input := [][]float64{{0.2, 0.3, 0.4}}
	prediction := rnn.forward(input)

	fmt.Printf("Input: %v, Prediction: %v\n", input, prediction)
}
