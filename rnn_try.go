package main

import (
	"fmt"
)

func main1() {
	// Example training data: sequences of inputs
	trainingData := [][][]float64{
		{{0.0, 0.1, 0.2}}, // Sequence 1
		{{0.1, 0.2, 0.3}}, // Sequence 2
		{{0.2, 0.3, 0.4}}, // Sequence 3
		{{0.3, 0.4, 0.5}}, // Sequence 4
		{{0.4, 0.5, 0.6}}, // Sequence 5
		{{0.5, 0.6, 0.7}}, // Sequence 6
		{{0.6, 0.7, 0.8}}, // Sequence 7
		{{0.7, 0.8, 0.9}}, // Sequence 8
		{{0.8, 0.9, 1.0}}, // Sequence 9
		{{0.9, 1.0, 1.1}}, // Sequence 10
		{{1.0, 1.1, 1.2}}, // Sequence 11
		{{1.1, 1.2, 1.3}}, // Sequence 12
		{{1.2, 1.3, 1.4}}, // Sequence 13
		{{1.3, 1.4, 1.5}}, // Sequence 14
		{{1.4, 1.5, 1.6}}, // Sequence 15
		{{1.5, 1.6, 1.7}}, // Sequence 16
		{{1.6, 1.7, 1.8}}, // Sequence 17
		{{1.7, 1.8, 1.9}}, // Sequence 18
		{{1.8, 1.9, 2.0}}, // Sequence 19
		{{1.9, 2.0, 2.1}}, // Sequence 20
	}

	// Corresponding targets for each sequence (predict the next value in the sequence)
	targets := [][]float64{
		{0.3, 0.4}, // Target for sequence 1
		{0.4, 0.5}, // Target for sequence 2
		{0.5, 0.6}, // Target for sequence 3
		{0.6, 0.7}, // Target for sequence 4
		{0.7, 0.8}, // Target for sequence 5
		{0.8, 0.9}, // Target for sequence 6
		{0.9, 1.0}, // Target for sequence 7
		{1.0, 1.1}, // Target for sequence 8
		{1.1, 1.2}, // Target for sequence 9
		{1.2, 1.3}, // Target for sequence 10
		{1.3, 1.4}, // Target for sequence 11
		{1.4, 1.5}, // Target for sequence 13
		{1.5, 1.6}, // Target for sequence 13
		{1.6, 1.7}, // Target for sequence 14
		{1.7, 1.8}, // Target for sequence 15
		{1.8, 1.9}, // Target for sequence 16
		{1.9, 2.0}, // Target for sequence 17
		{2.0, 2.1}, // Target for sequence 18
		{2.1, 2.2}, // Target for sequence 19
		{2.2, 2.3}, // Target for sequence 20
	}

	// Create an RNN with 3 inputs, 3 hidden layers (with 12, 8, 5 neurons), and 2 outputs
	rnn := NewRecurrentNeuralNetwork(3, []int{12, 8, 5}, 2, "relu", 0.001, 0.0)

	// Train the RNN
	rnn.train(trainingData, targets, 20000)

	// Testing the RNN with new data
	fmt.Println("TESTING RNN...")
	testInput := [][]float64{{0.3, 0.4, 0.5}}
	prediction := rnn.forward(testInput)
	fmt.Printf("Test Input: %v, Prediction: %v\n", testInput, prediction)

	testInput2 := [][]float64{{0.7, 0.8, 0.9}}
	prediction2 := rnn.forward(testInput2)
	fmt.Printf("Test Input: %v, Prediction: %v\n", testInput2, prediction2)

	testInput3 := [][]float64{{0.9, 1.5, 1.1}}
	prediction3 := rnn.forward(testInput3)
	fmt.Printf("Test Input: %v, Prediction: %v\n", testInput3, prediction3)
}
