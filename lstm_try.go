package main

import (
	"fmt"
)

func main() {
	// Initialize LSTM
	inputSize := 3
	hiddenLayers := []int{4, 4} // Example hidden layer sizes
	outputSize := 2
	learningRate := 0.001
	dropoutRate := 0.0

	lstm := NewLongShortTermMemory(inputSize, hiddenLayers, outputSize, "relu", learningRate, dropoutRate)

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

	// Train the LSTM
	lstm.train(trainingData, targets, 30000)

	// Testing the LSTM with new data
	fmt.Println("TESTING LSTM...")
	testInput := [][]float64{{0.1, 0.2, 0.3}}
	prediction := lstm.forward(testInput)
	fmt.Printf("Test Input: %v, Prediction: %v\n", testInput, prediction)

	testInput2 := [][]float64{{0.4, 0.5, 0.6}}
	prediction2 := lstm.forward(testInput2)
	fmt.Printf("Test Input: %v, Prediction: %v\n", testInput2, prediction2)

}
