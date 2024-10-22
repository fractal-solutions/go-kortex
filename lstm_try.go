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
	dropoutRate := 0.2

	lstm := NewLongShortTermMemory(inputSize, hiddenLayers, outputSize, "sigmoid", learningRate, dropoutRate)

	// Example training data (you can replace this with your actual data)
	trainingData := [][][]float64{
		{{0.1, 0.2, 0.3}},
		{{0.3, 0.4, 0.5}},
		{{0.5, 0.6, 0.7}},
	}
	targets := [][]float64{
		{0.4, 0.5},
		{0.6, 0.7},
		{0.8, 0.9},
	}

	// Train the LSTM
	lstm.train(trainingData, targets, 20000)

	// Testing the LSTM with new data
	testInput := [][]float64{{0.1, 0.2, 0.3}}
	prediction := lstm.forward(testInput)
	fmt.Printf("Test Input: %v, Prediction: %v\n", testInput, prediction)

}
