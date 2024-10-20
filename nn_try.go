package main

import (
	"fmt" // Import the neural network package
)

// Test XOR function
func testXOR() {
	// XOR training data
	trainingData := []map[string][]float64{
		{"input": {0, 0}, "output": {0}},
		{"input": {0, 1}, "output": {1}},
		{"input": {1, 0}, "output": {1}},
		{"input": {1, 1}, "output": {0}},
	}

	// Initialize Neural Network with 2 input, 2 hidden neurons, and 1 output
	network := NewNeuralNetwork(2, []int{2}, 1, "relu")

	// Train the network
	network.train(trainingData, 0.1, 1000)

	// Test the network
	fmt.Println("TESTING XOR...")
	testData := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}

	for _, input := range testData {
		prediction := network.forward(input)
		fmt.Printf("Input: %v, Prediction: %v\n", input, prediction)
	}
}

// Test Polynomial (y = x²)
func testPolynomial() {
	// Polynomial data (y = x^2)
	polynomialData := []map[string][]float64{}
	for x := -5.0; x <= 5.0; x += 0.1 {
		polynomialData = append(polynomialData, map[string][]float64{
			"input":  {x},
			"output": {x * x},
		})
	}

	// Initialize Neural Network with 1 input, 5 hidden neurons, and 1 output
	network := NewNeuralNetwork(1, []int{5, 5}, 1, "swish")

	// Train the network
	network.train(polynomialData, 0.001, 10000)

	// Test the network
	fmt.Println("TESTING Polynomial (y = x^2)...")
	for x := -2.0; x <= 2.0; x += 0.4 {
		prediction := network.forward([]float64{x})
		fmt.Printf("Input: %.2f, Predicted: %.2f\n", x, prediction[0])
	}
}

// Test Fibonacci sequence prediction
func testFibonacci() {
	// Fibonacci data
	fibonacci := []float64{0, 1}
	for i := 2; i < 11; i++ {
		fibonacci = append(fibonacci, fibonacci[i-1]+fibonacci[i-2])
	}

	trainingData := []map[string][]float64{}
	for i := 0; i < len(fibonacci)-1; i++ {
		trainingData = append(trainingData, map[string][]float64{
			"input":  {fibonacci[i]},
			"output": {fibonacci[i+1]},
		})
	}

	// Initialize Neural Network with 1 input, 20 and 10 hidden neurons, and 1 output
	network := NewNeuralNetwork(1, []int{20, 10, 10}, 1, "swish")

	// Train the network
	network.train(trainingData, 0.0001, 20000)

	// Test the network
	fmt.Println("TESTING Fibonacci sequence prediction...")
	for i := 0; i < len(fibonacci)-1; i++ {
		prediction := network.forward([]float64{fibonacci[i]})
		fmt.Printf("Input: %.0f, Predicted: %.0f\n", fibonacci[i], prediction[0])
	}
}

// func main() {
// 	// Run all tests
// 	//testXOR()
// 	//testPolynomial()
// 	testFibonacci()
// }
