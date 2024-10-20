package main

import (
	"fmt"
	"math/rand"
)

// type Environment interface {
// 	Reset() []float64
// 	Step(action int) ([]float64, float64, bool)
// 	StateSize() int
// 	ActionSize() int
// }

// SimpleEnv - A simple environment for testing the agent
type SimpleEnv struct {
	stateSize    int
	actionSize   int
	currentState []float64
}

// Reset - Resets the environment to a random initial state
func (env *SimpleEnv) Reset() []float64 {
	env.currentState = []float64{
		rand.Float64(),
		rand.Float64(),
		rand.Float64(),
		rand.Float64(),
	}
	return env.currentState
}

// Step - Simulates taking an action in the environment
func (env *SimpleEnv) Step(action int) ([]float64, float64, bool) {
	nextState := []float64{
		rand.Float64(),
		rand.Float64(),
		rand.Float64(),
		rand.Float64(),
	}
	reward := -1.0
	if rand.Float64() < 0.5 {
		reward = 1.0
	}
	done := rand.Float64() < 0.1 // Randomly end episode
	return nextState, reward, done
}

func (env *SimpleEnv) StateSize() int {
	return env.stateSize
}

func (env *SimpleEnv) ActionSize() int {
	return env.actionSize
}

// Gridworld - A grid environment where agent moves on the grid
type Gridworld struct {
	gridSize   int
	stateSize  int
	actionSize int
	grid       [][]int
	position   []int
}

// Reset - Resets the grid environment
func (grid *Gridworld) Reset() []float64 {
	grid.grid = make([][]int, grid.gridSize)
	for i := range grid.grid {
		grid.grid[i] = make([]int, grid.gridSize)
	}
	grid.grid[3][3] = 1         // Set goal at random position
	grid.position = []int{0, 0} //grid.position = []int{rand.Intn(grid.gridSize), rand.Intn(grid.gridSize)} // Start at random position
	return grid.GetState()
}

// GetState - Returns the current state as a flattened grid
func (grid *Gridworld) GetState() []float64 {
	flatGrid := make([]float64, grid.gridSize*grid.gridSize)
	for i, row := range grid.grid {
		for j, cell := range row {
			flatGrid[i*grid.gridSize+j] = float64(cell)
		}
	}
	// Mark the agent's current position
	index := grid.position[0]*grid.gridSize + grid.position[1]
	flatGrid[index] = -0.5
	return flatGrid
}

// Step - Moves the agent based on the action and returns next state, reward, and done
func (grid *Gridworld) Step(action int) ([]float64, float64, bool) {
	x, y := grid.position[0], grid.position[1]
	newX, newY := x, y

	switch action {
	case 0: // Move up
		if x > 0 {
			newX--
		}
	case 1: // Move down
		if x < grid.gridSize-1 {
			newX++
		}
	case 2: // Move left
		if y > 0 {
			newY--
		}
	case 3: // Move right
		if y < grid.gridSize-1 {
			newY++
		}
	}

	grid.position = []int{newX, newY}
	done := newX == grid.gridSize-1 && newY == grid.gridSize-1 // Check if goal is reached
	reward := -0.1
	if done {
		reward = 10
	}
	return grid.GetState(), reward, done
}

func (grid *Gridworld) StateSize() int {
	return grid.stateSize
}

func (grid *Gridworld) ActionSize() int {
	return grid.actionSize
}

// TestGridworldAgent - Test DQN agent with the Gridworld environment
func TestGridworldAgent() {
	gridEnv := &Gridworld{
		gridSize:   5, // 8x8 grid
		stateSize:  25,
		actionSize: 4, // Up, Down, Left, Right
	}

	agent := NewDQNAgent(gridEnv) // Create the DQN agent

	fmt.Println("Training on Gridworld...")
	agent.Train(2000, 32, 300) // Train for 200 episodes

	fmt.Println("Evaluating performance...")
	agent.Evaluate(100, 300) // Evaluate on 100 episodes
}

func main() {
	TestGridworldAgent()
}
