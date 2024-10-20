package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// DQNAgent struct
type DQNAgent struct {
	env          *Gridworld
	stateSize    int
	actionSize   int
	gamma        float64
	epsilon      float64
	epsilonMin   float64
	epsilonDecay float64
	learningRate float64
	memory       []Experience
	scoreCurve   []float64
	temperature  float64
	model        *NeuralNetwork
	targetModel  *NeuralNetwork
}

// Experience struct to store experiences
type Experience struct {
	state     []float64
	action    int
	reward    float64
	nextState []float64
	done      bool
	priority  float64
}

// NewDQNAgent initializes a new DQN agent
func NewDQNAgent(env *Gridworld) *DQNAgent {
	agent := &DQNAgent{
		env:          env,
		stateSize:    env.stateSize,
		actionSize:   env.actionSize,
		gamma:        0.99,
		epsilon:      1.0,
		epsilonMin:   0.1,
		epsilonDecay: 0.999985,
		learningRate: 0.0001,
		temperature:  1.0,
		model:        NewNeuralNetwork(env.stateSize, []int{24, 16}, env.actionSize, "relu"),
		targetModel:  NewNeuralNetwork(env.stateSize, []int{24, 16}, env.actionSize, "relu"),
	}
	return agent
}

// Remember stores an experience with priority
func (agent *DQNAgent) Remember(state []float64, action int, reward float64, nextState []float64, done bool) {
	qValues := agent.model.forward(state)
	nextQValues := agent.targetModel.forward(nextState)
	target := reward
	if !done {
		target += agent.gamma * max(nextQValues)
	}
	tdError := math.Abs(target - qValues[action])

	agent.memory = append(agent.memory, Experience{
		state:     state,
		action:    action,
		reward:    reward,
		nextState: nextState,
		done:      done,
		priority:  tdError,
	})
}

// Act selects an action using epsilon-greedy strategy
func (agent *DQNAgent) Act(state []float64, explore bool) int {
	if explore && rand.Float64() <= agent.epsilon {
		return rand.Intn(agent.actionSize)
	}
	qValues := agent.model.forward(state)
	return argmax(qValues)
}

// Replay samples a batch from memory and performs training
func (agent *DQNAgent) Replay(batchSize int) {
	if len(agent.memory) < batchSize {
		return
	}

	// Sort experiences by priority and take the top batchSize experiences
	batch := agent.memory[:batchSize]
	for _, exp := range batch {
		target := exp.reward
		if !exp.done {
			nextQ := agent.targetModel.forward(exp.nextState)
			target += agent.gamma * max(nextQ)
		}
		targetQ := agent.model.forward(exp.state)
		targetQ[exp.action] = target
		agent.model.backward(targetQ, agent.learningRate)
	}

	if agent.epsilon > agent.epsilonMin {
		agent.epsilon *= agent.epsilonDecay
	}
	if len(agent.memory) > 50000 {
		agent.memory = agent.memory[1:] // Remove oldest experience
	}
}

// UpdateTargetModel updates the target model
func (agent *DQNAgent) UpdateTargetModel() {
	agent.targetModel.weights = cloneWeights(agent.model.weights)
	agent.targetModel.biases = cloneBiases(agent.model.biases)
}

// Train the DQN agent
func (agent *DQNAgent) Train(episodes int, batchSize int, steps int) {
	fmt.Println("Starting training...")
	startTime := time.Now()

	for episode := 0; episode < episodes; episode++ {
		state := agent.env.Reset()
		totalReward := 0.0

		for t := 0; t < steps; t++ {
			action := agent.Act(state, true)
			nextState, reward, done := agent.env.Step(action)
			agent.Remember(state, action, reward, nextState, done)

			state = nextState
			totalReward += reward

			if done {
				fmt.Printf("Episode: %d, Score: %.2f, Epsilon: %.3f\n", episode+1, totalReward, agent.epsilon)
				agent.scoreCurve = append(agent.scoreCurve, totalReward)
				break
			}

			if len(agent.memory) >= batchSize {
				agent.Replay(batchSize)
			}
		}
		agent.UpdateTargetModel()
	}

	fmt.Printf("Training time: %.2f seconds\n", time.Since(startTime).Seconds())
}

// Evaluate - Evaluate the agent's performance over multiple episodes
func (agent *DQNAgent) Evaluate(episodes int, maxSteps int) {
	fmt.Println("Starting evaluation...")
	totalRewards := 0.0
	var rewardHistory []float64

	for episode := 0; episode < episodes; episode++ {
		state := agent.env.Reset()
		totalReward := 0.0
		done := false
		steps := 0

		for !done && steps < maxSteps {
			action := agent.Act(state, false) // No exploration, act greedily
			nextState, reward, episodeDone := agent.env.Step(action)

			state = nextState
			totalReward += reward
			done = episodeDone
			steps++
		}

		// If episode terminated early due to max steps
		if steps >= maxSteps {
			fmt.Printf("Episode %d reached maximum step limit.", episode+1)
		}

		totalRewards += totalReward
		rewardHistory = append(rewardHistory, totalReward)
	}

	averageReward := totalRewards / float64(episodes)
	fmt.Printf("Evaluation over %d episodes: Average score: %.2f", episodes, averageReward)
	fmt.Println("Scores:", rewardHistory)
}

// Utility functions

// max finds the maximum value in a slice
func max(array []float64) float64 {
	maxVal := array[0]
	for _, val := range array {
		if val > maxVal {
			maxVal = val
		}
	}
	return maxVal
}

// argmax returns the index of the maximum value in a slice
func argmax(array []float64) int {
	maxIndex := 0
	for i, val := range array {
		if val > array[maxIndex] {
			maxIndex = i
		}
	}
	return maxIndex
}

// cloneWeights creates a deep copy of the weights
func cloneWeights(weights [][][]float64) [][][]float64 {
	cloned := make([][][]float64, len(weights))
	for i := range weights {
		cloned[i] = make([][]float64, len(weights[i]))
		for j := range weights[i] {
			cloned[i][j] = append([]float64{}, weights[i][j]...)
		}
	}
	return cloned
}

// cloneBiases creates a deep copy of the biases
func cloneBiases(biases [][]float64) [][]float64 {
	cloned := make([][]float64, len(biases))
	for i := range biases {
		cloned[i] = append([]float64{}, biases[i]...)
	}
	return cloned
}
