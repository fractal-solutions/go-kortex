
## Neural Network

The `NeuralNetwork` struct provides a basic feedforward neural network implementation.

### Initialization

To initialize a new neural network, use the `NewNeuralNetwork` function:

```go
model := NewNeuralNetwork(inputSize, hiddenLayers, outputSize, activationType, learningRate)
```

- `inputSize`: Number of input neurons.
- `hiddenLayers`: Slice specifying the number of neurons in each hidden layer.
- `outputSize`: Number of output neurons.
- `activationType`: Activation function type (e.g., "relu", "sigmoid").
- `learningRate`: Learning rate for training.

### Example Usage

 // Start of Selection
Refer to [`nn_try.go`](./nn_try.go) for a complete example:

```go
package main

import (
    "fmt"
)

func main() {
    // Initialize the neural network
    nn := NewNeuralNetwork(10, []int{24, 16}, 4, "relu", 0.01)

    // Example input
    input := []float64{0.5, 0.1, -0.3, 0.8, 0.2, 0.9, -0.5, 0.4, 0.7, -0.2}

    // Perform a forward pass
    output := nn.Forward(input)

    // Print the output
    fmt.Println("Neural Network Output:", output)
}
```

---

## Recurrent Neural Network (RNN)

The `RecurrentNeuralNetwork` struct extends the basic neural network with recurrent connections, enabling it to handle sequential data.

### Initialization

Use `NewRecurrentNeuralNetwork` to create an RNN:

```go
rnn := NewRecurrentNeuralNetwork(inputSize, hiddenLayers, outputSize, activationType, learningRate, dropoutRate)
```

- `dropoutRate`: Rate for dropout regularization to prevent overfitting.

 // Start of Selection
### Example Usage

Refer to [`rnn_try.go`](./rnn_try.go) for a complete example:

```go
package main

import (
    "fmt"
)

func main() {
    // Initialize the RNN
    rnn := NewRecurrentNeuralNetwork(10, []int{24, 16}, 4, "tanh", 0.01, 0.2)

    // Example input: 5 time steps with 10 features each
    inputs := [][]float64{
        {0.1, 0.2, -0.1, 0.3, 0.0, 0.5, -0.2, 0.1, 0.4, 0.2},
        {0.0, 0.1, 0.3, -0.2, 0.4, 0.1, 0.0, 0.3, 0.2, -0.1},
        {0.2, -0.1, 0.0, 0.1, 0.3, 0.2, 0.4, -0.3, 0.1, 0.0},
        {0.1, 0.0, -0.2, 0.2, 0.1, 0.3, 0.0, 0.2, -0.1, 0.4},
        {0.3, 0.1, 0.2, 0.0, 0.2, 0.1, 0.3, 0.0, 0.2, 0.1},
    }

    // Perform a forward pass
    output := rnn.forward(inputs)

    // Print the output
    fmt.Println("RNN Output:", output)
}
```

---

## Long Short-Term Memory (LSTM) Network

The `LongShortTermMemory` struct builds upon the RNN by adding gates to better capture long-term dependencies.

### Initialization

Create an LSTM network using `NewLongShortTermMemory`:

```go
lstm := NewLongShortTermMemory(inputSize, hiddenLayers, outputSize, activationType, learningRate, dropoutRate)
```

 // Start of Selection
 ### Example Usage

 See [`lstm_try.go`](./lstm_try.go) for a practical example:

 ```go
 package main

 import (
     "fmt"
 )

 func main() {
     lstm := NewLongShortTermMemory(10, []int{24, 16}, 4, "relu", 0.01, 0.2)

     // Example input: 5 time steps with 10 features each
     inputs := [][]float64{
         {0.1, 0.2, -0.1, 0.3, 0.0, 0.5, -0.2, 0.1, 0.4, 0.2},
         {0.0, 0.1, 0.3, -0.2, 0.4, 0.1, 0.0, 0.3, 0.2, -0.1},
         {0.2, -0.1, 0.0, 0.1, 0.3, 0.2, 0.4, -0.3, 0.1, 0.0},
         {0.1, 0.0, -0.2, 0.2, 0.1, 0.3, 0.0, 0.2, -0.1, 0.4},
         {0.3, 0.1, 0.2, 0.0, 0.2, 0.1, 0.3, 0.0, 0.2, 0.1},
     }

     // Perform a forward pass
     output := lstm.forward(inputs)

     // Print the output
     fmt.Println("LSTM Output:", output)
 }
 ```

---

## Deep Q-Network (DQN) Agent

The `DQNAgent` struct implements a DQN for reinforcement learning tasks.

### Initialization

Initialize the DQN agent with the environment:

```go
agent := NewDQNAgent(env)
```

- `env`: An instance of the `Environment` interface.

### Example Usage

Refer to [`dqn_try.go`](./dqn_try.go) for an example setup:

```go
package main

import (
    "fmt"
)

func main() {
    env := &SimpleEnv{
        stateSize: 4,
        actionSize: 2,
    }
    agent := NewDQNAgent(env)
    episodes := 100

    for e := 1; e <= episodes; e++ {
        state := env.Reset()
        totalReward := 0.0
        done := false

        for !done {
            action := agent.Act(state, true)
            nextState, reward, done := env.Step(action)
            agent.Remember(state, action, reward, nextState, done)
            agent.Replay(32)
            state = nextState
            totalReward += reward
        }

        fmt.Printf("Episode %d: Total Reward: %.2f, Epsilon: %.4f\n", e, totalReward, agent.epsilon)
    }
}
```

---

## Additional Resources

For more detailed information and advanced usage, please refer to the respective `.go` files in this repository:

- [`nn.go`](./nn.go)
- [`rnn.go`](./rnn.go)
- [`lstm.go`](./lstm.go)
- [`dqn.go`](./dqn.go)

Each file contains comprehensive implementations and additional comments to help you understand and extend the functionality of the neural networks and agents.








