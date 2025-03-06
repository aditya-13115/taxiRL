# 🚖 Q-Learning for OpenAI Gym Taxi-v3

## 📌 Overview
This project implements **Q-learning**, a reinforcement learning algorithm, to train an AI agent for **Taxi-v3**, an OpenAI Gym environment. The goal is to **efficiently pick up and drop off passengers** while optimizing rewards.

## 📂 Project Structure
```
q_learning_taxi/
│── main.py              # Main script for training and testing the agent
│── requirements.txt     # Dependencies required to run the project
│── q_learning_data/     # Directory to store the trained Q-table
│   └── q_table.npy      # Saved Q-table for persistent learning
```

## ⚙️ Requirements & Installation
Make sure you have Python installed. Then, install the necessary dependencies:
```sh
pip install -r requirements.txt
```
### Dependencies:
- `numpy`
- `gymnasium`
- `pygame`

## 🏗 How It Works
### **1. Q-Learning Algorithm**
- The algorithm maintains a **Q-table**, mapping states to action values.
- The agent explores actions using an **epsilon-greedy policy**.
- The Q-table is **updated iteratively** based on **rewards and future estimates**.

### **2. Training Process**
- The agent is trained for **2000 episodes**.
- It learns by interacting with the environment and adjusting its strategy over time.
- The learned **Q-values** are stored in `q_table.npy` to persist knowledge.

### **3. Testing the Trained Agent**
- After training, the script runs **10 test episodes** where the agent navigates the taxi.
- Uses OpenAI Gym’s **human render mode** for visualization.

## 🚀 Running the Project
To train the model:
```sh
python main.py
```
This will:
✅ Train the agent  
✅ Save the Q-table to `q_learning_data/q_table.npy`  
✅ Test the trained model  

## 📈 Performance Optimization
- **Hyperparameter tuning** (learning rate, discount factor, exploration decay)
- **Increase training episodes** for better convergence
- **Extend to Deep Q-Networks (DQN)** for improved performance

## 🎯 Future Enhancements
- **Integrate visualization tools** to track training progress  
- **Implement DQN** (Deep Q-Learning) to handle more complex environments  
- **Optimize hyperparameters dynamically** for faster learning  

