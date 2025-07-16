# Contributing to PPO_PyBullet_Minitaur

We're thrilled you're interested in contributing to this reinforcement learning project! Your contributions help make this project better for everyone. Please take a moment to review this document to ensure a smooth and effective contribution process.

## How Can I Contribute?

There are many ways to contribute, not just by writing code:

*   **Reporting Bugs**: If you find an issue, please report it.
*   **Suggesting Enhancements**: Have an idea for a new feature or an improvement? Let us know!
*   **Submitting Code**: Fix bugs, implement new features, or improve existing ones.
*   **Improving Documentation**: Clarify existing documentation or add new sections.
*   **Giving Feedback**: Use the project and tell us what you think.

## Getting Started

To get a local copy of the project up and running, follow these steps.

### Prerequisites

Make sure you have the following installed:

*   **Python 3.x**
*   **pip** (Python package installer)
*   **git**

### Cloning the Repository

First, clone the repository to your local machine:

```bash
 
git clone https://github.com/EricChen0104/PPO_PyBullet_Minitaur.git
cd PPO_PyBullet_Minitaur
 
```

# Setting Up Your Environment
It's highly recommended to use a virtual environment to manage dependencies:

1. Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

On macOS/Linux:

```bash
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

3. Create a requirements.txt file in the root of your project with the following content (if it doesn't already exist):

```bash
gym
pybullet
pybullet_envs
numpy
torch
matplotlib
```

4. Install the dependencies:

```bash
pip install -r requirements.txt
```

# Running the Project
The test_Trotting.py script serves as the main entry point for running the PPO agent, either for testing with visualization or for training.

- To run the test_agent function (with human rendering):
Ensure that test_agent() is uncommented and train() is commented out in the **if __name__ == '__main__'**: block of test_Trotting.py. Then run:

```bash
python test_Trotting.py
```

- To run the train function (for training):
Ensure that train(load_existing_model=True) (or False if starting from scratch) is uncommented and test_agent() is commented out in the **if __name__ == '__main__'**: block of test_Trotting.py. Then run:

```bash
python test_Trotting.py
```

# Project Structure
- test_Trotting.py: This script handles the environment setup (MinitaurBulletEnv-v0), manages the observation stacking, orchestrates the training and testing loops, and includes basic plotting for scores.
- PPO.py: This file contains the core implementation of the Proximal Policy Optimization (PPO) algorithm. It defines the PPOMemory for experience replay, ActorNetwork and CriticNetwork for the policy and value functions (using PyTorch), and the Agent class that wraps the PPO logic.

# Making Changes Branching
Always create a new branch for your changes:

```bash
git checkout -b feature/your-feature-name  # For new features or significant improvements
git checkout -b bugfix/your-bug-name      # For bug fixes
git checkout -b docs/update-readme        # For documentation changes
```

# Coding Style
- Adhere to PEP 8 guidelines for Python code style.
- Use clear, descriptive variable and function names.
- Add comments where the code's intent isn't immediately obvious, especially for complex RL logic.
- Consider adding type hints for improved readability and maintainability.

# Testing
While the current test_Trotting.py primarily serves as an execution script, contributions are highly encouraged to include:
- Unit Tests: For individual functions and classes (e.g., in PPO.py). Libraries like unittest or pytest are excellent for this.
- Integration Tests: To ensure different components of the system (e.g., agent-environment interaction) work correctly together.

# Commit Messages
Write clear, concise, and descriptive commit messages. A good commit message explains what changed and why the change was made.

Example:

```bash
feat: Implement observation stacking for Minitaur env

This commit introduces a new observation stacking mechanism in `test_Trotting.py`
to provide the PPO agent with a history of states. This is expected to
improve the agent's understanding of dynamics and potentially enhance
learning performance.
```

# Submitting a Pull Request (PR)
When you're ready to submit your changes:

1. Push your branch:
```bash
git push origin feature/your-feature-name
```
2. Open a Pull Request: Go to the GitHub repository and open a new Pull Request against the main branch.
3. Describe your PR:
- Clearly describe the problem your PR solves or the new feature/improvement it adds.
- Reference any related issues (e.g., "Closes #123" or "Fixes #45").
- Include screenshots or GIFs if your changes involve visual aspects (like improved rendering or training plots).
4. Review Process: Your PR will be reviewed by maintainers. Be prepared to address feedback and make further changes if necessary.

# Reporting Bugs
If you find a bug, please open an issue on GitHub. Include:
- A clear and concise description of the bug.
- Steps to reproduce the behavior.
- Expected behavior.
- Actual behavior.
- Your environment details (Operating System, Python version, gym, pybullet, torch, numpy versions).
- Any error messages or tracebacks.

# Suggesting Enhancements
If you have an idea for an enhancement or a new feature, please open an issue on GitHub. Describe:
- The proposed enhancement.
- Why it would be beneficial to the project.
- Any potential alternatives you considered.

Thank you for your contributions!
