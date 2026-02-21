# Hidden Markov Model – Baum–Welch Algorithm Implementation

## Student Information

**Name:** Alan Ali

**Register Number:** TCR24CS106

**Course:** Pattern Recognition

---

## Project Description

This project presents a complete implementation of the **Baum–Welch algorithm** for training a Hidden Markov Model (HMM).

The Baum–Welch algorithm is a special case of the Expectation–Maximization (EM) algorithm used to estimate the parameters of an HMM when only the observation sequence is known.

All mathematical computations are implemented manually in Python without using any external HMM libraries.

An interactive Streamlit-based interface is provided to:

* Input observed sequences
* Configure model parameters
* Train the HMM
* Visualize convergence behavior
* Display learned state transition diagrams

---

## Hidden Markov Model Formulation

A Hidden Markov Model is defined by:

[
\lambda = (A, B, \pi)
]

Where:

* **A** → State Transition Probability Matrix
* **B** → Emission Probability Matrix
* **π** → Initial State Distribution

The objective of Baum–Welch is to estimate these parameters such that the likelihood of the observed sequence ( P(O \mid \lambda) ) is maximized.

---

## Algorithm Implementation Details

The following components are implemented explicitly:

### 1. Forward Algorithm (α)

Computes the probability of partial observation sequences:

[
\alpha_t(i) = P(O_1, O_2, ..., O_t, q_t = i \mid \lambda)
]

Scaling is applied to prevent numerical underflow.

---

### 2. Backward Algorithm (β)

Computes the probability of the remaining observations:

[
\beta_t(i) = P(O_{t+1}, ..., O_T \mid q_t = i, \lambda)
]

---

### 3. State Posterior Probability (γ)

[
\gamma_t(i) = P(q_t = i \mid O, \lambda)
]

Represents how responsible state *i* is for the observation at time *t*.

---

### 4. Transition Posterior Probability (ξ)

[
\xi_t(i,j) = P(q_t = i, q_{t+1} = j \mid O, \lambda)
]

Represents expected transitions from state *i* to *j*.

---

### 5. Parameter Re-estimation

Parameters are updated as follows:

* Initial distribution:
  [
  \pi_i = \gamma_1(i)
  ]

* Transition matrix:
  [
  a_{ij} = \frac{\sum_{t=1}^{T-1} \xi_t(i,j)}{\sum_{t=1}^{T-1} \gamma_t(i)}
  ]

* Emission matrix:
  [
  b_i(k) = \frac{\sum_{t: O_t = k} \gamma_t(i)}{\sum_{t=1}^{T} \gamma_t(i)}
  ]

These updates are repeated until convergence.

---

## Inputs

The program accepts:

* Observed state sequence (comma-separated integers)
* Number of hidden states
* Maximum training iterations
* Convergence tolerance

The user can either:

* Provide a manual observation sequence
  or
* Generate a synthetic sequence for experimentation

---

## Outputs

After training, the following are displayed:

* Initial Transition Matrix (A)
* Final Transition Matrix (A)
* Initial Emission Matrix (B)
* Final Emission Matrix (B)
* Initial Distribution (π)
* Final Distribution (π)
* Log-Likelihood ( P(O \mid \lambda) )
* Convergence status

Optional intermediate values (α, β, γ) can also be examined if enabled.

---

## Visualization

The application includes:

* Log-likelihood vs iteration graph
* State transition diagram of learned hidden states
* Numerical matrices displayed in tabular format

The likelihood curve demonstrates monotonic increase until convergence, confirming correct EM implementation.

---

## Technologies Used

* Python
* NumPy
* Matplotlib
* Streamlit
* Graphviz

---

## Project Structure

```
HMM-Baum-Welch/
│
├── baum_welch.py        # Core algorithm implementation
├── diagram_generator.py # State transition diagram generator
├── app.py               # Streamlit interface
├── requirements.txt     # Required Python packages
├── README.md            # Documentation
```

---

## How to Run the Project

### 1. Clone the Repository

```
git clone https://github.com/AlanAli-byte/HMM-Baum-Welch.git
cd HMM-Baum-Welch
```

### 2. Create Virtual Environment (Recommended)

```
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

### 4. Install Graphviz

Download and install Graphviz from:

[https://graphviz.org/download/](https://graphviz.org/download/)

Ensure it is added to system PATH.

### 5. Run the Application

```
streamlit run app.py
```

The application will open automatically in your browser.

---

## Important Notes

* The Baum–Welch algorithm is implemented from first principles.
* No external HMM libraries are used.
* Scaling is applied in the forward-backward procedure for numerical stability.
* The log-likelihood increases monotonically across iterations.
* The repository is public as required by the assignment guidelines.

---

## Conclusion

This project demonstrates a complete and mathematically accurate implementation of the Baum–Welch algorithm for training Hidden Markov Models.

The interactive interface and visualizations provide insight into how model parameters evolve during iterative optimization.
