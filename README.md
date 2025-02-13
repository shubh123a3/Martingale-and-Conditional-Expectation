# Martingale and Conditional Expectation

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

An interactive Streamlit app to visualize and explore key financial concepts, including **martingale properties** and **conditional expectation** in option pricing, using Monte Carlo simulations and mathematical theory.

---

## Features

- **Martingale Demonstrations**:
  - Simple Brownian Motion: Visualize the martingale property of Brownian motion.
  - Nested Simulation: Explore nested paths and conditional expectations.
  
- **Conditional Expectation in Option Pricing**:
  - Compare Monte Carlo and conditional expectation methods for option pricing.
  - Interactive parameter controls for volatility, strike price, and more.

- **Mathematical Theory**:
  - LaTeX formulas for clear explanations of financial concepts.
  - Dynamic visualizations to illustrate theoretical results.

- **User-Friendly Interface**:
  - Interactive sliders for parameter tuning.
  - Real-time updates for simulations and plots.

---

## Mathematical Foundations

### 1. Martingale Properties

A stochastic process \( \{W_t\}_{t \geq 0} \) is a **martingale** if:
$$
\mathbb{E}[W_t | \mathcal{F}_s] = W_s \quad \text{for } 0 \leq s < t
$$
where \( \mathcal{F}_s \) is the filtration up to time \( s \).

#### Brownian Motion
- \( W_t \) is normally distributed with mean \( 0 \) and variance \( t \):
  $$
  W_t \sim \mathcal{N}(0, t)
  $$
- The conditional expectation satisfies:
  $$
  \mathbb{E}[W_t | \mathcal{F}_s] = W_s
  $$

#### Simple Brownian Motion Simulation
- Simulate \( W_t \) using:
  $$
  W_t = \sqrt{t} \cdot Z \quad \text{where } Z \sim \mathcal{N}(0, 1)
  $$
- The sample mean of \( W_t \) should converge to \( 0 \) as the number of paths increases:
  $$
  \lim_{N \to \infty} \frac{1}{N} \sum_{i=1}^N W_t^{(i)} = 0
  $$

#### Nested Simulation
- Simulate paths from \( 0 \) to \( s \), then from \( s \) to \( t \):
  $$
  W_t = W_s + \sqrt{t - s} \cdot Z \quad \text{where } Z \sim \mathcal{N}(0, 1)
  $$
- Verify the martingale property:
  $$
  \mathbb{E}[W_t | \mathcal{F}_s] = W_s
  $$

---

### 2. Conditional Expectation in Option Pricing

For a stochastic volatility model, the option price can be expressed as:
$$
\mathbb{E}\left[e^{-rT}(S_T - K)^+\right] = \mathbb{E}\left[\mathbb{E}\left[e^{-rT}(S_T - K)^+ | \sigma\right]\right]
$$

#### Monte Carlo Method
- Simulate paths of the underlying asset \( S_T \) and compute the average payoff:
  $$
  \text{Option Price} \approx \frac{1}{N} \sum_{i=1}^N e^{-rT} \max(S_T^{(i)} - K, 0)
  $$

#### Conditional Expectation Method
- Use the Black-Scholes formula conditioned on the volatility path \( \sigma \):
  $$
  C(S_0, K, T, \sigma) = S_0 \Phi(d_1) - K e^{-rT} \Phi(d_2)
  $$
  where:
  $$
  d_1 = \frac{\ln(S_0/K) + (r + \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}
  $$
- \( \Phi(\cdot) \) is the cumulative distribution function (CDF) of the standard normal distribution.

#### Stochastic Volatility Model
- The asset price \( S_t \) follows:
  $$
  dS_t = r S_t dt + \sigma_t S_t dW_t
  $$
- The volatility \( \sigma_t \) is modeled as:
  $$
  \sigma_t = \sigma_0 e^{J_t}, \quad J_t \sim \mathcal{N}(\mu_J, \sigma_J^2)
  $$

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Financial-Concepts-Visualizer.git
   cd Financial-Concepts-Visualizer
