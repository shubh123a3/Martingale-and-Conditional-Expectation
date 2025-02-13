# Financial Concepts Visualizer

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

### Martingale Properties

A stochastic process \( \{W_t\}_{t \geq 0} \) is a **martingale** if:
$$
\mathbb{E}[W_t | \mathcal{F}_s] = W_s \quad \text{for } 0 \leq s < t
$$
where \( \mathcal{F}_s \) is the filtration up to time \( s \).

For **Brownian Motion**:
- \( W_t \) is normally distributed with mean \( 0 \) and variance \( t \):
  $$
  W_t \sim \mathcal{N}(0, t)
  $$
- The conditional expectation satisfies:
  $$
  \mathbb{E}[W_t | \mathcal{F}_s] = W_s
  $$

### Conditional Expectation in Option Pricing

For a stochastic volatility model, the option price can be expressed as:
$$
\mathbb{E}\left[e^{-rT}(S_T - K)^+\right] = \mathbb{E}\left[\mathbb{E}\left[e^{-rT}(S_T - K)^+ | \sigma\right]\right]
$$

- **Monte Carlo Method**:
  Simulate paths of the underlying asset and compute the average payoff.

- **Conditional Expectation Method**:
  Use the Black-Scholes formula conditioned on the volatility path:
  $$
  C(S_0, K, T, \sigma) = S_0 \Phi(d_1) - K e^{-rT} \Phi(d_2)
  $$
  where:
  $$
  d_1 = \frac{\ln(S_0/K) + (r + \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}
  $$

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Financial-Concepts-Visualizer.git
   cd Financial-Concepts-Visualizer
