import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
import scipy.stats as stats

st.set_page_config(layout="wide", page_title="Financial Concepts Visualizer")

# Add custom CSS for styling
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
    }
    h1, h2, h3 {
        color: #2a3f5f;
    }
    .stRadio > div {
        display: flex;
        justify-content: center;
    }
    .stSlider {
        padding: 0 20px;
    }
</style>
""", unsafe_allow_html=True)


class OptionType(Enum):
    CALL = 1.0
    PUT = -1.0


def martingale_a(num_paths, t):
    np.random.seed(42)
    W_t = np.random.normal(0.0, np.sqrt(t), [num_paths, 1])
    E_W_t = np.mean(W_t)
    return E_W_t


def martingale_b(NoOfPaths,NoOfSteps, t, s):
    np.random.seed(42)

    Z = np.random.normal(0.0, 1.0, [NoOfPaths, NoOfSteps])
    W = np.zeros([NoOfPaths, NoOfSteps + 1])

    dt1 = s / NoOfSteps
    for i in range(NoOfSteps):
        Z[:, i] = (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])
        W[:, i + 1] = W[:, i] + np.sqrt(dt1) * Z[:, i]

    W_s = W[:, -1]
    # for evwry path we create asub simulation until time t and caluculation
    # time step form [s,t]

    dt2 = (t - s) / float(NoOfSteps)
    W_t = np.zeros([NoOfPaths, NoOfSteps + 1]);
    # to store the result
    E_W_t = np.zeros([NoOfPaths])
    Error = []
    for i in range(0, NoOfSteps):

        # Sub-simulation from time "s" until "t"
        W_t[:, 0] = W_s[i];
        for j in range(0, NoOfSteps):
            Z[:, j] = (Z[:, j] - np.mean(Z[:, j])) / np.std(Z[:, j]);
            # path simulation, from "s" until "t"
            W_t[:, j + 1] = W_t[:, j] + pow(dt2, 0.5) * Z[:, j];
        E_W_t[i] = np.mean(W_t[:, -1])
        Error.append(E_W_t[i] - W_s[i])

        if i == 0:

            fig4, ax4 = plt.subplots(figsize=(20, 16))
            ax4.plot(np.linspace(0, s, NoOfSteps + 1), W[0, :], label="W(t) from 0 to s")
            for j in range(0, NoOfPaths):
                ax4.plot(np.linspace(s, t, NoOfSteps + 1), W_t[j, :], alpha=0.3)
            ax4.set_title("Nested Brownian Motion Simulation")
            ax4.set_xlabel("Time")
            ax4.set_ylabel("W(t)")
            ax4.legend()
            ax4.grid()


    error = np.max(np.abs(E_W_t - W_s))


    return fig4, error


def main():
    st.title("Martingale and Conditional Expectation Visualization")

    concept = st.radio("Select Concept:",
                       ["Martingale Properties", "Conditional Expectation"],
                       horizontal=True)

    if concept == "Martingale Properties":
        st.header("Martingale Demonstration")
        mart_type = st.radio("Select Martingale Type:",
                             ["Simple Brownian Motion", "Nested Simulation"],
                             horizontal=True)

        if mart_type == "Simple Brownian Motion":
            st.markdown(r"""
            **Theory**: A martingale is a stochastic process where the conditional expectation of the next value 
            is equal to the present value. For Brownian motion $W_t$:
            $$E[W_t | \mathcal{F}_s] = W_s \quad \text{for } 0 \leq s < t$$
            This means that given all information up to time s ($\mathcal{F}_s$), 
            the expected future value is the current value.
            """)

            num_paths = st.slider("Number of Paths", 100, 1000000, 1000)
            t = st.slider("Time t", 1.0, 20.0, 10.0)

            E_W_t = martingale_a(num_paths, t)
            st.success(f"Mean value: {E_W_t:.4f} (Expected: 0.0)")
            st.write("As we increase the number of paths, the sample mean should approach 0.")

        else:
            st.markdown(r"""
            **Nested Simulation**: Demonstrates the martingale property through nested paths.
            We simulate Brownian motion up to time s, then create sub-paths from s to t.
            The expectation of W_t given $\mathcal{F}_s$ should equal W_s.
            """)

            NoOfPaths = st.slider("Paths", 100, 1000, 500)
            NoOfSteps = st.slider("Steps", 50, 1000, 500)
            t = st.slider("Total Time t", 1.0, 20.0, 10.0)
            s = st.slider("Intermediate Time s", 0.1, t - 0.1, 5.0)

            fig, error = martingale_b(NoOfPaths,NoOfSteps, t, s)
            st.pyplot(fig)
            st.error(f"Maximum Error: {error:.10f}")
            st.write("The error shows the difference between E[W_t|F_s] and W_s - should be near zero.")

    else:
        st.header("Conditional Expectation in Option Pricing")
        st.markdown(r"""
        **Theory**: Using conditional expectation can reduce variance in Monte Carlo simulations.
        For a stochastic volatility model with volatility $\sigma$, the option price can be expressed as:
        $$E[e^{-rT}(S_T-K)^+] = E\left[E[e^{-rT}(S_T-K)^+ | \sigma]\right]$$
        By conditioning on the volatility path, we can compute the inner expectation analytically.
        """)

        col1, col2 = st.columns(2)
        with col1:
            NoOfPaths = st.slider("Paths", 100, 1000, 200)
            NoOfSteps = st.slider("Steps", 50, 200, 100)
            T = st.slider("Maturity T", 1.0, 10.0, 5.0)
            S0 = st.slider("Initial Price", 50.0, 200.0, 100.0)
            CP = st.radio("Option Type", (OptionType.CALL,OptionType.PUT) )
        with col2:
            K = st.slider("Strike Price", 50.0, 200.0, 80.0)
            muJ = st.slider("Volatility Mean", 0.1, 0.5, 0.3)
            sigmaJ = st.slider("Volatility Std", 0.001, 0.1, 0.005)
            r = st.slider("Risk-free Rate", 0.0, 0.1, 0.0)

        # Generate paths and plots
        def GeneratePaths(NoOfPaths, NoOfSteps, S0, T, muJ, sigmaJ, r):
            X = np.zeros([NoOfPaths, NoOfSteps + 1])
            S = np.zeros([NoOfPaths, NoOfSteps + 1])
            time = np.zeros([NoOfSteps + 1])

            dt = T / float(NoOfSteps)
            X[:, 0] = np.log(S0)
            S[:, 0] = S0

            Z = np.random.normal(0.0, 1.0, [NoOfPaths, NoOfSteps])
            J = np.random.normal(muJ, sigmaJ, [NoOfPaths, NoOfSteps])
            for i in range(0, NoOfSteps):
                if NoOfPaths > 1:
                    Z[:, i] = (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])
                X[:, i + 1] = X[:, i] + (r - 0.5 * J[:, i] ** 2.0) * dt + J[:, i] * np.sqrt(dt) * Z[:, i]
                time[i + 1] = time[i] + dt
            S = np.exp(X)
            paths = {"time": time,
                     "X": X,
                     "S": S,
                     "J": J}
            return paths

        def EUOptionPriceFromMCPaths(CP, S, K, T, r):
            # S is a vector of Monte Carlo samples at T
            if CP == OptionType.CALL:
                return np.exp(-r * T) * np.mean(np.maximum(S - K, 0.0))
            elif CP == OptionType.PUT:
                return np.exp(-r * T) * np.mean(np.maximum(K - S, 0.0))

        def BS_Call_Put_Option_Price(CP, S_0, K, sigma, t, T, r):
            K = np.array(K).reshape([len(K), 1])
            d1 = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma, 2.0))
                  * (T - t)) / (sigma * np.sqrt(T - t))
            d2 = d1 - sigma * np.sqrt(T - t)
            if CP == OptionType.CALL:
                value = stats.norm.cdf(d1) * S_0 - stats.norm.cdf(d2) * K * np.exp(-r * (T - t))
            elif CP == OptionType.PUT:
                value = stats.norm.cdf(-d2) * K * np.exp(-r * (T - t)) - stats.norm.cdf(-d1) * S_0
            return value

        def CallOption_CondExpectation(NoOfPaths, T, S0, K, J, r):
            # single value of j
            J_i = J[:, -1]
            result = np.zeros([NoOfPaths])
            for j in range(0, NoOfPaths):
                sigma = J_i[j]
                result[j] = BS_Call_Put_Option_Price(OptionType.CALL, S0, [K], sigma, 0.0, T, r)

            return np.mean(result)

        Paths = GeneratePaths(NoOfPaths, NoOfSteps, S0, T, muJ, sigmaJ, r)
        timeGrid = Paths["time"]
        X = Paths["X"]
        S = Paths["S"]

        # ploting
        fig1, ax1 = plt.subplots(figsize=(20, 16))
        ax1.plot(timeGrid, X.T)
        ax1.set_xlabel("time")
        ax1.set_ylabel("X(t)")
        ax1.set_title("stock price simulation")
        ax1.grid(True)
        st.pyplot(fig1)  # Display the figure in Streamlit

        # Plot 2: Log Stock Price Simulation
        fig2, ax2 = plt.subplots(figsize=(20, 16))
        ax2.plot(timeGrid, S.T)
        ax2.set_xlabel("time")
        ax2.set_ylabel("S(t)")
        ax2.set_title("log stock price simulation")
        ax2.grid(True)
        st.pyplot(fig2)  # Display the second figure in Streamlit
        # Add option price comparison plots
        NGrid = range(100, 10000, 1000)
        NoOfRuns = len(NGrid)
        resultMc = np.zeros([NoOfRuns])
        resultCondExp = np.zeros([NoOfRuns])
        CP=OptionType.CALL
        for (i, N) in enumerate(NGrid):
            Paths = GeneratePaths(N, NoOfSteps, S0, T, muJ, sigmaJ, r)
            timeGrid = Paths["time"]
            S = Paths["S"]
            resultMc[i] = EUOptionPriceFromMCPaths(CP, S[:, -1], K, T, r)
            J = Paths["J"]
            resultCondExp[i] = CallOption_CondExpectation(N, T, S0, K, J, r)
        fig3, ax3 = plt.subplots(figsize=(20, 16))
        ax3.plot(NGrid, resultMc)
        ax3.plot(NGrid, resultCondExp)
        ax3.set_xlabel("time")
        ax3.set_ylabel('Option price for a given strike, K')
        ax3.set_title('Call Option Price- Convergence')
        ax3.legend(['MC', 'Conditional Expectation'])
        ax3.grid()
        st.pyplot(fig3)
        st.write("Implementation of the option pricing comparison would go here")
        st.write("Including convergence plots comparing MC and Conditional Expectation methods")


if __name__ == "__main__":
    main()