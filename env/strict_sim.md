# Environment Specification (+10% Friction)

## 1. Action & Observation Space
*   **Action Space:** Discrete $\{-1, 0, 1\}$ representing [Short, Flat, Long]. (Continuous sizing added in Stage 2).
*   **Observation Space:** $N$-dimensional Box vector defined in `features/engineering.md`.

## 2. Friction Mechanics (+10% Strictness)
Indian F&O markets have high statutory costs. We simulate these and multiply by $1.10$.
*   **Base Transaction Cost ($C_{\text{base}}$):**
    Brokerage (₹20) + STT (0.0125%) + Exchange Txn (0.0019%) + SEBI (0.0001%) + GST (18% on brokerage/txn).
*   **Simulation Cost ($C_{\text{sim}}$):** $C_{\text{sim}} = C_{\text{base}} \times 1.10$
*   **Slippage ($S_t$):** Slippage scales with current market volatility to simulate illiquidity.
    $S_t = (\text{Min\_Tick\_Size} + (\alpha \times V_t)) \times 1.10$

## 3. Drawdown-Penalized Reward Function
The step reward $r_t$ incorporates the portfolio return $P_t$, costs, and a dynamic penalty for approaching the 30% drawdown limit.

$R_t = (\text{Action}_{t-1} \cdot R_{\text{market}, t}) - C_{\text{sim}} - S_t$

**Drawdown Penalty ($\lambda$):**
Let $DD_t$ be the current peak-to-trough drawdown.
$ \text{Penalty}_t = \lambda \cdot (DD_t)^2 $
*If $DD_t > 0.25$ (25%), trigger massive terminal penalty $\beta$ and end episode.*

**Final Step Reward:**
$\text{Reward}_t = R_t - \text{Penalty}_t$
