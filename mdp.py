# mdp.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class MDP:
    """
    Dense finite MDP:
      - P[s, a, s'] transition probabilities (row-stochastic)
      - R[s, a, s'] reward/cost on transition s --a--> s'
      - terminal[s] indicates absorbing terminal states
      - start_state is the initial state index
    """
    P: np.ndarray          # (S, A, S), row-stochastic
    R: np.ndarray          # (S, A, S), reward on s--a-->s'
    gamma: float           # (0,1]
    terminal: np.ndarray   # (S,), bool
    start_state: int

    def __post_init__(self):
        assert self.P.ndim == 3 and self.R.ndim == 3, "P and R must be 3D (S,A,S)."
        S, A, S2 = self.P.shape
        assert S == S2 and self.R.shape == (S, A, S), "Shapes must match: P:(S,A,S), R:(S,A,S)."
        assert self.terminal.shape == (S,), "terminal must be shape (S,)."
        assert 0 < self.gamma <= 1.0, "gamma must be in (0,1]."
        assert 0 <= self.start_state < S, "start_state out of range."

        # normalize transitions; sanitize
        for s in range(S):
            for a in range(A):
                row = self.P[s, a]
                row[row < 0] = 0.0
                psum = float(row.sum())
                if psum <= 0.0:
                    row[:] = 0.0
                    row[s] = 1.0
                else:
                    row[:] = row / psum

        # terminals absorbing, zero further reward
        for s in range(S):
            if bool(self.terminal[s]):
                for a in range(A):
                    self.P[s, a, :] = 0.0
                    self.P[s, a, s] = 1.0
                    self.R[s, a, :] = 0.0

    @property
    def n_states(self) -> int:
        return int(self.P.shape[0])

    @property
    def n_actions(self) -> int:
        return int(self.P.shape[1])


def build_betting_game_mdp(
    M: int = 5,
    MAX_MONEY: int = 100,
    STAGES: int = 10,
    p_win: float = 0.7,
    p_jackpot: float = 0.05,
    gamma: float = 1.0,       # undiscounted
) -> MDP:
    """
    Betting game MDP (cost-minimization friendly via INF_COST on disabled actions).

    State: (money m in [0..MAX_MONEY], stage s in [1..STAGES]) plus a SINK terminal.
    Actions (A=7): ["zero","one","two","three","four","five","end"]

      - For stages 1..STAGES-1:
          "zero": advance stage, money unchanged
          k in {1..5}: if m > k-1, then:
              win with prob p_win -> m += k
              jackpot with prob p_jackpot -> m += 10k
              loss with remaining prob -> m -= k
            else disabled (INF_COST self-loop)
          "end" disabled (INF_COST self-loop)

      - For stage STAGES:
          only "end" enabled -> go to SINK with reward/cost = MAX_MONEY - m
          all other actions disabled (INF_COST self-loop)

    start state: (m=M, s=1)
    """
    # Define actions locally so the index mapping is stable
    BET_ACTIONS = ["zero", "one", "two", "three", "four", "five", "end"]
    A = len(BET_ACTIONS)
    AIDX = {name: i for i, name in enumerate(BET_ACTIONS)}

    # Number of states: STAGES*(MAX_MONEY+1) + 1 sink
    S_no_sink = (MAX_MONEY + 1) * STAGES
    SINK = S_no_sink
    S = S_no_sink + 1

    P = np.zeros((S, A, S), dtype=float)
    R = np.zeros((S, A, S), dtype=float)
    terminal = np.zeros(S, dtype=bool)
    terminal[SINK] = True

    def idx(m: int, s: int) -> int:
        """Map (m,s) with s in [1..STAGES] into [0..S_no_sink-1]."""
        return (s - 1) * (MAX_MONEY + 1) + m

    # Large positive cost so disabled actions are never optimal when minimizing
    INF_COST = 1e12
    loss_prob = 1.0 - (p_win + p_jackpot)
    if loss_prob < -1e-12:
        raise ValueError("p_win + p_jackpot must be <= 1.")
    loss_prob = max(loss_prob, 0.0)

    # Build transitions for all non-sink states
    for s in range(1, STAGES + 1):
        for m in range(0, MAX_MONEY + 1):
            s_idx = idx(m, s)

            if s < STAGES:
                # ---- betting stages: s = 1..STAGES-1 ----

                # [zero]: always enabled, money unchanged, s -> s+1
                a_zero = AIDX["zero"]
                sp = idx(m, s + 1)
                P[s_idx, a_zero, sp] = 1.0
                # reward stays 0

                # Stakes 1..5
                stake_names = ["one", "two", "three", "four", "five"]
                for k in range(1, 6):
                    a = AIDX[stake_names[k - 1]]

                    if m > k - 1:
                        # win: +k
                        m_win = min(m + k, MAX_MONEY)
                        # jackpot: +10k
                        m_jp = min(m + 10 * k, MAX_MONEY)
                        # lose: -k
                        m_loss = max(m - k, 0)

                        sp_win = idx(m_win, s + 1)
                        sp_jp = idx(m_jp, s + 1)
                        sp_loss = idx(m_loss, s + 1)

                        P[s_idx, a, sp_win] += p_win
                        P[s_idx, a, sp_jp] += p_jackpot
                        P[s_idx, a, sp_loss] += loss_prob
                        # reward stays 0
                    else:
                        # Disabled action -> penalized self-loop
                        P[s_idx, a, s_idx] = 1.0
                        R[s_idx, a, s_idx] = INF_COST

                # [end] disabled before final stage -> penalize
                a_end = AIDX["end"]
                P[s_idx, a_end, s_idx] = 1.0
                R[s_idx, a_end, s_idx] = INF_COST

            else:
                # ---- final stage: s == STAGES ----
                # Only [end] enabled: goes to SINK with reward/cost MAX_MONEY - m
                a_end = AIDX["end"]
                P[s_idx, a_end, SINK] = 1.0
                R[s_idx, a_end, SINK] = float(MAX_MONEY - m)

                # All betting actions disabled -> penalized self-loops
                for name in ["zero", "one", "two", "three", "four", "five"]:
                    a = AIDX[name]
                    P[s_idx, a, s_idx] = 1.0
                    R[s_idx, a, s_idx] = INF_COST

    # SINK: absorbing, zero reward (also enforced by MDP.__post_init__)
    P[SINK, :, SINK] = 1.0
    R[SINK, :, :] = 0.0

    start_state = idx(M, 1)
    return MDP(P=P, R=R, gamma=gamma, terminal=terminal, start_state=start_state)

# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------
if __name__ == "__main__":
    mdp = build_betting_game_mdp(
        M=5, MAX_MONEY=100, STAGES=10,
        p_win=0.7, p_jackpot=0.05, gamma=1.0
    )
    print("[Betting MDP]")
    print(f"S={mdp.n_states}, A={mdp.n_actions}, gamma={mdp.gamma}, start_state={mdp.start_state}")
    print(f"terminal states: {int(np.sum(mdp.terminal))}")
