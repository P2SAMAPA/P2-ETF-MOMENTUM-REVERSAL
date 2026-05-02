# P2-ETF-MOMENTUM-REVERSAL

**P2Quant Engine #93 · Cross-Sectional Momentum-Reversal Transition**

Implements the Jegadeesh-Titman multi-horizon framework — the only engine in the P2Quant suite that explicitly models the momentum-to-reversal transition zone.

## Algorithm

```
Score = α·z(r_12m_skip - r_1m) + β·z(r_6m) - γ·z(r_1m) - δ·z(r_36m)
```

| Horizon | Window | Role |
|---|---|---|
| `r_1m` | 21d | Short-term reversal zone |
| `r_3m` | 63d | Early momentum |
| `r_6m` | 126d | Core momentum confirmation |
| `r_12m_skip` | 22–252d | Full momentum (skip-month adjusted) |
| `r_36m` | 756d | Long-run reversal drag |

Weights `α, β, γ, δ` are refitted every 63 days via rolling OLS on forward 21-day returns.

## Key Features
- **Skip-month adjustment** — excludes most recent month from 12m return to avoid reversal contamination
- **VIX regime conditioning** — boosts reversal penalty in high-vol, momentum weight in low-vol
- **Dispersion filter** — suppresses signal when cross-sectional spread is low (crisis periods)
- **Rolling OLS weights** — adapts to changing return predictability regimes

## Data
- Input: `P2SAMAPA/fi-etf-macro-signal-master-data`
- Output: `P2SAMAPA/p2-etf-momentum-reversal-results`

## Usage
```bash
pip install -r requirements.txt
export HF_TOKEN=your_token
python main.py --universe combined
```

## References
- Jegadeesh & Titman (1993) — *Returns to Buying Winners and Selling Losers*
- Jegadeesh & Titman (2001) — *Profitability of Momentum Strategies*
- De Bondt & Thaler (1985) — *Does the Stock Market Overreact?*
