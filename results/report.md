# Royal Blackjack — exact grade of the shipped agents

Generated 2026-09-25T03:18:32+00:00 (commit 2be6a32).

All numbers are exact expected values from `solver.py` (infinite deck), not simulations.

## Dealer S17 (S17, DAS, no surrender, blackjack pays 3:2)

* Training: 30,000,000 hands of Monte Carlo self-play + 10,000,000 precision-practice deals (497/500 decisions settled at z=3.0, tie tolerance 0.001)
* Perfect play: **-0.5704** per $100
* Agent: **-0.5716** per $100
* Gap to perfect play: **0.13¢** per $100
* Perfect move on **99.7%** of the 330 first decisions

| Hand | Dealer | Agent | Perfect | Cost per $1 | Frequency |
|---|---|---|---|---|---|
| soft 18 | 2 | double | stand | 0.199¢ | 0.091% |

## Dealer H17 (H17, DAS, no surrender, blackjack pays 3:2)

* Training: 30,000,000 hands of Monte Carlo self-play + 10,000,000 precision-practice deals (494/500 decisions settled at z=3.0, tie tolerance 0.001)
* Perfect play: **-0.7892** per $100
* Agent: **-0.7892** per $100
* Gap to perfect play: **0.00¢** per $100
* Perfect move on **100.0%** of the 330 first decisions

