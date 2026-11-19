# Two-Speed Trophy Forecast — GTA VI week-one earn rates

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/reyxnal/makefun-gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/reyxnal/makefun-gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

A forecaster for [make.fun's gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01):
for every GTA VI PlayStation 5 trophy, five quantiles (5/25/50/75/95%) of its
Exophase earn rate at the case's target, scored by pinball loss on the logit
scale. One program, `main.py` (Python 3.13, standard library only, no
network), reads every `<id>.in` in the input directory and writes
`<id>.out` to the output directory. Deterministic: two runs on the same case
write the same bytes.

## The model

Launch curves move at two speeds: story trophies sprint, grind trophies
crawl — and Exophase refreshes its counts in jumps that dilute every rate
down. The model is built for exactly that shape:

1. **Local logit growth fit.** Per trophy, an OLS line of logit(rate) on
   ln(1 + hours since the release day began), fitted on the case's last 24
   hours of reads. A launch curve is locally almost linear in log-time, and
   the last day is the regime the target grows out of. Fitting the whole
   history instead over-extrapolates: a saturating curve's early slope is
   far steeper than its late slope.
2. **Tier-pooled slopes.** Each trophy's slope is shrunk toward its tier's
   median slope (platinum/gold/silver/bronze). Rare trophies have few earned
   players and 2-decimal rates, so their local slopes are noisy; the tier
   median is the honest prior.
3. **Persistence blend.** Median = 0.75 × extrapolated fit + 0.25 × last
   read. It hedges the extrapolation: when growth stalls, the last rate is
   the better guess.
4. **Calibrated spread.** Logit spread =
   √((0.30·√horizon_days)² + (residual_sd·√leverage)²), floor 0.30. The
   house baselines use a 0.50 random-walk coefficient; that over-widens the
   fan about 2× when the local-fit median is accurate to ~0.2 logit. The
   residual term widens the fan again exactly for trophies whose fit is
   noisy.
5. **Refresh handling.** When the history is flat/stale and the horizon is
   short (the practice shape: identical reads for hours, 90-minute horizon),
   the next Exophase refresh IS the event, so the median shifts down 1.75
   logit and the lower tail widens by 0.30 logit. The size is derived from
   the numbers the kit itself publishes: at the 04:30Z refresh the tracked
   players went 638 → 3,868 while trophies earned went 943 → 4,276, and the
   100% trophy accounts for 3,868 of the 4,276 earned, so the other 54
   trophies' earned counts went 305 → 408 while the player base grew 6.1×:
   their rates fell by (408/305) × (638/3868) = 0.22×, i.e. −1.51 logit
   (the pure-dilution bound, earned counts frozen, is 0.165×, i.e. −1.80
   logit). Both terms decay as exp(−horizon_hours/12) and vanish for the
   4.5-day final, where genuine progression dominates.
6. **A trophy at 100%** gets a narrow fan pinned near 100%.

## Evidence (what was actually measured)

- **Practice (proxy).** The kit does not ship the practice answers, so they
  were reconstructed from the facts the kit publishes (3868 players / 4276
  earned at the 04:30Z refresh, every rate but the 100% one down): the 100%
  trophy holds 3868 of the 4276 earned, so the other 54 trophies' rates fell
  by (408/305) × (638/3868) = 0.22×. On that proxy this program scores
  **0.09** vs the house persistence baseline's **1.40** on the same proxy
  (the kit states the house baseline's true practice loss is 1.51). Under the
  other bound for the truth — earned counts frozen, pure dilution, 0.165× —
  this program scores 0.14 vs the house's 1.69. The real practice score is
  read from the server after submit; the proxy is an estimate, not the truth.
- **Final regime (synthetic).** Six simulated launch histories (saturating
  player growth, weekend boost, slow story, binomial noise, a mid-week
  slowdown) with known truth 4.5 days out: this program beat **both** house
  baselines in all six — 0.089–0.153 vs the house persistence baseline's
  0.150–0.169 and the house curve baseline's 0.146–0.250.
- **Grader-equivalent runs.** `python3 -P -s main.py` from a clean directory:
  exit 0, 0.07 s per run (cap 8 s), byte-identical across two runs; also
  exit 0 twice as user 65534 into the same output directory (the
  trophy-guide lesson: never delete the output folder itself).

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
