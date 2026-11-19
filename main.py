"""Two-Speed Trophy Forecast for make.fun gta6-trophy-rarity-forecast-01.

Forecasts, for every GTA VI PlayStation 5 trophy, five quantiles (5/25/50/75/95%)
of its Exophase earn rate at the case's target. Scored by pinball loss on the
logit scale. Python 3.13, standard library only, no network, deterministic.

Why this shape wins, in one paragraph: launch curves run at two speeds --
story trophies sprint while grind trophies crawl -- and Exophase only refreshes
its tracked-player counts in jumps, which dilutes every rate downward when a
refresh lands. So the forecaster fits each trophy's recent curve locally in
log-time, shrinks noisy slopes toward the tier average, hedges the
extrapolation with the last read, keeps the fan honestly narrow, and expects
the next refresh to bite when the history has gone stale.

The pieces, and the evidence behind each:

* Local log-time line. For each trophy, logit(rate) is regressed on
  ln(1 + hours since the release day began), using only the last 24 hours of
  reads. A launch curve is locally near-linear in log-time; the whole history
  would over-extrapolate because a saturating curve's early slope is far
  steeper than its late slope.
* Tier shrinkage. A trophy's slope is a weighted mix of its own fit and its
  tier's median slope (platinum / gold / silver / bronze), weight n / (n + 4).
  Rare trophies have few earned players and 2-decimal rates, so their own
  slopes are noisy; the tier median is the honest prior.
* Persistence hedge. The median is 0.75 * extrapolated value + 0.25 * last
  read. When growth stalls the last rate is the better guess; when it runs
  on, the fit is.
* Honest spread. Logit sigma = sqrt((0.30 * sqrt(horizon_days))^2 +
  (residual_sd * sqrt(leverage))^2), floored at 0.30. The house baselines use
  a 0.50 random-walk coefficient, which over-widens the fan about 2x when the
  local-fit median is accurate to ~0.2 logit. The residual term re-widens
  the fan exactly for trophies whose fit is noisy.
* Refresh bite. When the history is flat for hours and the horizon is short
  (the practice shape: identical reads, 90-minute horizon), the next Exophase
  refresh is the event: the median shifts down 1.75 logit and the lower tail
  widens by 0.30 logit. The size is derived from the numbers the kit itself
  publishes, not guessed: at the 04:30Z refresh the tracked players went
  638 -> 3,868 while trophies earned went 943 -> 4,276, and every rate but
  the 100% one fell. The 100% trophy accounts for 3,868 of the 4,276 earned,
  so the other 54 trophies' earned counts went 305 -> 408 while the player
  base grew 6.1x: their rates fell by (408/305) * (638/3868) = 0.22x, i.e.
  ln(0.22) = -1.51 logit. The other bound -- earned counts frozen, pure
  dilution by new players -- is 638/3868 = 0.165x, i.e. -1.80 logit. The
  shift sits between the two bounds after the exp(-horizon/12) decay. Both
  terms decay as exp(-horizon_hours / 12), so they vanish for the 4.5-day
  final, where genuine progression dominates.
* A trophy at 100% stays near 100% with a narrow fan pinned high.

Grader contract: read every <id>.in in $MAKEFUN_INPUT_DIR (else /input),
write <id>.out to $MAKEFUN_OUTPUT_DIR (else /output), one JSON object
{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]} per case, at most
65,536 bytes. The output directory is only ever created and written into --
never removed (the trophy-guide lesson: the grader runs as a user who cannot
remove its own /output mount). Two runs on the same case write the same bytes.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Quantile levels, in the order each output row lists them, with the standard
# normal's quantile at each level.
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
ZSCORES = (-1.645, -0.674, 0.0, 0.674, 1.645)


class Tuning:
    """The model's tuned numbers, each justified in the module docstring."""

    window_hours = 24.0      # local-fit window: the case's last day of reads
    fit_share = 0.75         # median = fit_share * fit + (1 - fit_share) * last
    walk_sigma = 0.30        # random-walk spread, logit per sqrt(day) (house: 0.50)
    sigma_floor = 0.30       # narrowest logit spread any trophy gets
    shrink = 4.0             # slope weight n / (n + shrink) on a trophy's own slope
    refresh_shift = -1.75    # median logit shift when a refresh is the event
    refresh_low = 0.30       # extra lower-tail logit width for that refresh
    refresh_tau = 12.0       # refresh effect decays over this many hours
    stale_hours = 6.0        # flat history this long counts as fully stale


def pct_in_range(x):
    """Clamp a percent into the range the logit can take (0.01 .. 99.99)."""
    if x < 0.01:
        return 0.01
    if x > 99.99:
        return 99.99
    return x


def to_logit(pct):
    p = pct_in_range(pct)
    return math.log(p / (100.0 - p))


def from_logit(value):
    """Inverse logit, written overflow-safe for large |value|."""
    if value >= 0:
        return 100.0 / (1.0 + math.exp(-value))
    e = math.exp(value)
    return 100.0 * e / (1.0 + e)


def round_half_up_4(x):
    """Round to 4 decimals, halves up, as JavaScript's Math.round does.

    Whole numbers come back as ints, so json.dumps writes 100 rather than
    100.0. All quantiles here are positive, so floor(v + 0.5) is the
    half-up rule.
    """
    whole = math.floor(x * 10000 + 0.5)
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_stamp(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def median_of(values):
    ordered = sorted(values)
    count = len(ordered)
    mid = count // 2
    if count % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def fit_line(xs, ys):
    """Least squares for y ~ a + b*x, from raw sums.

    Returns (intercept, slope, residual_sd, mean_x, sxx, n). A flat x (no
    variance) means the rate is not moving, so the slope is 0.
    """
    n = len(xs)
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs) - sx * sx / n
    sxy = sum(x * y for x, y in zip(xs, ys)) - sx * sy / n
    slope = sxy / sxx if sxx > 1e-12 else 0.0
    mean_x = sx / n
    intercept = sy / n - slope * mean_x
    sq = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    resid_sd = math.sqrt(sq / max(n - 2, 1))
    return intercept, slope, resid_sd, mean_x, sxx, n


class Reading:
    """One house read of the Exophase page: when, and every trophy's rate."""

    __slots__ = ("at", "rates", "players")

    def __init__(self, at, rates, players):
        self.at = at
        self.rates = rates
        self.players = players


def collect_readings(case):
    """Parse the case's history into time-ordered readings, skipping bad rows."""
    readings = []
    for raw in case.get("history") or []:
        try:
            at = parse_stamp(raw["at"])
        except (KeyError, ValueError):
            continue
        readings.append(Reading(at, raw.get("rates") or [], raw.get("players_tracked")))
    readings.sort(key=lambda r: r.at)
    return readings


def checkpoint_of(case, readings):
    """The case's `at` is the extrapolation start, not the last read's time:
    the last read can sit up to a slot behind it, and the target is `at`
    plus the horizon."""
    try:
        return parse_stamp(case["at"])
    except (KeyError, ValueError):
        return readings[-1].at


def refresh_pressure(readings, checkpoint, horizon_hours):
    """How much the next Exophase refresh should move the forecast.

    A refresh dilutes every rate (new tracked players hold fewer trophies).
    When the history is flat for hours -- same rates and same player count --
    and the horizon is short, that refresh is the event. Returns
    (median_shift, extra_lower_width), both in logit units.
    """
    changes = 1
    last_change = readings[0].at
    for before, after in zip(readings, readings[1:]):
        if after.rates != before.rates or after.players != before.players:
            changes += 1
            last_change = after.at
    stale_h = max(0.0, (checkpoint - last_change).total_seconds() / 3600.0)
    if changes <= 2:
        stale = 1.0
    else:
        stale = min(1.0, stale_h / Tuning.stale_hours)
    weight = math.exp(-horizon_hours / Tuning.refresh_tau) * stale
    return Tuning.refresh_shift * weight, Tuning.refresh_low * weight


def build_series(window, anchor):
    """Per-trophy (x, y) points inside the window, x = ln(1 + hours since
    the release day began), y = logit(rate). Also the x of the last read."""
    xs = [math.log(1.0 + max(0.0, (r.at - anchor).total_seconds() / 3600.0))
          for r in window]
    series = {}
    for pos, x in enumerate(xs):
        rates = window[pos].rates
        for i, rate in enumerate(rates):
            if rate is not None:
                series.setdefault(i, []).append((x, to_logit(rate)))
    return series, xs[-1]


def forecast_case(case):
    trophies = case["trophies"]
    readings = collect_readings(case)
    if not readings:
        # No reads at all: an honest wide fan around 1% beats crashing
        # (a crash makes the whole case invalid).
        return {"trophies": [
            {"id": t["id"], "q": [0.05, 0.3, 1.0, 3.0, 10.0]} for t in trophies
        ]}

    horizon_minutes = float(case.get("horizon_minutes") or 0)
    horizon_hours = horizon_minutes / 60.0
    horizon_days = horizon_minutes / 1440.0
    checkpoint = checkpoint_of(case, readings)

    release = (case.get("game") or {}).get("release_date") or "2026-11-19"
    try:
        anchor = datetime.fromisoformat(release).replace(tzinfo=timezone.utc)
    except ValueError:
        anchor = datetime(2026, 11, 19, tzinfo=timezone.utc)

    shift, low_extra = refresh_pressure(readings, checkpoint, horizon_hours)

    # The local window: the last day of reads, falling back to the last
    # three reads when the case is younger than that.
    cutoff = checkpoint.timestamp() - Tuning.window_hours * 3600.0
    window = [r for r in readings if r.at.timestamp() >= cutoff]
    if len(window) < 3:
        window = readings[-3:]
    series, x_last = build_series(window, anchor)
    x_target = math.log(
        1.0 + max(0.0, (checkpoint - anchor).total_seconds() / 3600.0)
        + horizon_hours
    )

    # First pass: fit every trophy, and gather each tier's slopes.
    fits = {}
    slopes_by_tier = {}
    for idx, trophy in enumerate(trophies):
        points = series.get(idx)
        if not points or len(points) < 2:
            continue
        xs, ys = zip(*points)
        fitted = fit_line(xs, ys)
        fits[idx] = fitted
        slopes_by_tier.setdefault(trophy.get("tier", "bronze"), []).append(fitted[1])
    tier_slope = {tier: median_of(v) for tier, v in slopes_by_tier.items()}

    walk = Tuning.walk_sigma * math.sqrt(max(horizon_days, 0.0))
    rows = []
    for idx, trophy in enumerate(trophies):
        last = None
        for reading in reversed(readings):
            if idx < len(reading.rates) and reading.rates[idx] is not None:
                last = float(reading.rates[idx])
                break
        if last is None:
            last = 0.5
        if last >= 99.99:
            # A trophy every tracked player holds stays held: a narrow fan
            # pinned high; a wide one would pay at every upper quantile.
            rows.append({"id": trophy["id"], "q": [99.0, 99.5, 99.9, 99.99, 99.99]})
            continue
        y_last = to_logit(last)
        fitted = fits.get(idx)
        if fitted is None:
            centre = y_last
            sigma = max(walk, Tuning.sigma_floor)
        else:
            _, slope, resid_sd, mean_x, sxx, n = fitted
            own = n / (n + Tuning.shrink)
            pooled = own * slope + (1.0 - own) * tier_slope.get(
                trophy.get("tier", "bronze"), slope
            )
            # Re-anchor at the last read so the pooled slope moves the
            # forecast from where the trophy actually stands.
            extrapolated = y_last + pooled * (x_target - x_last)
            centre = (Tuning.fit_share * extrapolated
                      + (1.0 - Tuning.fit_share) * y_last)
            leverage = 1.0 + (x_target - mean_x) ** 2 / max(sxx, 1e-9)
            sigma = math.sqrt(walk * walk + (resid_sd * math.sqrt(leverage)) ** 2)
            sigma = max(sigma, Tuning.sigma_floor)
        centre += shift
        sigma_low = math.sqrt(sigma * sigma + low_extra * low_extra)
        quantiles = []
        for z in ZSCORES:
            spread = sigma_low if z < 0 else sigma
            quantiles.append(round_half_up_4(pct_in_range(from_logit(centre + z * spread))))
        # Non-decreasing, whatever the rounding did.
        for k in range(1, len(quantiles)):
            quantiles[k] = max(quantiles[k], quantiles[k - 1])
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        text = json.dumps(forecast_case(case), separators=(",", ":")) + "\n"
        if len(text.encode("utf-8")) > 65536:
            raise ValueError(f"{path.name}: forecast over the 65,536-byte cap")
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
