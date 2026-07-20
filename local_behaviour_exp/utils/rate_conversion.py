"""
Helper functions to calibrate SIR-style epidemiological parameters in units
of "per day" instead of "per environment step", regardless of grid_size.

Usage:
    from rate_conversion import estimate_steps_per_day, day_to_step_prob

    steps_per_day = estimate_steps_per_day(grid_size=20)  # -> ~60

    env_config = {
        "beta": day_to_step_prob(0.5, steps_per_day),               # 50% / day
        "recovery_rate": day_to_step_prob(0.1, steps_per_day),      # 10% / day
        "immunity_loss_prob": day_to_step_prob(0.1, steps_per_day), # 10% / day
        "lethality": day_to_step_prob(0.1, steps_per_day),          # 10% / day
        ...
    }
"""

def estimate_steps_per_day(
    grid_size: int,
    home_stay: float = 20.0,       # midpoint of rng.integers(15, 26)
    work_stay: float = 20.0,       # midpoint of rng.integers(18, 23)
    movement_speed: float = 1.0,   # grid units moved per step (see line 483)
    commute_fraction: float = 0.5, # home/work zones are ~grid_size * 0.5 apart
) -> float:
    """
    Estimate how many environment steps correspond to one simulated day,
    based on the workplace_home_cycle movement pattern. Scales automatically
    with grid_size so you don't need to hand-tune it per environment config.

    steps_per_day = (home_stay + work_stay) + 2 * (commute_distance / movement_speed)
    where commute_distance ~= grid_size * commute_fraction
    """
    commute_distance = grid_size * commute_fraction
    commute_steps = 2 * (commute_distance / movement_speed)  # there and back
    return home_stay + work_stay + commute_steps


def day_to_step_prob(p_day: float, steps_per_day: float) -> float:
    """
    Convert a "probability of event happening within one day" into the
    equivalent per-step probability, assuming the event is checked once
    per step with a constant per-step probability (geometric process).

    Derivation: if p_step is the per-step probability of the event NOT
    happening, then the probability of it not happening across a full day
    is (1 - p_step) ** steps_per_day. We want that to equal (1 - p_day):

        (1 - p_step) ** steps_per_day = 1 - p_day
        p_step = 1 - (1 - p_day) ** (1 / steps_per_day)
    """
    if not (0.0 <= p_day <= 1.0):
        raise ValueError("p_day must be in [0, 1]")
    if steps_per_day <= 0:
        raise ValueError("steps_per_day must be > 0")

    return 1.0 - (1.0 - p_day) ** (1.0 / steps_per_day)


def step_to_day_prob(p_step: float, steps_per_day: float) -> float:
    """
    Inverse of day_to_step_prob: given a per-step probability, compute the
    equivalent cumulative probability of the event happening within one day.
    Useful for sanity-checking the numbers you get out of day_to_step_prob,
    or for reporting per-step rates back in daily terms.
    """
    if not (0.0 <= p_step <= 1.0):
        raise ValueError("p_step must be in [0, 1]")
    if steps_per_day <= 0:
        raise ValueError("steps_per_day must be > 0")

    return 1.0 - (1.0 - p_step) ** steps_per_day


if __name__ == "__main__":
    # Quick sanity check
    for grid_size in [10, 20, 40]:
        spd = estimate_steps_per_day(grid_size)
        print(f"grid_size={grid_size:>3}  steps_per_day≈{spd:.1f}")

        for name, p_day in [
            ("beta", 0.5),
            ("recovery_rate", 0.1),
            ("immunity_loss_prob", 0.1),
            ("lethality", 0.1),
        ]:
            p_step = day_to_step_prob(p_day, spd)
            # round-trip check
            recovered_p_day = step_to_day_prob(p_step, spd)
            print(
                f"  {name:<20} p_day={p_day:<5} -> p_step={p_step:.5f} "
                f"(round-trip p_day={recovered_p_day:.4f})"
            )