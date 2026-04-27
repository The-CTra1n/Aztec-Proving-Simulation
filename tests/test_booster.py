"""Pin the Python booster to the Solidity semantics.

Expected values computed by hand from RewardBooster.sol; documented inline.
"""

from proving_sim.booster import (
    ActivityScore,
    BoostConfig,
    decayed_value,
    record_submission,
    to_shares,
)


def test_first_submission_current():
    cfg = BoostConfig.current()
    score = record_submission(ActivityScore(), epoch=1, cfg=cfg)
    assert score.time == 1
    assert score.value == 125_000
    # After one submission: value=125k, t=14.875M, rhs=22B >> k → floor minimum.
    assert to_shares(score.value, cfg) == cfg.minimum


def test_reach_max_score_current_in_600_epochs():
    """Net +25k/epoch with consecutive submissions; 15M / 25k = 600."""
    cfg = BoostConfig.current()
    score = ActivityScore()
    for epoch in range(1, 601):
        score = record_submission(score, epoch, cfg)
    assert score.value == cfg.max_score
    assert to_shares(score.value, cfg) == cfg.k


def test_one_miss_proposed_yields_75pct():
    """Forum: at max score, 1 miss → ~75% share."""
    cfg = BoostConfig.proposed()
    # time=0, value=max: submit at epoch=2 means 1 missed epoch (epoch 1).
    score = ActivityScore(time=0, value=cfg.max_score)
    score = record_submission(score, epoch=2, cfg=cfg)
    shares = to_shares(score.value, cfg)
    # Exact value: 756_951.
    assert 750_000 < shares < 760_000


def test_two_misses_proposed_near_floor():
    """Forum: at max score, 2 misses → ~1% share."""
    cfg = BoostConfig.proposed()
    score = ActivityScore(time=0, value=cfg.max_score)
    score = record_submission(score, epoch=3, cfg=cfg)
    shares = to_shares(score.value, cfg)
    # Exact value: 14_151 (above minimum=10k).
    assert 10_000 < shares < 20_000


def test_shares_floor_at_zero_value():
    cfg = BoostConfig.current()
    assert to_shares(0, cfg) == cfg.minimum


def test_decay_clamps_at_zero():
    score = ActivityScore(time=0, value=50_000)
    # Decay for 10 epochs @ 100k/epoch → 1M, far exceeds value.
    assert decayed_value(score, epoch=10) == 0


def test_repeat_submission_same_epoch_noop():
    """Solidity only increments when store.time != currentEpoch."""
    cfg = BoostConfig.current()
    score = ActivityScore()
    s1 = record_submission(score, epoch=5, cfg=cfg)
    s2 = record_submission(s1, epoch=5, cfg=cfg)
    assert s1 == s2


def test_reach_max_proposed_in_263_epochs():
    """Net +1400/epoch; 367500 / 1400 = 262.5 → saturates on epoch 263."""
    cfg = BoostConfig.proposed()
    score = ActivityScore()
    for epoch in range(1, 264):
        score = record_submission(score, epoch, cfg)
    assert score.value == cfg.max_score
