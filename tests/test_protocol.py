"""Frozen protocol constants and the rule-free feature set.

A refactor that silently changes one of these values would move every headline
number without touching a single shipped table, so they are pinned here.
"""
import pytest

from rst_audit import config as C
from rst_audit.features import FEATURE_SETS


def test_frozen_rule_and_audit_parameters():
    assert C.SI_MULT == 4.0
    assert C.SI_MIN_MEAN == 100.0
    assert C.GLOBAL_SHARE == 0.10
    assert C.BASE_LEN == 5
    assert C.MIN_BASE_POSITIVE == 2
    assert C.MIN_RECORD_RATIO == 0.5
    assert C.TOP_K == 100


def test_evaluation_window_is_frozen():
    assert C.TRAIN_START == 2005
    assert C.TEST_YEARS == [2018, 2019, 2020, 2021, 2022]
    assert C.SEEDS == [0, 1, 2]
    assert C.N_ROUNDS == 300


def test_feature_set_sizes():
    assert len(FEATURE_SETS["full"]) == 69
    assert len(FEATURE_SETS["no_leak_strict"]) == 60
    assert len(FEATURE_SETS["no_leak_strict_plus"]) == 58


@pytest.mark.parametrize(
    "column",
    ["r1", "r_prev", "max_r3", "si_lag1", "si_lag2", "hv200_lag",
     "m5_log1p", "log5sum", "en_flag"],
)
def test_strict_feature_set_excludes_rule_and_rule_levels(column):
    assert column not in FEATURE_SETS["no_leak_strict"]


def test_strict_feature_set_has_no_duplicates():
    columns = FEATURE_SETS["no_leak_strict"]
    assert len(columns) == len(set(columns))
