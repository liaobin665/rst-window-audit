import numpy as np
import pandas as pd
import pytest

from rst_audit.audit import case_verify, si_tables, taxon_group
from rst_audit.labels import roll_sum


def _panel(v, records=None, exporter="XA", taxon="Alpha beta"):
    years = list(range(2016, 2023))
    if len(v) != len(years):
        raise ValueError
    return pd.DataFrame({
        "year": years,
        "exporter": exporter,
        "taxon": taxon,
        "v_cnt": pd.Series(v, dtype=float),
        "records": pd.Series(records if records is not None else [10] * len(years),
                            dtype=float),
        "v_imp_total": 0.0,
        "v_impn_total": 0.0,
        "importers_n": 2.0,
        "top_importer_share": 0.5,
        "cls": "Mammalia",
        "ord": "Primates",
    })


def _labels(exporter="XA", taxon="Alpha beta"):
    return pd.DataFrame([{
        "year": 2022, "exporter": exporter, "taxon": taxon,
        "v_cnt": 10.0, "m5": 100.0, "si_c": 0, "si3_c": 1,
    }])


def _case():
    return pd.DataFrame([{
        "test_year": 2022, "exporter": "XA", "taxon": "Alpha beta",
        "taxon_group": "animal", "business_code": "three_year_window",
        "score": 0.99,
    }])


def test_roll_sum_uses_only_past():
    m = np.array([[1.0, 2.0, 3.0, 4.0]])
    out = roll_sum(m, -5, 0)
    # column j is the sum of the five years strictly before j
    assert out[0, 0] == 0.0
    assert out[0, 3] == 6.0


def test_si_tables_flags_fourfold_jump():
    panel = _panel([100, 100, 100, 100, 100, 1000, 10])
    si = si_tables(panel).set_index(["exporter", "taxon", "year"])
    assert si.loc[("XA", "Alpha beta", 2021), "si_c"] == 1
    assert si.loc[("XA", "Alpha beta", 2020), "si_c"] == 0


def test_case_verify_multi_year_support_is_confirmed():
    panel = _panel([100, 100, 100, 100, 100, 1000, 10])
    out = case_verify(None, panel, _labels(), _case()).iloc[0]
    assert out["verdict"] == "confirmed_multi_base"
    assert bool(out["verified"]) is True
    assert int(out["base_positive_years"]) == 4


def test_case_verify_rejects_thin_trigger_records():
    panel = _panel([100, 100, 100, 100, 100, 1000, 10],
                   records=[10, 10, 10, 10, 10, 1, 10])
    out = case_verify(None, panel, _labels(), _case()).iloc[0]
    assert out["verdict"] == "confirmed_multi_base"
    assert bool(out["low_trigger_records"]) is True
    assert bool(out["verified"]) is False


def test_case_verify_single_year_support_is_watchlist():
    panel = _panel([0, 0, 0, 0, 600, 10000, 10])
    out = case_verify(None, panel, _labels(), _case()).iloc[0]
    assert out["verdict"] == "confirmed_single_base"
    assert bool(out["verified"]) is False


def test_taxon_group_maps_class_and_order():
    panel = pd.DataFrame([
        {"taxon": "A", "cls": "Mammalia", "ord": "Primates"},
        {"taxon": "B", "cls": "UNK", "ord": "Asterales"},
        {"taxon": "C", "cls": "UNK", "ord": "UNK"},
    ])
    groups = taxon_group(panel)
    assert groups.loc["A"] == "animal"
    assert groups.loc["B"] == "plant"
    assert groups.loc["C"] == "other"
