"""Reproducible audit of the CITES RST three-year sharp-increase window.

The package is intentionally split along the paper's evidence chain:

``panel``      CITES trade records -> (year, exporter, taxon) panel
``labels``     panel -> rule states and audit labels
``features``   panel -> 69 predictor columns (three feature sets)
``model``      rolling-origin LightGBM ranking
``audit``      fixed-budget audit, window comparison, case verification
``sensitivity`` threshold / model-family / reporting-field stress tests
``figures``    publication figures drawn from shipped aggregate tables
``checks``     machine-readable recomputation of every headline number
"""

__version__ = "1.0.0"
