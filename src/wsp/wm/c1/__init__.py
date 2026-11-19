"""wme-c1: the 9B confirmatory run orchestration (PREREG_WME, FROZEN 19 Nov 2026; Part II §II.28 serialized path).

One controller (the AWS host) owns the single study log through ``wsp.wm.runguard.ProductionGuard`` and issues
every confirmatory unit through ``wsp.wm.harness.run_units``. One worker process on the single active Lambda
instance executes requests. Nothing in this package selects, tunes or re-reads anything on a confirmatory outcome.
"""
