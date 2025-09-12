XXPLO Variants for Ablation Study

Algorithms implemented (same signature as baseline):
1) XXPLO_PB_RTR_A      : RTR + Archive A only
2) XXPLO_PB_RTR_ACC    : RTR + Acceptance-rate success history only
3) XXPLO_PB_RTR_k      : RTR + Adaptive kRTR only
4) XXPLO_PB_RTR_FDB    : RTR + Low-frequency FDB fallback only (requires fdb_survivor_selection.m)

Location:
- All files are under XXPLO_variants/.

Notes:
- Ensure XXPLO_variants is on MATLAB path or attached to the parallel pool.
- For FDB variant, if using parfor, also attach XXPLO_variants/fdb_survivor_selection.m via addAttachedFiles if needed.

