# Graph Report - Adversarial_Attack_Defence_Paper  (2026-11-15)

## Corpus Check
- 30 files · ~1,496,860 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1527 nodes · 2660 edges · 67 communities detected
- Extraction: 76% EXTRACTED · 23% INFERRED · 1% AMBIGUOUS · INFERRED: 622 edges (avg confidence: 0.81)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_Community 58|Community 58]]
- [[_COMMUNITY_Community 59|Community 59]]
- [[_COMMUNITY_Community 60|Community 60]]
- [[_COMMUNITY_Community 61|Community 61]]
- [[_COMMUNITY_Community 62|Community 62]]
- [[_COMMUNITY_Community 63|Community 63]]
- [[_COMMUNITY_Community 64|Community 64]]
- [[_COMMUNITY_Community 65|Community 65]]
- [[_COMMUNITY_Community 66|Community 66]]
- [[_COMMUNITY_Community 67|Community 67]]

## God Nodes (most connected - your core abstractions)
1. `adaptive_bpda()` - 34 edges
2. `clean_control()` - 32 edges
3. `run()` - 28 edges
4. `parse_args()` - 23 edges
5. `run()` - 21 edges
6. `main()` - 20 edges
7. `attack_frame()` - 19 edges
8. `main()` - 18 edges
9. `run()` - 18 edges
10. `evaluate_condition()` - 17 edges

## Surprising Connections (you probably didn't know these)
- `Detection-Level Ensembles (7 NMS-merged ensembles)` --semantically_similar_to--> `External Defenses (input preprocessing, image/prompt ensembling, JPEG compression)`  [INFERRED] [semantically similar]
  CV_Research_Paper___CVPR_Format.pdf → docs/Presentation.pdf
- `Detection-Level Ensembles (7 NMS-merged ensembles)` --semantically_similar_to--> `Prompt Smoothing Ensemble (early defense pipeline concept)`  [INFERRED] [semantically similar]
  CV_Research_Paper___CVPR_Format.pdf → docs/Presentation.pdf
- `Attack Damage Result (YOLO 0.4519->0.0619 PGD; Florence 0.3300->0.2005 PGD)` --semantically_similar_to--> `Attack Damage at N=1000 (YOLO 0.4765->0.0737 PGD; Florence 0.3605->0.2243 PGD)`  [INFERRED] [semantically similar]
  CV_Research_Paper___CVPR_Format.pdf → docs/CS24MTECH14020_CVPR_Project_Report.pdf
- `Attack Damage Result (YOLO 0.4519->0.0619 PGD; Florence 0.3300->0.2005 PGD)` --semantically_similar_to--> `Preliminary Florence-2 Clean Baseline 0.297 mAP`  [INFERRED] [semantically similar]
  CV_Research_Paper___CVPR_Format.pdf → docs/Presentation.pdf
- `t_cli_defaults()` --calls--> `parse_args()`  [INFERRED]
  tests/test_rescore.py → archive/Phase3/FGSM_Florence2_OCR_Robust.py

## Hyperedges (group relationships)
- **Phase-3 v2 Locked Paper Pipeline (N=1000, 9 tracks)** — readme_locked_v2_scripts, memory_florence_2_base, memory_yolov8x_worldv2, memory_fgsm_attack, memory_pgd_attack, memory_patch_attack, claude_apply_all_defenses_gpu, memory_v2_results_n1000 [EXTRACTED 1.00]
- **Documented Negative-Result Defenses** — memory_nlm_defense, memory_bit_depth_defense, memory_random_resize_pad_defense, memory_phase2_archive_variant_c, memory_tier2_survey_completeness, memory_phase2_archive_nlm_failure_insight [EXTRACTED 0.95]
- **Novel Patch-Specific Defenses (2024-2026)** — memory_pad_patch_defense, memory_dispatch_defense, memory_saliuitl, memory_patch_attack, memory_tier3_novel_methods [INFERRED 0.85]
- **Ens_JMG three-branch NMS-merged defense (JPEG + Median + Gaussian)** — cv_research_paper___cvpr_format_ens_jmg, cv_research_paper___cvpr_format_jpeg_compression, cv_research_paper___cvpr_format_median_filter, cv_research_paper___cvpr_format_gaussian_blur, cv_research_paper___cvpr_format_class_aware_nms_merge [EXTRACTED 1.00]
- **Nine attack/model/task evaluation settings (3 attacks x YOLO-OD, Florence-OD, Florence-OCR)** — cv_research_paper___cvpr_format_fgsm_attack, cv_research_paper___cvpr_format_pgd_attack, cv_research_paper___cvpr_format_patch_attack, cv_research_paper___cvpr_format_yolov8x_worldv2, cv_research_paper___cvpr_format_florence_2_base, cv_research_paper___cvpr_format_map_metric, cv_research_paper___cvpr_format_ocr_self_consistency_metric [EXTRACTED 1.00]
- **Defense screening funnel: solos and pipelines and GPU novel filters screened down to detection-level ensembles** — cv_research_paper___cvpr_format_solo_transforms, cv_research_paper___cvpr_format_sequential_pipelines, cs24mtech14020_cvpr_project_report_gpu_novel_filters, cv_research_paper___cvpr_format_detection_level_ensembles, cv_research_paper___cvpr_format_no_single_solo_robust_everywhere [EXTRACTED 1.00]
- **Novelty Pipeline: attacked image -> JPEG-75 -> Prompt Smoothing Ensemble -> Gaussian noise -> spatial smoothing -> quantization/mixing -> Florence-2 inference -> mAP** — fgsm_fast_gradient_sign_method, pgd_projected_gradient_descent, jpeg_compression_quality_75, prompt_smoothing_ensemble, adaptive_gaussian_noise, adaptive_spatial_smoothing, quantization_channel_mixing_dropout, florence_2_base, coco_map_metric [EXTRACTED 1.00]
- **PGD result chain: clean 0.297 -> attacked 0.166 -> defended 0.199, validated by the stage-level ablation** — result_apscon_clean_map_0297, result_apscon_pgd_undefended_0166, result_apscon_defended_pgd_0199, ablation_external_internal_full_pipeline [EXTRACTED 1.00]
- **Zero-shot VFM comparison on COCO/Flickr30k/RefCOCO benchmarks** — florence_2_base, florence_2_large, flamingo, kosmos_2, dino_self_supervised, clip, refcoco_benchmark_family [EXTRACTED 1.00]
- **Ensembles sweep the top ranks on Florence-2 detection under both gradient attacks and are the only net-positive defense under Patch** — fgsm_phase3_florence_v2_ens_jpeg_median_gaussian, pgd_phase3_florence_v2_ens_jpeg_median_gaussian, patch_phase3_florence_v2_ens_blur_tvm_combo, patch_phase3_yolo_v2_ens_jpeg_median_tvm, pgd_phase3_yolo_v2_ens_blur_tvm_combo [INFERRED 0.90]
- **Patch attack causes little damage (Florence 0.0338, YOLO 0.0228) so most solo transforms are net-harmful — the survey's core negative result** — patch_phase3_florence_v2_attacked, patch_phase3_yolo_v2_attacked, patch_phase3_florence_v2_tvm, patch_phase3_yolo_v2_blur_tvm, patch_phase3_florence_v2_gaussian, patch_phase3_yolo_v2_finding, patch_phase3_florence_v2_finding [EXTRACTED 1.00]
- **TVM ranks #1 on OCR self-consistency under FGSM and PGD while ranking last / harmful on Florence-2 detection — the clearest task-dependent defense split** — fgsm_florence2_ocr_robust_tvm, pgd_florence2_ocr_robust_tvm, fgsm_phase3_florence_v2_tvm, pgd_phase3_florence_v2_tvm, patch_phase3_florence_v2_tvm, patch_florence2_ocr_robust_tvm [INFERRED 0.88]
- **Ensemble-vs-solo crossover: ensembling wins on Florence-2 (both attacks) and on YOLO under PGD, but a solo (blur_tvm) wins on YOLO under FGSM** — phase3_fgsm_florance_finding, phase3_fgsm_yolo_finding, phase3_pgd_florance_finding, phase3_pgd_yolo_finding, phase3_verdict_taxonomy [INFERRED 0.90]
- **Phase-2 negative-result cluster: variants A-D all conclude that input transforms fail once judged against a clean+defense floor rather than the attacked baseline** — phase2_variant_a_finding, phase2_variant_b_finding, phase2_variant_c_finding, phase2_variant_d_finding, phase2_variant_a_net_gain_definition, phase2_variant_b_strictgain_metric [INFERRED 0.88]
- **ens_jpeg_median_tvm is the top-ranked defense in three independent tracks: Florence-2 OCR (FGSM), Florence-2 detection (PGD), and YOLO detection (PGD)** — fsgm_phase2_ocra_ens_jpeg_median_tvm, phase3_fgsm_florance_ens_jpeg_median_tvm, phase3_pgd_florance_ens_jpeg_median_tvm, phase3_pgd_yolo_ens_jpeg_median_tvm [EXTRACTED 1.00]
- **Ensembling consistently outperforms solo transforms across patch (Florence-2, YOLO) and FGSM (Variant Z) tracks** — phase3_patch_florance_finding_ensembling_wins, phase3_patch_yolo_finding_ensembling_wins, phase2_variant_z_finding_ensembles_dominate_ranking, vqa_ens_jpeg_median_tvm [INFERRED 0.90]
- **Under the net-gain criterion (defended_mAP minus clean+defense floor) every Variant X/Y/A/B/C defense is NOT HELPFUL, even when raw Recovery% is large** — variant_x_finding_all_not_helpful, variant_y_finding_cutout_destroys, yolo_variant_a_finding_recovery_vs_net_gain, yolo_variant_b_finding_all_not_helpful, yolo_variant_c_finding_resize_is_the_poison [INFERRED 0.90]
- **Geometry-altering transforms (cutout, random_resize_pad, nlm_resize, full_pipeline) collapse clean accuracy far below any attack damage** — variant_y_cutout_family, yolo_variant_b_random_resize_pad, yolo_variant_c_nlm_resize, yolo_variant_c_full_pipeline [INFERRED 0.85]
- **Both archived Florence-2 phase-2 tracks report near-floor mAP (0.030 and 0.0006), so their defense rankings are not interpretable** — results_phase2_v2_results_plot_v2_low_absolute_map, results_phase2_varianta_results_plot_varianta_near_zero_map, results_phase2_varianta_results_plot_varianta_above_clean_anomaly, results_phase2_varianta_yolo_results_plot_monotonic_degradation [INFERRED 0.85]
- **The results_phase2_variantA directory contains stale 'v2'-titled duplicates of its own Variant A figures plus a broken axis/layout render** — results_phase2_varianta_results_plot_v2_figure, results_phase2_varianta_results_plot_v2_degenerate_axis, results_phase2_varianta_results_plot_v2_layout_bug, results_phase2_varianta_visual_comparison_v2_stale_title, results_phase2_varianta_visual_comparison_varianta_duplicate_pair [INFERRED 0.85]
- **All four archived phase-2 visual comparison grids use the same COCO val2017 living-room sample at FGSM eps=0.03** — results_phase2_v2_visual_comparison_v2_grid, results_phase2_varianta_visual_comparison_v2_grid, results_phase2_varianta_visual_comparison_varianta_grid, results_phase2_varianta_yolo_visual_comparison_grid, results_phase2_v2_visual_comparison_v2_coco_scene [EXTRACTED 1.00]
- **Phase-2 variant sweep: variants D, X, Y and Z all evaluate FGSM eps=0.03 against a clean baseline of ~0.36 mAP on the same COCO val2017 sample** — results_phase2_variantd_results_plot_v2_figure, results_phase2_variantx_results_plot_v2_figure, results_phase2_varianty_results_plot_v2_figure, results_phase2_variantz_results_plot_variantz_figure [EXTRACTED 1.00]
- **Negative-result cluster: sequential JPEG/median/TVM chains (D), novel GPU transforms (X) and cutout variants (Y) all fail to beat the undefended attacked mAP, isolating Variant Z's NMS-merged ensembles as the only design that recovers** — results_phase2_variantd_results_plot_v2_negative_result, results_phase2_variantx_results_plot_v2_cost_vs_gain_tradeoff, results_phase2_varianty_results_plot_v2_cutout_negative_result, results_phase2_variantz_results_plot_variantz_ensemble_floor [INFERRED 0.85]
- **Core 3x2 experimental grid: {FGSM, PGD, Patch} x {Florence-2-Base, YOLOv8x-worldv2}** — results_phase3_florence_fgsm_attacked_vs_recovered_figure, results_phase3_florence_pgd_attacked_vs_recovered_figure, results_phase3_florence_patch_attacked_vs_recovered_figure, results_phase3_yolo_fgsm_attacked_vs_recovered_figure, results_phase3_yolo_pgd_attacked_vs_recovered_figure, results_phase3_yolo_patch_attacked_vs_recovered_figure, attack_fgsm_eps003, attack_pgd_eps003, attack_patch, model_track_florence2_base, model_track_yolov8x_worldv2 [EXTRACTED 1.00]
- **Identical 8-defense bank ranked in all six panels (5 solos + 3 ensembles)** — defense_solo_tvm, defense_solo_median, defense_solo_gaussian, defense_solo_blur_tvm, defense_solo_jpeg, defense_ens_jpeg_median_tvm, defense_ens_jpeg_median_gaussian, defense_ens_blur_tvm_combo, phase3_v1_defense_bank_five_solos_three_ensembles [EXTRACTED 1.00]
- **PGD story: largest collapse on both tracks, largest defense gains, ensembles at the top** — attack_pgd_eps003, results_phase3_yolo_pgd_attacked_vs_recovered_baseline, results_phase3_florence_pgd_attacked_vs_recovered_baseline, results_phase3_yolo_pgd_attacked_vs_recovered_best, results_phase3_florence_pgd_attacked_vs_recovered_best, defense_ens_jpeg_median_tvm, defense_ens_blur_tvm_combo, phase3_v1_finding_pgd_most_damaging [INFERRED 0.90]
- **Authoritative v2 {FGSM, PGD, Patch} x {Florence-2, YOLOv8x-worldv2} experimental grid** — results_phase3_florence_fgsm_v2_attacked_vs_recovered_figure, results_phase3_florence_pgd_v2_attacked_vs_recovered_figure, results_phase3_florence_patch_v2_attacked_vs_recovered_figure, results_phase3_yolo_fgsm_v2_attacked_vs_recovered_figure, results_phase3_yolo_pgd_v2_attacked_vs_recovered_figure, results_phase3_yolo_patch_v2_attacked_vs_recovered_figure, attack_fgsm_eps003, attack_pgd_eps003, attack_patch, model_florence2_base, model_yolov8x_worldv2 [EXTRACTED 1.00]
- **Locked v2 defense bank evaluated identically in all six panels: 5 solos + 3 ensembles** — defense_solo_tvm, defense_solo_blur_tvm, defense_solo_gaussian, defense_solo_median, defense_solo_jpeg, defense_ens_blur_tvm_combo, defense_ens_jpeg_median_tvm, defense_ens_jpeg_median_gaussian, figure_family_attacked_vs_recovered_barchart [EXTRACTED 1.00]
- **v1 -> v2 lineage: each current figure has an archived v1 counterpart under archive/Logs_Extra/** — archive_results_phase3_florence_fgsm_v1, archive_results_phase3_florence_pgd_v1, archive_results_phase3_florence_patch_v1, archive_results_phase3_yolo_fgsm_v1, archive_results_phase3_yolo_pgd_v1, archive_results_phase3_yolo_patch_v1 [INFERRED 0.80]

## Communities

### Community 0 - "Community 0"
Cohesion: 0.03
Nodes (146): Exception, build_branches(), {branch: PIL} for the requested branches.      `pc.apply_all_defenses_gpu` produ, anchor_scores_from_pred(), assert_gaussian_matches(), assert_median_matches(), attack_frame(), bpda_branch() (+138 more)

### Community 1 - "Community 1"
Cohesion: 0.02
Nodes (124): log(), print_banner(), base_parser(), branch_names(), clean_control(), consensus_vote_text(), defense_table(), env_flag() (+116 more)

### Community 2 - "Community 2"
Cohesion: 0.04
Nodes (109): cell(), main(), +0.044 [+0.037, +0.052]', bold when the 95% CI excludes zero., Load with the Phase-3 flash-attn / erfinv workarounds.          float16 on GPU (, _area_all_index(), bc_percentile_ci(), bias_decomposition(), boot_summary() (+101 more)

### Community 3 - "Community 3"
Cohesion: 0.04
Nodes (93): Archived v1 counterpart: archive/Logs_Extra/results_phase3_florence_fgsm, Archived v1 counterpart: archive/Logs_Extra/results_phase3_florence_patch, Archived v1 counterpart: archive/Logs_Extra/results_phase3_florence_pgd, Archived v1 counterpart: archive/Logs_Extra/results_phase3_yolo_fgsm, Archived v1 counterpart: archive/Logs_Extra/results_phase3_yolo_patch, Archived v1 counterparts: archive/Logs_Extra/results_phase3_yolo_pgd and results_phase3_pgd_yolo (two v1 dirs for one v2 cell), Attack: FGSM eps=0.03, Attack: Adversarial Patch (+85 more)

### Community 4 - "Community 4"
Cohesion: 0.04
Nodes (82): consensus_vote_text(), decode_ocr(), defend_blur_tvm(), defend_gaussian(), defend_jpeg(), defend_median(), defend_tvm(), fgsm_attack_ocr() (+74 more)

### Community 5 - "Community 5"
Cohesion: 0.05
Nodes (77): apply_all_defenses_gpu (5 locked solos), apply_survey_defenses (full survey bank), Dataset Path Inconsistency by Track, Rationale: Never Modify Frozen v2 / SOLO_DEFENSES / ENSEMBLES, merge_branches_nms (class-aware NMS merge), phase3_common.py (shared core), vlm_ftune Conda Environment, Adversarial Robustness Survey (Project) (+69 more)

### Community 6 - "Community 6"
Cohesion: 0.04
Nodes (76): Stage-level ablation under PGD: External-only +0.012 (8.8%), Internal-only +0.021 (16.4%), Full pipeline +0.033 (25.2%), Adaptive Gaussian Noise Injection (sigma proportional to mean|x|, sigma_base ~= 0.06), Adaptive Spatial Smoothing (5x5 kernel, gradient-magnitude-weighted average/median blend), Adversarial Training (effective but costly retraining baseline), AutoAttack (named as out-of-scope stronger adaptive attack), BART-based Multi-Modality Encoder-Decoder Transformer, BPDA / EOT adaptive evaluation (deferred to future work), Broader impacts: VFM fragility risks autonomous driving, medical image analysis, surveillance and dexterous-robot control; bypassed visual recognition and misleading generative outputs (+68 more)

### Community 7 - "Community 7"
Cohesion: 0.04
Nodes (75): 25+ Defense Configurations Tested (design history table), Attack Damage at N=1000 (YOLO 0.4765->0.0737 PGD; Florence 0.3605->0.2243 PGD), Rationale: Classification-Era Winners Fail on Detection (NLM, random resize), GPU-Accelerated Novel Filters (spectral gate, Laplacian smooth, gamma stabilize, cutout) - all not helpful, NIPS 2017 Defense Competition (random resize-and-pad ranked #2), Three-Ensemble Recommendation at N=1000 (Ens_JMG for OD, Ens_JMT for OD+OCR), Anisotropic Diffusion Defense, Attack Damage Result (YOLO 0.4519->0.0619 PGD; Florence 0.3300->0.2005 PGD) (+67 more)

### Community 8 - "Community 8"
Cohesion: 0.04
Nodes (68): Variant Z FGSM eps=0.03 attacked mAP 0.2733 (damage 0.0871), Variant Z clean baseline mAP 0.3604, ens_4way rank 4: 0.3195 (+53.1%), ens_blur_tvm_combo rank 1: 0.2733 to 0.3263 (+0.0530, +60.8%), Phase 2 Variant Z: head-to-head attacked vs recovered mAP ranking (11 defenses), Finding: ensembles take 4 of the top 6 ranks under FGSM; solo avg +0.0134 vs ensemble sweep of the podium, median solo rank 5: 0.3160 (+49.0%), top solo defense, tvm solo rank 11: 0.2625 (-0.0108, -12.4%, HURTS) (+60 more)

### Community 9 - "Community 9"
Cohesion: 0.05
Nodes (65): Attacked OCR baseline sim_avg = 0.3057 (converged over 500/500 images), ens_blur_tvm_combo: defended_sim 0.4771, recovery +0.1714 (rank 2), ens_jpeg_median_gaussian: defended_sim 0.4706, recovery +0.1648 (rank 3), ens_jpeg_median_tvm: defended_sim 0.5064, recovery +0.2007 (rank 1), FGSM Phase-2 OCR Recovery Console Summary (Florence-2 OCR, 500 images), Finding: all top-3 OCR defenses are ENSEMBLES; each lifts similarity by +0.16 to +0.20 (~54-66% relative) but none restores the ~1.0 clean self-consistency, Metric: OCR similarity average (char-level self-consistency vs clean output), Methodology shift: phase-2 variants A-D judge single/chained transforms with strict net-gain floors (all negative), phase-3 switches to NMS-merged ensembles ranked by verdict (mostly positive) (+57 more)

### Community 10 - "Community 10"
Cohesion: 0.04
Nodes (50): evaluate_coco(), evaluate_coco(), fgsm_attack(), fgsm_attack_multi_eps(), _letterbox_image(), _parse_yolo_results(), Convert a single YOLO Results object to COCO-format detections., Run YOLOv8-World inference on a single image. (+42 more)

### Community 11 - "Community 11"
Cohesion: 0.07
Nodes (55): FGSM OCR attacked self-consistency 0.3287, ens_jpeg_median_tvm — rank 2 FGSM OCR, 0.5288 (+0.2001, +29.8%), Figure: FGSM vs Florence-2 OCR defense ranking table, Finding: all 8 defenses RECOVER FGSM OCR; solo TVM beats every ensemble (best-of-kind winner SOLO), median (solo) — worst FGSM OCR defense, 0.4275 (+0.0987, +14.7%), Metric: Florence-2 OCR self-consistency (char SequenceMatcher vs own clean output), TVM (solo) — best FGSM OCR defense, 0.3287 to 0.5615 (+0.2327, +34.7%), FGSM eps0.03 attacked Florence mAP 0.2731 (damage 0.0874) (+47 more)

### Community 12 - "Community 12"
Cohesion: 0.06
Nodes (52): Attacked (no defense): mAP 0.0256 / 0.0259 / 0.0225 at eps 0.003 / 0.010 / 0.030, Clean baseline mAP = 0.030 (dashed reference line, both panels), Defense-cost bars on clean images: jpeg -0.001, nlm -0.004, tvm -0.004, smoothvlm -0.023, diffpure -0.030, DiffPure total failure: mAP ~0.000 at all epsilon; clean-image cost -0.030 (full baseline), Figure: FGSM mAP vs Epsilon + Defense Cost on Clean Images (Variant 2), two-panel, JPEG defense is best: 0.0290 / 0.0289 / 0.0273, above attacked at every epsilon, Caveat: absolute Florence mAP ceiling is only 0.030, so all deltas sit in a near-floor regime, NLM defense flat at ~0.0230, below the undefended attacked curve at small epsilon (+44 more)

### Community 13 - "Community 13"
Cohesion: 0.05
Nodes (50): Variant D clean baseline mAP 0.360 (dashed reference line, both panels), Variant D right panel: Defense Cost on Clean Images bar chart vs clean baseline 0.360, Variant D left panel: FGSM eps=0.03 single-point scatter, attacked (no defense) mAP ~0.274, Variant D results plot: FGSM mAP-vs-epsilon + defense cost on clean (two-panel), jpeg_median_tvm (Variant D): clean mAP 0.264, cost -0.097, attacked-recovery ~0.268 (below attacked), jpeg_tvm (Variant D): clean mAP 0.292, clean cost -0.068, jpeg_tvm_svd (Variant D): clean mAP 0.290, cost -0.074, only defense above attacked (~0.279), median_tvm (Variant D): clean mAP 0.268, cost -0.093, worst recovery ~0.256 (+42 more)

### Community 14 - "Community 14"
Cohesion: 0.08
Nodes (42): Florence Clean Baseline mAP 0.360 (Variant B), FGSM Epsilon Sweep Grid (0.003, 0.010, 0.030), Variant B Florence FGSM mAP-vs-Epsilon + Defense-Cost Figure, Florence FGSM Curves Nearly Flat Across Epsilon (0.298 to 0.269), random_resize_pad Collapses Florence mAP to ~0.09 (clean cost -0.249), Variant B Solo Defense Bank (tvm, nlm, svd, random_resize_pad), SVD Near-Zero Clean Cost on Florence (-0.001), TVM Solo Marginal Gain on Florence at eps=0.03 (0.280 vs 0.269) (+34 more)

### Community 15 - "Community 15"
Cohesion: 0.07
Nodes (33): _box_iou_local(), _compute_score(), _map_label(), non_max_suppression(), patch_attack(), process_one(), run_inference(), _box_iou_local() (+25 more)

### Community 16 - "Community 16"
Cohesion: 0.15
Nodes (23): consensus_vote_text(), decode_ocr(), fgsm_attack_ocr(), get_ocr_text(), image_id_from_name(), load_state(), make_default_state(), normalize_text() (+15 more)

### Community 17 - "Community 17"
Cohesion: 0.14
Nodes (24): defend_blur_tvm(), defend_full_pipeline(), defend_jpeg_tvm_nlm(), defend_nlm_resize(), fgsm_attack(), fgsm_attack_multi_eps(), _gaussian_blur(), _jpeg() (+16 more)

### Community 18 - "Community 18"
Cohesion: 0.18
Nodes (21): consensus_vote_text(), decode_ocr(), defend_blur_tvm(), defend_gaussian(), defend_jpeg(), defend_median(), defend_tvm(), format_eps() (+13 more)

### Community 19 - "Community 19"
Cohesion: 0.19
Nodes (20): consensus_vote_text(), decode_ocr(), defend_blur_tvm(), defend_gaussian(), defend_jpeg(), defend_median(), defend_tvm(), get_ocr_text() (+12 more)

### Community 20 - "Community 20"
Cohesion: 0.13
Nodes (16): evaluate_coco(), fgsm_attack(), fgsm_attack_multi_eps(), _letterbox_image(), _parse_yolo_results(), Convert a single YOLO Results object to COCO-format detections., Run YOLOv8-World inference on a single image., Run YOLOv8-World inference on a batch of images. (+8 more)

### Community 21 - "Community 21"
Cohesion: 0.15
Nodes (14): box_iou(), _compute_score(), evaluate_coco(), fgsm_attack_multi_eps(), _map_label(), non_max_suppression(), Map Florence-2 label to COCO name. Try exact match first, then mapping., Heuristic confidence score based on box geometry.     COCO mAP needs varying sco (+6 more)

### Community 22 - "Community 22"
Cohesion: 0.16
Nodes (13): box_iou(), _compute_score(), fgsm_attack_multi_eps(), _map_label(), non_max_suppression(), Map Florence-2 label to COCO name. Try exact match first, then mapping., Heuristic confidence score based on box geometry.     COCO mAP needs varying sco, Run Florence-2 OD with label mapping and heuristic scores. (+5 more)

### Community 23 - "Community 23"
Cohesion: 0.2
Nodes (12): _box_iou_local(), _compute_score(), _map_label(), non_max_suppression(), process_one(), run_inference(), _box_iou_local(), _compute_score() (+4 more)

### Community 24 - "Community 24"
Cohesion: 0.19
Nodes (13): _box_iou_local(), compute_score(), hf_cached(), hf_hub_cache(), hf_offline(), non_max_suppression(), preflight_weights(), VERBATIM from run_survey_florence_detection.py:335-347. (+5 more)

### Community 25 - "Community 25"
Cohesion: 0.25
Nodes (8): letterbox(), load_novel_defenses(), main(), make_attacks(), make_inference_fn(), parse_args(), _pick_freest_gpu(), # NOTE: This is a SURVEY script — it does NOT modify phase3_common.py or any

### Community 26 - "Community 26"
Cohesion: 0.18
Nodes (11): AP50 Metric (reported alongside mAP in the N=1000 report), Rationale: Detection Cache Bug (silently reused stale entries across configurations), N=1000 COCO val2017 Evaluation (earlier report version), COCO val2017 Full Set (5000 images), Limitation: Only Two Models and One Dataset (Grounding DINO, OWLv2, LVIS, Objects365 untested), Limitation: No Adaptive Attacks (recoveries are an upper bound), mAP@[.5:.95] Detection Metric, pycocotools COCOeval (+3 more)

### Community 27 - "Community 27"
Cohesion: 0.53
Nodes (5): letterbox(), _parse_yolo_result(), pgd_attack(), run_inference(), unletterbox()

### Community 28 - "Community 28"
Cohesion: 0.67
Nodes (2): Run ablation study to analyze each defense component, run_ablation_study()

### Community 30 - "Community 30"
Cohesion: 1.0
Nodes (1): A compute_one that drops a condition must abort the run, not report.

### Community 31 - "Community 31"
Cohesion: 1.0
Nodes (1): An image that raises must fail the count assert, not be silently dropped.

### Community 32 - "Community 32"
Cohesion: 1.0
Nodes (1): pycocotools' boilerplate must not reach the committed run.log.

### Community 33 - "Community 33"
Cohesion: 1.0
Nodes (1): The OCR determinism pass is skippable and actually saves a generate.

### Community 34 - "Community 34"
Cohesion: 1.0
Nodes (1): The full table must not be echoed into stdout (it is in the .md).

### Community 35 - "Community 35"
Cohesion: 1.0
Nodes (1): A second run in the same directory must not clobber the first.

### Community 36 - "Community 36"
Cohesion: 1.0
Nodes (1): Evaluate phase3_common's ensemble constants WITHOUT importing torch.      phase3

### Community 37 - "Community 37"
Cohesion: 1.0
Nodes (1): The classifier must agree with the repo's real .gitignore rules.

### Community 38 - "Community 38"
Cohesion: 1.0
Nodes (1): Smoke-test output must be impossible to commit.

### Community 39 - "Community 39"
Cohesion: 1.0
Nodes (1): Duplicate a text stream into a file handle.

### Community 40 - "Community 40"
Cohesion: 1.0
Nodes (1): Tee stdout+stderr into `path` (append) with a session header.      Append, not t

### Community 41 - "Community 41"
Cohesion: 1.0
Nodes (1): Where pycocotools' per-condition detection dumps go.      keep=False (default):

### Community 42 - "Community 42"
Cohesion: 1.0
Nodes (1): True if .gitignore's results/ rules keep this filename.

### Community 43 - "Community 43"
Cohesion: 1.0
Nodes (1): Print what the run left behind, split into committed vs local-only.

### Community 44 - "Community 44"
Cohesion: 1.0
Nodes (1): Loads YOLOv8x-worldv2 once and exposes inference to p4_core.

### Community 45 - "Community 45"
Cohesion: 1.0
Nodes (1): PGD on the bare model — VERBATIM from run_survey_yolo.py:255-280.      The attac

### Community 46 - "Community 46"
Cohesion: 1.0
Nodes (1): Grad-enabled twin of pc.gaussian_gpu (phase3_common.py:144-165).      pc.gaussia

### Community 47 - "Community 47"
Cohesion: 1.0
Nodes (1): Refuse to run if the BPDA Gaussian differs from the deployed one.

### Community 48 - "Community 48"
Cohesion: 1.0
Nodes (1): The PIL uint8 round-trip the deployed pipeline performs.

### Community 49 - "Community 49"
Cohesion: 1.0
Nodes (1): Forward = the REAL transform. Backward = BPDA.      jpeg / median : non-differen

### Community 50 - "Community 50"
Cohesion: 1.0
Nodes (1): PGD whose per-step gradient is the EOT mean over the defense branches.

### Community 51 - "Community 51"
Cohesion: 1.0
Nodes (1): Holder for the heavy modules, populated by load_runtime().      A shared object

### Community 52 - "Community 52"
Cohesion: 1.0
Nodes (1): Resolve the run id, the output dir and the --smoke-test overrides.

### Community 53 - "Community 53"
Cohesion: 1.0
Nodes (1): Shared preflight. `modules` = [(name, needed_by)], `weights` = [(label, present_

### Community 54 - "Community 54"
Cohesion: 1.0
Nodes (1): The requested solos plus every ensemble member, order-stable.

### Community 55 - "Community 55"
Cohesion: 1.0
Nodes (1): {branch: PIL} for the requested branches.      `pc.apply_all_defenses_gpu` produ

### Community 56 - "Community 56"
Cohesion: 1.0
Nodes (1): The OCR analogue of the detection NMS merge (no boxes to merge).

### Community 57 - "Community 57"
Cohesion: 1.0
Nodes (1): Config-keyed checkpoint whose FILENAME also carries the signature.      Phase 3

### Community 58 - "Community 58"
Cohesion: 1.0
Nodes (1): Run `compute_one(pil, img_id) -> {tag: value}` over `files`.      Returns (elaps

### Community 59 - "Community 59"
Cohesion: 1.0
Nodes (1): Stamp image_id onto a detection list and file it under `tag`.

### Community 60 - "Community 60"
Cohesion: 1.0
Nodes (1): COCO-eval every condition. Bulky dumps -> temp dir; output -> one line.      pyc

### Community 61 - "Community 61"
Cohesion: 1.0
Nodes (1): Assemble ensembles via the Phase-3 NMS merge, then COCO-eval everything.      Re

### Community 62 - "Community 62"
Cohesion: 1.0
Nodes (1): {defense: {value, AP50, delta_vs_measured, kind}} for one condition prefix.

### Community 63 - "Community 63"
Cohesion: 1.0
Nodes (1): Average self-consistency per branch/ensemble, plus the per-image rows.

### Community 64 - "Community 64"
Cohesion: 1.0
Nodes (1): Write summary_<stem>_n<N>_<run_id>.json plus the _latest mirror.

### Community 65 - "Community 65"
Cohesion: 1.0
Nodes (1): Per-track timing table for a multi-track run (clean_control).

### Community 66 - "Community 66"
Cohesion: 1.0
Nodes (1): Compact end-of-run block for the log: baseline + best/worst per track.      The

### Community 67 - "Community 67"
Cohesion: 1.0
Nodes (1): Last thing every entry point calls: list what is safe to share.

## Ambiguous Edges - Review These
- `Phase 2 Variant B — Advanced Denoising` → `DiffPure (dropped — classification only)`  [AMBIGUOUS]
  memory_phase2_archive.md · relation: conceptually_related_to
- `Wang et al., One Object, Multiple Lies: Benchmark for Cross-task Adversarial Attack on Unified VLMs (arXiv:2403.09761)` → `CRAFT Attack (Cross-task Region-based Attack Framework with Token-alignment)`  [AMBIGUOUS]
  docs/Presentation.pdf · relation: cites
- `FGSM epsilon sweep 0.003 / 0.01 / 0.03 (eps=0.003 and eps=0.01 rows are byte-identical - likely a duplicated/unswept eps bug)` → `attacked (no defense): mAP 0.3021 / AP50 0.4143 at eps 0.003-0.01; mAP 0.2721 / AP50 0.3881 at eps 0.03`  [AMBIGUOUS]
  figures/Result_Images/Phase2_Variant_A.jpeg · relation: conceptually_related_to
- `YOLO PGD eps0.03: clean 0.4649 -> attacked 0.0676 (attack damage 0.3974 - the most destructive attack in the set)` → `AMBIGUOUS: these PGD-YOLO figures (0.4649 -> 0.0676 -> 0.4064) differ from the locked v2 numbers cited in memory.md (0.4765 -> 0.0737 -> 0.4128) - likely an earlier phase-3 run, do not mix in the paper`  [AMBIGUOUS]
  figures/Result_Images/Phase3_PGD_yolo.jpeg · relation: conceptually_related_to
- `Defense-cost bars ~0.00061 jpeg / 0.00039 gaussian_blur / 0.00047 median_filter / 0.00040 bit_depth, all annotated '-0.000'` → `Rendering bug: bar annotations float in a huge blank canvas region far above the axes`  [AMBIGUOUS]
  archive/Logs_Extra/results_phase2_variantA/results_plot_v2.png · relation: conceptually_related_to
- `Bit-depth reduction is worst: ~0.00027-0.000285, roughly half the undefended attacked mAP` → `Variant A defended panels (jpeg, gaussian_blur, median_filter, bit_depth) are all visually clean, no artifact visible at display scale`  [AMBIGUOUS]
  archive/Logs_Extra/results_phase2_variantA/visual_comparison_variantA.png · relation: conceptually_related_to
- `Variant B Florence FGSM mAP-vs-Epsilon + Defense-Cost Figure` → `Florence Figures Titled 'Variant 2' with No Model Named (label ambiguity)`  [AMBIGUOUS]
  archive/Logs_Extra/results_phase2_variantB/results_plot_v2.png · relation: references
- `Florence Figures Titled 'Variant 2' with No Model Named (label ambiguity)` → `Variant C Florence FGSM mAP-vs-Epsilon + Defense-Cost Figure`  [AMBIGUOUS]
  archive/Logs_Extra/results_phase2_variantC/results_plot_v2.png · relation: references
- `Variant B YOLOv8x-worldv2 FGSM mAP-vs-Epsilon + Defense-Cost Figure` → `YOLO Clean-Cost Labels Read Negative While Bars Sit Above Baseline (sign unclear)`  [AMBIGUOUS]
  archive/Logs_Extra/results_phase2_variantB_yolo/results_plot.png · relation: references
- `Attacked YOLO mAP 0.59 at eps=0.003 Exceeds Clean 0.511 (anomaly)` → `YOLO Clean-Cost Labels Read Negative While Bars Sit Above Baseline (sign unclear)`  [AMBIGUOUS]
  archive/Logs_Extra/results_phase2_variantB_yolo/results_plot.png · relation: conceptually_related_to
- `Variant Y results plot: cutout-augmented defense bank, FGSM mAP-vs-epsilon + clean cost` → `Rendering defect: Variant Y bar-chart x tick labels overlap and collide, making per-bar attribution partly unreadable`  [AMBIGUOUS]
  archive/Logs_Extra/results_phase2_variantY/results_plot_v2.png · relation: references
- `Variant Z attacked-vs-recovered: horizontal bar ranking of 11 defenses by mAP delta over attacked` → `solo tvm (Variant Z): the only negative delta (bar ends left of the attacked line, ~0.262); its numeric label is occluded by the legend box`  [AMBIGUOUS]
  archive/Logs_Extra/results_phase2_variantZ/attacked_vs_recovered_variantZ.png · relation: references
- `Defense [solo] tvm` → `Solo tvm is the only net-negative defense here (delta label occluded by legend, ~-0.011)`  [AMBIGUOUS]
  results_phase3_florence_fgsm_v2/attacked_vs_recovered_fgsm_eps0.03.png · relation: references
- `Defense [solo] blur_tvm` → `AMBIGUOUS: Florence Patch blur_tvm delta partly hidden by legend (~-0.053)`  [AMBIGUOUS]
  archive/Logs_Extra/results_phase3_florence_patch/attacked_vs_recovered_patch.png · relation: references
- `Defense [solo] blur_tvm` → `AMBIGUOUS: YOLO Patch blur_tvm delta hidden behind legend (bar ends ~0.395, so ~-0.045)`  [AMBIGUOUS]
  archive/Logs_Extra/results_phase3_yolo_patch/attacked_vs_recovered_patch.png · relation: references
- `Defense [solo] jpeg` → `AMBIGUOUS: YOLO PGD solo jpeg delta hidden behind legend (bar ends ~0.352, so ~+0.28)`  [AMBIGUOUS]
  archive/Logs_Extra/results_phase3_yolo_pgd/attacked_vs_recovered_pgd_eps0.03.png · relation: references
- `Florence-2 x Patch — Attacked vs Recovered (v1)` → `AMBIGUOUS: Florence Patch blur_tvm delta partly hidden by legend (~-0.053)`  [AMBIGUOUS]
  archive/Logs_Extra/results_phase3_florence_patch/attacked_vs_recovered_patch.png · relation: references
- `YOLOv8x-worldv2 x PGD eps0.03 — Attacked vs Recovered (v1)` → `AMBIGUOUS: YOLO PGD solo jpeg delta hidden behind legend (bar ends ~0.352, so ~+0.28)`  [AMBIGUOUS]
  archive/Logs_Extra/results_phase3_yolo_pgd/attacked_vs_recovered_pgd_eps0.03.png · relation: references
- `YOLOv8x-worldv2 x Patch — Attacked vs Recovered (v1)` → `AMBIGUOUS: YOLO Patch blur_tvm delta hidden behind legend (bar ends ~0.395, so ~-0.045)`  [AMBIGUOUS]
  archive/Logs_Extra/results_phase3_yolo_patch/attacked_vs_recovered_patch.png · relation: references
- `YOLO PGD v2 attacked mAP 0.0737 (-0.4028, near-total collapse; headline worst case)` → `Archived v1 counterparts: archive/Logs_Extra/results_phase3_yolo_pgd and results_phase3_pgd_yolo (two v1 dirs for one v2 cell)`  [AMBIGUOUS]
  results_phase3_yolo_pgd_v2/attacked_vs_recovered_pgd_eps0.03.png · relation: conceptually_related_to

## Knowledge Gaps
- **378 isolated node(s):** `Phase 3 shared utilities: GPU-accelerated defenses, ensemble NMS merger, checkpo`, `TRUE bilateral filter (joint spatial+range Gaussian) on GPU.      tensors: [N,C,`, `Bit-depth reduction to 2^bits levels per channel (Xu et al. NDSS 2018).      Flo`, `Non-Local Means denoising (skimage). Buades et al. 2005.      h = h_rel * estima`, `Random resize + pad (Xie et al. ICLR 2018). Stochastic defense.      Pass `seed`` (+373 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 28`** (3 nodes): `F2_Optimized.py`, `Run ablation study to analyze each defense component`, `run_ablation_study()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 30`** (1 nodes): `A compute_one that drops a condition must abort the run, not report.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 31`** (1 nodes): `An image that raises must fail the count assert, not be silently dropped.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 32`** (1 nodes): `pycocotools' boilerplate must not reach the committed run.log.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 33`** (1 nodes): `The OCR determinism pass is skippable and actually saves a generate.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 34`** (1 nodes): `The full table must not be echoed into stdout (it is in the .md).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 35`** (1 nodes): `A second run in the same directory must not clobber the first.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 36`** (1 nodes): `Evaluate phase3_common's ensemble constants WITHOUT importing torch.      phase3`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 37`** (1 nodes): `The classifier must agree with the repo's real .gitignore rules.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 38`** (1 nodes): `Smoke-test output must be impossible to commit.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 39`** (1 nodes): `Duplicate a text stream into a file handle.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 40`** (1 nodes): `Tee stdout+stderr into `path` (append) with a session header.      Append, not t`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 41`** (1 nodes): `Where pycocotools' per-condition detection dumps go.      keep=False (default):`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 42`** (1 nodes): `True if .gitignore's results/ rules keep this filename.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 43`** (1 nodes): `Print what the run left behind, split into committed vs local-only.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 44`** (1 nodes): `Loads YOLOv8x-worldv2 once and exposes inference to p4_core.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 45`** (1 nodes): `PGD on the bare model — VERBATIM from run_survey_yolo.py:255-280.      The attac`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 46`** (1 nodes): `Grad-enabled twin of pc.gaussian_gpu (phase3_common.py:144-165).      pc.gaussia`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 47`** (1 nodes): `Refuse to run if the BPDA Gaussian differs from the deployed one.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 48`** (1 nodes): `The PIL uint8 round-trip the deployed pipeline performs.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 49`** (1 nodes): `Forward = the REAL transform. Backward = BPDA.      jpeg / median : non-differen`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 50`** (1 nodes): `PGD whose per-step gradient is the EOT mean over the defense branches.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (1 nodes): `Holder for the heavy modules, populated by load_runtime().      A shared object`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 52`** (1 nodes): `Resolve the run id, the output dir and the --smoke-test overrides.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 53`** (1 nodes): `Shared preflight. `modules` = [(name, needed_by)], `weights` = [(label, present_`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 54`** (1 nodes): `The requested solos plus every ensemble member, order-stable.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 55`** (1 nodes): `{branch: PIL} for the requested branches.      `pc.apply_all_defenses_gpu` produ`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 56`** (1 nodes): `The OCR analogue of the detection NMS merge (no boxes to merge).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 57`** (1 nodes): `Config-keyed checkpoint whose FILENAME also carries the signature.      Phase 3`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 58`** (1 nodes): `Run `compute_one(pil, img_id) -> {tag: value}` over `files`.      Returns (elaps`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 59`** (1 nodes): `Stamp image_id onto a detection list and file it under `tag`.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 60`** (1 nodes): `COCO-eval every condition. Bulky dumps -> temp dir; output -> one line.      pyc`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 61`** (1 nodes): `Assemble ensembles via the Phase-3 NMS merge, then COCO-eval everything.      Re`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 62`** (1 nodes): `{defense: {value, AP50, delta_vs_measured, kind}} for one condition prefix.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 63`** (1 nodes): `Average self-consistency per branch/ensemble, plus the per-image rows.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 64`** (1 nodes): `Write summary_<stem>_n<N>_<run_id>.json plus the _latest mirror.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 65`** (1 nodes): `Per-track timing table for a multi-track run (clean_control).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 66`** (1 nodes): `Compact end-of-run block for the log: baseline + best/worst per track.      The`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 67`** (1 nodes): `Last thing every entry point calls: list what is safe to share.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Phase 2 Variant B — Advanced Denoising` and `DiffPure (dropped — classification only)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Wang et al., One Object, Multiple Lies: Benchmark for Cross-task Adversarial Attack on Unified VLMs (arXiv:2403.09761)` and `CRAFT Attack (Cross-task Region-based Attack Framework with Token-alignment)`?**
  _Edge tagged AMBIGUOUS (relation: cites) - confidence is low._
- **What is the exact relationship between `FGSM epsilon sweep 0.003 / 0.01 / 0.03 (eps=0.003 and eps=0.01 rows are byte-identical - likely a duplicated/unswept eps bug)` and `attacked (no defense): mAP 0.3021 / AP50 0.4143 at eps 0.003-0.01; mAP 0.2721 / AP50 0.3881 at eps 0.03`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `YOLO PGD eps0.03: clean 0.4649 -> attacked 0.0676 (attack damage 0.3974 - the most destructive attack in the set)` and `AMBIGUOUS: these PGD-YOLO figures (0.4649 -> 0.0676 -> 0.4064) differ from the locked v2 numbers cited in memory.md (0.4765 -> 0.0737 -> 0.4128) - likely an earlier phase-3 run, do not mix in the paper`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Defense-cost bars ~0.00061 jpeg / 0.00039 gaussian_blur / 0.00047 median_filter / 0.00040 bit_depth, all annotated '-0.000'` and `Rendering bug: bar annotations float in a huge blank canvas region far above the axes`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Bit-depth reduction is worst: ~0.00027-0.000285, roughly half the undefended attacked mAP` and `Variant A defended panels (jpeg, gaussian_blur, median_filter, bit_depth) are all visually clean, no artifact visible at display scale`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Variant B Florence FGSM mAP-vs-Epsilon + Defense-Cost Figure` and `Florence Figures Titled 'Variant 2' with No Model Named (label ambiguity)`?**
  _Edge tagged AMBIGUOUS (relation: references) - confidence is low._