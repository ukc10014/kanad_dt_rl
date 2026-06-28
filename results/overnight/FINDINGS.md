# Overnight batch — FINDINGS (auto-extracted Wed Jun 24 19:16:22 BST 2026)

Raw dump; fold into results.md. Job status table follows, then per-job results.

## Status table
# Overnight batch — started Wed Jun 24 17:45:28 BST 2026

| job | status | reason | wall | log |
|---|---|---|---|---|
| mechanism-ladder | DONE | ok | 48m | results/run_mech_14b.log |
| self-snapshot-hysteresis | DONE | ok | 25m | results/run_hyst_onebox_seedref.log |
| lora-r8-kl0.02 | SMOKE-FAIL | see results/run_lora_r8_kl0.02.log.smoke | - | - |
| lora-r32-kl0.02 | SMOKE-FAIL | see results/run_lora_r32_kl0.02.log.smoke | - | - |

_done Wed Jun 24 19:15:23 BST 2026_

## (1) Mechanism-credibility ladder (14B)
```
tag,onebox_rate,margin_level,margin_hi_lo,cred_gap_adj_slope,cred_gap_adj_ci,cred_gap_at_0.5,cred_saturated,cred_gap_adj_slope_prediction,cred_gap_adj_ci_prediction,cred_gap_at_0.5_prediction,cred_saturated_prediction,level,label,order
mech_m0_14b,0.5875,3.2921874700398868,0.44999995874451626,0.21042786347455858,"[0.11702689851329662, 0.31120122844978015]",0.189,False,0.09250152443301807,"[0.04283336935852727, 0.14325178979965436]",0.7433447033557534,False,m0,statistical,0
mech_m0pad_14b,0.5875,2.7468749721303913,1.000000040872283,0.17479235010818742,"[0.08030711244503388, 0.27392593529699166]",0.24400000000000005,False,0.08805103327511835,"[0.040875616826187434, 0.13665755785134526]",0.7358671595979259,False,m0pad,stat+pad,1
mech_m1_14b,0.6125,3.5484374925888944,-0.599999924103146,0.18519752914078322,"[0.11354230648426059, 0.26100596775319335]",0.23500000000000001,False,0.07113955139930409,"[0.04054093042143586, 0.10303788025087661]",0.790072842592268,False,m1,indiv-model,2
mech_m2_14b,0.625,3.6671874678775467,1.3624999036158334,0.30310427863474576,"[0.24094092447825796, 0.35756787883018093]",0.095,False,0.09379587487115408,"[0.06089873681329662, 0.12619507764693186]",0.8112487909735803,False,m2,process-scan,3
mech_m3_14b,0.6875,3.7890624974757445,0.30000007553405794,0.3095588748516787,"[0.21821503978502138, 0.3897023975710199]",0.13,False,0.07448502534640394,"[0.04171968529312126, 0.10776265614988903]",0.8139476585429506,False,m3,exact-copy,4
```
Read: does one-box rate / credence gap_adj rise m0(statistical)->m3(exact-copy)?
Plots: results/credence/mechanism_signature.png

## (2) Self-snapshot one-box-basin probe (3B)
First + last eval lines (did K stay ~1 [bistable] or decay to 0 [two-box only]?):
```
[evidential_modelpred] step=0 eval mean_K=1.000  K@p=0.80->1.00  K@p=0.80->1.00  slope(hi-lo)=+0.00
[evidential_modelpred] step=1 train reward=100.00 K=1.00 invalid=0.00 gen_len=2.0 p_model=1.000
[evidential_modelpred] step=15 train reward=100.00 K=1.00 invalid=0.00 gen_len=2.0 p_model=1.000
...
[evidential_modelpred] step=150 train reward=100.00 K=1.00 invalid=0.00 gen_len=2.0 p_model=1.000
[evidential_modelpred] step=150 eval mean_K=1.000  K@p=0.80->1.00  K@p=0.80->1.00  slope(hi-lo)=+0.00
[evidential_modelpred] step=150 eval mean_K=1.000  K@p=0.80->1.00  K@p=0.80->1.00  slope(hi-lo)=+0.00
```
Endpoint logprob slope (ep_hyst_onebox_seedref):
```
p,p_non_cdt_mean,p_non_cdt_se,margin_mean,margin_se,k_rate_argmax,n
0.5,0.9999999995435171,1.7761869231201038e-10,24.8,0.8504255900491978,1.0,20
0.6,0.9999999999240587,4.052494685134634e-11,26.75625,0.7228345041534312,1.0,20
0.7,0.9999999999182304,2.8128463018297492e-11,25.40625,0.61797865265038,1.0,20
0.75,0.9999999996886546,1.6739659366730785e-10,25.11875,0.7984089338540086,1.0,20
0.8,0.9999999999028868,5.525446193798077e-11,25.70625,0.6720620149760618,1.0,20
0.85,0.9999999997020842,1.7714096706991348e-10,25.4875,0.7849792571848803,1.0,20
0.9,0.9999999999202579,5.560423673907772e-11,25.775,0.4951740789066534,1.0,20
0.99,0.9999999998898312,4.6594822831746096e-11,25.10625,0.6369434490100687,1.0,20
```

## (3) LoRA extras (if reached)

_auto-extracted Wed Jun 24 19:16:22 BST 2026 — interpretation pending (assistant or human)._
