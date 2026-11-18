## headline

| Model | Custom 2,000 (2 seeds) | Custom 200 (seed 0) | PDEBench 200 (2 seeds) |
| --- | --- | --- | --- |
| U-Net | 0.0030780826 | 0.023217069 | 0.14207152 |
| FNO | 0.0033076495 | 0.032449453 | 0.27232642 |
| Soft FNO | 0.003807557 | 0.032493951 | 0.35230659 |
| Constrained FNO | 0.00077196435 | 0.0032311664 | 0.22136138 |

## custom2000

| Model | 100-step mean | 100-step seed SD | 200-step mean | t > 1 mean | 200-step failures |
| --- | --- | --- | --- | --- | --- |
| U-Net | 0.0030780826 | 0.000799512 | 0.0050067053 | 0.0069353279 | 0.00% |
| FNO | 0.0033076495 | 0.000464465 | 0.0052880611 | 0.0072684726 | 0.00% |
| Soft FNO | 0.003807557 | 0.000183787 | 0.0062691233 | 0.0087306897 | 0.00% |
| Constrained FNO | 0.00077196435 | 5.07065e-05 | 0.00085789531 | 0.00094382628 | 0.00% |

## custom200

| Model | 100-step mean | 100-step seed SD | 200-step mean | t > 1 mean | 200-step failures |
| --- | --- | --- | --- | --- | --- |
| U-Net | 0.023217069 | Not estimable (one seed) | 0.035115362 | 0.047013655 | 0.00% |
| FNO | 0.032449453 | Not estimable (one seed) | 0.050429698 | 0.068409944 | 0.00% |
| Soft FNO | 0.032493951 | Not estimable (one seed) | 0.050301331 | 0.068108711 | 0.00% |
| Constrained FNO | 0.0032311664 | Not estimable (one seed) | 0.0042608708 | 0.0052905752 | 0.00% |

## pdebench

| Model | 100-step mean | 100-step seed SD | 200-step mean | t > 1 mean | 200-step failures |
| --- | --- | --- | --- | --- | --- |
| U-Net | 0.14207152 | 0.00568777 | 0.19750082 | 0.25293013 | 0.00% |
| FNO | 0.27232642 | 0.00552291 | 0.37006495 | 0.46780349 | 0.00% |
| Soft FNO | 0.35230659 | 0.000845943 | 0.47492117 | 0.59753576 | 0.00% |
| Constrained FNO | 0.22136138 | 0.0255744 | Unavailable: failed rollout(s) | Unavailable: failed rollout(s) | 0.20% |

## lowdata

| Model | 2,000 trajectories, seed 0 | 200 trajectories, seed 0 | Error multiplier |
| --- | --- | --- | --- |
| U-Net | 0.0036434232 | 0.023217069 | 6.37× |
| FNO | 0.0029792234 | 0.032449453 | 10.89× |
| Soft FNO | 0.0039375139 | 0.032493951 | 8.25× |
| Constrained FNO | 0.00080781928 | 0.0032311664 | 4.00× |

## physics

| Study | Model | Normalized maximum mass drift | Energy error | Spectral residual RMS (time mean) |
| --- | --- | --- | --- | --- |
| Custom 2,000 · two seeds | U-Net | 0.00147163 | 0.000196805 | 0.184268 |
| Custom 2,000 · two seeds | FNO | 0.00145205 | 0.000240231 | 0.0204472 |
| Custom 2,000 · two seeds | Soft FNO | 0.00197374 | 0.000228934 | 0.0137705 |
| Custom 2,000 · two seeds | Constrained FNO | 2.78613e-07 | 3.23721e-05 | 0.0102258 |
| Custom 200 · seed 0 | U-Net | 0.00853772 | 0.00112817 | 0.938722 |
| Custom 200 · seed 0 | FNO | 0.0117092 | 0.00185636 | 0.0513052 |
| Custom 200 · seed 0 | Soft FNO | 0.0116048 | 0.00186038 | 0.0458821 |
| Custom 200 · seed 0 | Constrained FNO | 2.72578e-07 | 0.000156876 | 0.015383 |
| PDEBench 200 · two seeds | U-Net | 0.0479081 | 0.0126356 | 3.41557 |
| PDEBench 200 · two seeds | FNO | 0.104653 | 0.0335552 | 3.3728 |
| PDEBench 200 · two seeds | Soft FNO | 0.129594 | 0.0428504 | 3.8793 |
| PDEBench 200 · two seeds | Constrained FNO | 3.41178e-07 | 0.272742 | 132.249 |

## effects

| Study | Comparison | 100-step error difference | Paired bootstrap 95% interval |
| --- | --- | --- | --- |
| Custom 2,000 · two seeds | physics-soft minus fno | 0.000499907 | [-7.01143e-05, 0.00104779] |
| Custom 2,000 · two seeds | physics-constrained minus fno | -0.00253569 | [-0.00299983, -0.00209003] |
| Custom 2,000 · two seeds | physics-constrained minus physics-soft | -0.00303559 | [-0.00324325, -0.00284713] |
| Custom 200 · seed 0 | physics-soft minus fno | 4.44987e-05 | [-4.98002e-05, 0.000140457] |
| Custom 200 · seed 0 | physics-constrained minus fno | -0.0292183 | [-0.0311365, -0.0274673] |
| Custom 200 · seed 0 | physics-constrained minus physics-soft | -0.0292628 | [-0.0311925, -0.0274943] |
| PDEBench 200 · two seeds | physics-soft minus fno | 0.0799802 | [0.0641513, 0.0946275] |
| PDEBench 200 · two seeds | physics-constrained minus fno | -0.050965 | [-0.08213, -0.00975695] |
| PDEBench 200 · two seeds | physics-constrained minus physics-soft | -0.130945 | [-0.166994, -0.0921546] |

## resolution

| Study | Model | Execution | N=256 | N=512 | N=1024 |
| --- | --- | --- | --- | --- | --- |
| Custom 2,000 · two seeds | U-Net | native | 0.00307808 | 0.550089 | 0.821085 |
| Custom 2,000 · two seeds | U-Net | cubic | 0.00307808 | 0.00307701 | 0.00307698 |
| Custom 2,000 · two seeds | FNO | native | 0.00330765 | 0.0033083 | 0.00330821 |
| Custom 2,000 · two seeds | Soft FNO | native | 0.00380756 | 0.00380813 | 0.00380797 |
| Custom 2,000 · two seeds | Constrained FNO | native | 0.000771964 | 0.000771906 | 0.00077179 |
| PDEBench 200 · two seeds | U-Net | native | 0.142072 | 0.604042 | 0.840951 |
| PDEBench 200 · two seeds | U-Net | cubic | 0.142072 | 0.141849 | 0.141933 |
| PDEBench 200 · two seeds | FNO | native | 0.272326 | 0.272707 | 0.272761 |
| PDEBench 200 · two seeds | Soft FNO | native | 0.352307 | 0.352335 | 0.352368 |
| PDEBench 200 · two seeds | Constrained FNO | native | 0.221361 | 0.221825 | 0.221804 |

## cost

| Model | Parameters (real degrees of freedom) | Custom 2,000 minutes | Custom 200 minutes | PDEBench 200 minutes |
| --- | --- | --- | --- | --- |
| U-Net | 671457 | 91.36 | 11.09 | 11.41 |
| FNO | 671460 | 86.02 | 8.71 | 8.09 |
| Soft FNO | 671460 | 78.43 | 8.26 | 8.30 |
| Constrained FNO | 671460 | 116.70 | 13.01 | 13.43 |

## controls

| Study | N | Reference spectral residual | Reference FD residual | Oracle cubic error |
| --- | --- | --- | --- | --- |
| Custom 2,000 · two seeds | 256 | 0.0105096 | 0.0138636 | 0 |
| Custom 2,000 · two seeds | 512 | 0.0105197 | 0.0111389 | 2.90557e-07 |
| Custom 2,000 · two seeds | 1024 | 0.0106072 | 0.0106353 | 2.69401e-07 |
| PDEBench 200 · two seeds | 256 | 3.38834 | 2.67075 | 0 |
| PDEBench 200 · two seeds | 512 | 3.76181 | 3.14382 | 0.0101087 |
| PDEBench 200 · two seeds | 1024 | 3.8012 | 3.55971 | 0.0103565 |

## failures

| Test trajectory | Seed | First failed step | First failed time | 100-step mean error |
| --- | --- | --- | --- | --- |
| test-09012 | 0 | 170 | 1.70 | 0.461127 |
| test-09652 | 0 | 116 | 1.16 | 5.80429 |
