## Table 4 - main results (mean ± SD over seeds)
| model       |   n_seeds | Macro-F1 (test)   | Accuracy (test)   | Balanced acc (test)   | Macro-F1 (val)   | Subject-bootstrap 95% CI (mean of seeds)   |
|:------------|----------:|:------------------|:------------------|:----------------------|:-----------------|:-------------------------------------------|
| cnn_bilstm  |         3 | 0.931 ± 0.003     | 0.931 ± 0.004     | 0.932 ± 0.004         | 0.987 ± 0.000    | [0.867, 0.978]                             |
| resnet1d    |         3 | 0.935 ± 0.008     | 0.935 ± 0.008     | 0.935 ± 0.008         | 0.990 ± 0.002    | [0.857, 0.984]                             |
| rf          |         3 | 0.912 ± 0.003     | 0.913 ± 0.003     | 0.912 ± 0.003         | 0.939 ± 0.002    | [0.871, 0.951]                             |
| transformer |         3 | 0.914 ± 0.009     | 0.914 ± 0.009     | 0.915 ± 0.009         | 0.982 ± 0.004    | [0.842, 0.967]                             |

## Per-class F1 (test)
| model       | Walking       | Upstairs      | Downstairs    | Sitting       | Standing      | Laying        |
|:------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
| cnn_bilstm  | 0.994 ± 0.001 | 0.960 ± 0.005 | 0.966 ± 0.002 | 0.817 ± 0.010 | 0.852 ± 0.015 | 0.998 ± 0.003 |
| resnet1d    | 0.972 ± 0.007 | 0.951 ± 0.021 | 0.948 ± 0.013 | 0.850 ± 0.016 | 0.888 ± 0.010 | 0.998 ± 0.003 |
| rf          | 0.952 ± 0.001 | 0.922 ± 0.005 | 0.906 ± 0.006 | 0.838 ± 0.007 | 0.858 ± 0.005 | 1.000 ± 0.000 |
| transformer | 0.970 ± 0.015 | 0.939 ± 0.011 | 0.937 ± 0.006 | 0.810 ± 0.012 | 0.834 ± 0.032 | 0.995 ± 0.000 |

## Table 7 - computational cost
| model       |   params |   size_MB |   train_s |   infer_ms | hardware                                                                                           |
|:------------|---------:|----------:|----------:|-----------:|:---------------------------------------------------------------------------------------------------|
| cnn_bilstm  |   144518 |     0.586 |    47.805 |      1.086 | GPU: NVIDIA GeForce GTX 1650 | CPU: AMD64 Family 23 Model 96 Stepping 1, AuthenticAMD (12 threads) |
| resnet1d    |   183971 |     0.764 |    76.916 |      2.911 | GPU: NVIDIA GeForce GTX 1650 | CPU: AMD64 Family 23 Model 96 Stepping 1, AuthenticAMD (12 threads) |
| rf          |   109280 |    12.348 |     1.266 |     52.321 | GPU: NVIDIA GeForce GTX 1650 | CPU: AMD64 Family 23 Model 96 Stepping 1, AuthenticAMD (12 threads) |
| transformer |   105670 |     0.457 |    49.927 |      5.518 | GPU: NVIDIA GeForce GTX 1650 | CPU: AMD64 Family 23 Model 96 Stepping 1, AuthenticAMD (12 threads) |

## Table 5 - ablation
| model       | component removed   | Macro-F1 (test)   |   change vs full | full model    |   params |
|:------------|:--------------------|:------------------|-----------------:|:--------------|---------:|
| cnn_bilstm  | no_aug              | 0.930 ± 0.007     |           -0.001 | 0.931 ± 0.003 |   144518 |
| cnn_bilstm  | no_bilstm           | 0.930 ± 0.008     |           -0.001 | 0.931 ± 0.003 |    45190 |
| cnn_bilstm  | no_gyro             | 0.919 ± 0.006     |           -0.012 | 0.931 ± 0.003 |   143558 |
| resnet1d    | no_attn_pool        | 0.930 ± 0.007     |           -0.005 | 0.935 ± 0.008 |   183842 |
| resnet1d    | no_aug              | 0.918 ± 0.008     |           -0.017 | 0.935 ± 0.008 |   183971 |
| resnet1d    | no_gyro             | 0.914 ± 0.003     |           -0.02  | 0.935 ± 0.008 |   183299 |
| resnet1d    | no_se               | 0.928 ± 0.009     |           -0.007 | 0.935 ± 0.008 |   178343 |
| rf          | no_gyro             | 0.875 ± 0.003     |           -0.037 | 0.912 ± 0.003 |   113240 |
| transformer | no_attention        | 0.924 ± 0.012     |            0.01  | 0.914 ± 0.009 |    55750 |
| transformer | no_aug              | 0.920 ± 0.008     |            0.006 | 0.914 ± 0.009 |   105670 |
| transformer | no_gyro             | 0.905 ± 0.008     |           -0.009 | 0.914 ± 0.009 |   104134 |

## sweep_aug (validation macro-F1)
| model      | variant                        | val_f1        |   seeds |
|:-----------|:-------------------------------|:--------------|--------:|
| cnn_bilstm | names=[]                       | 0.987 ± 0.001 |       3 |
| cnn_bilstm | names=jitter                   | 0.987 ± 0.001 |       3 |
| cnn_bilstm | names=jitter-scaling-time_warp | 0.987 ± 0.000 |       3 |
| cnn_bilstm | names=time_warp                | 0.987 ± 0.001 |       3 |
| resnet1d   | names=[]                       | 0.986 ± 0.004 |       3 |
| resnet1d   | names=jitter                   | 0.991 ± 0.004 |       3 |
| resnet1d   | names=jitter-scaling-time_warp | 0.990 ± 0.002 |       3 |
| resnet1d   | names=time_warp                | 0.989 ± 0.005 |       3 |

## sweep_ssl (validation macro-F1)
| model       | variant                                                      | val_f1        |   seeds |
|:------------|:-------------------------------------------------------------|:--------------|--------:|
| transformer | pretrained=checkpoints-transformer_ssl.pt_label_fraction=0.1 | 0.972 ± 0.003 |       3 |
| transformer | pretrained=checkpoints-transformer_ssl.pt_label_fraction=1.0 | 0.984 ± 0.005 |       3 |
| transformer | pretrained=null_label_fraction=0.1                           | 0.968 ± 0.015 |       3 |
| transformer | pretrained=null_label_fraction=1                             | 0.982 ± 0.004 |       3 |
| transformer | pretrained=null_label_fraction=1.0                           | 0.982 ± 0.004 |       3 |

## sweep_window (validation macro-F1)
| model       | variant        | val_f1        |   seeds |
|:------------|:---------------|:--------------|--------:|
| cnn_bilstm  | window_len=128 | 0.987 ± 0.000 |       3 |
| cnn_bilstm  | window_len=64  | 0.987 ± 0.001 |       3 |
| cnn_bilstm  | window_len=96  | 0.987 ± 0.001 |       3 |
| resnet1d    | window_len=128 | 0.990 ± 0.002 |       3 |
| resnet1d    | window_len=64  | 0.986 ± 0.002 |       3 |
| resnet1d    | window_len=96  | 0.990 ± 0.003 |       3 |
| transformer | window_len=128 | 0.982 ± 0.004 |       3 |
| transformer | window_len=64  | 0.974 ± 0.004 |       3 |
| transformer | window_len=96  | 0.980 ± 0.003 |       3 |

## Tuning (seed 0, validation)
| model       | variant        |   val_f1 |
|:------------|:---------------|---------:|
| cnn_bilstm  | lr=0.0003      | 0.98726  |
| cnn_bilstm  | lr=0.001       | 0.983665 |
| cnn_bilstm  | lr=0.003       | 0.98375  |
| resnet1d    | lr=0.0003      | 0.984986 |
| resnet1d    | lr=0.001       | 0.987039 |
| resnet1d    | lr=0.003       | 0.992966 |
| rf          | max_depth=10   | 0.933176 |
| rf          | max_depth=20   | 0.937563 |
| rf          | max_depth=None | 0.93524  |
| transformer | lr=0.0003      | 0.985717 |
| transformer | lr=0.001       | 0.982465 |
| transformer | lr=0.003       | 0.986088 |
