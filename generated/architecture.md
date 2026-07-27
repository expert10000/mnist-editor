# Enhanced Five-Block MNIST V1

Version: 0.1.0

## Summary

- Parameters: 1.35M
- Forward FLOPs: 260.22M
- Residual paths: 5
- Branches: 20
- Auxiliary heads: 1
- Embedding dimension: 128

## Main Path

| Node | Operator | Input | Output |
|---|---|---|---|
| MNIST image | input | [B, 1, 28, 28] | [B, 1, 28, 28] |
| Stem | conv_bn_gelu | [B, 1, 28, 28] | [B, 32, 28, 28] |
| Block 1 | multi_branch_residual | [B, 32, 28, 28] | [B, 48, 28, 28] |
| Block 2 | multi_branch_residual | [B, 48, 28, 28] | [B, 80, 14, 14] |
| Block 3 | multi_branch_residual | [B, 80, 14, 14] | [B, 96, 14, 14] |
| Block 4 | multi_branch_residual | [B, 96, 14, 14] | [B, 144, 7, 7] |
| Block 5 | multi_branch_residual | [B, 144, 7, 7] | [B, 160, 7, 7] |
| GAP + GMP | pooling_fusion | [B, 160, 7, 7] | [B, 320] |
| Feature head | feature_head | [B, 320] | [B, 128] |
| Classifier | classifier | [B, 128] | [B, 10] |

## Auxiliary Heads

| Head | Source | Input | Output |
|---|---|---|---|
| Auxiliary head | block3 | [B, 96, 14, 14] | [B, 10] |

## Validation

No validation errors.
