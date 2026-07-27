import torch
from torch import nn

from blocks import AuxiliaryClassifier, ConvBnGelu, FeatureHead, MultiBranchResidual, PoolingFusion


class GeneratedMNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = ConvBnGelu(1, 32, 3, 1, 1)
        self.block1 = MultiBranchResidual(32, 48, stride=1, branch_count=4, use_se=False, se_reduction=8, drop_path=0)
        self.block2 = MultiBranchResidual(48, 80, stride=2, branch_count=4, use_se=False, se_reduction=8, drop_path=0.01)
        self.block3 = MultiBranchResidual(80, 96, stride=1, branch_count=4, use_se=True, se_reduction=8, drop_path=0.02)
        self.block4 = MultiBranchResidual(96, 144, stride=2, branch_count=4, use_se=True, se_reduction=8, drop_path=0.03)
        self.block5 = MultiBranchResidual(144, 160, stride=1, branch_count=4, use_se=True, se_reduction=8, drop_path=0.05)
        self.pooling = PoolingFusion(mode="gap_gmp")
        self.feature_head = FeatureHead(320, hidden_features=192, embedding_features=128, dropout=0.2)
        self.classifier = nn.Linear(128, 10)
        self.auxiliary_head = AuxiliaryClassifier(96, classes=10)

    def forward(self, x, return_features=False, return_auxiliary=False):
        auxiliary_logits = None
        embedding = None
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        auxiliary_logits = self.auxiliary_head(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.pooling(x)
        x = self.feature_head(x)
        embedding = x
        x = self.classifier(x)
        logits = x
        if return_features and return_auxiliary:
            return embedding, logits, auxiliary_logits
        if return_features:
            return embedding, logits
        if return_auxiliary:
            return logits, auxiliary_logits
        return logits
