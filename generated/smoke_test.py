import torch

from model import GeneratedMNISTModel


def smoke_test() -> None:
    model = GeneratedMNISTModel()
    images = torch.randn(8, 1, 28, 28)
    embedding, logits, auxiliary_logits = model(images, return_features=True, return_auxiliary=True)

    assert embedding.shape == (8, 128)
    assert logits.shape == (8, 10)
    assert auxiliary_logits.shape == (8, 10)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(embedding).all()

    loss = logits.mean()
    if auxiliary_logits is not None:
        loss = loss + 0.1 * auxiliary_logits.mean()
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)


if __name__ == "__main__":
    smoke_test()
