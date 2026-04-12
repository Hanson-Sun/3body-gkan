import pytest
import torch

from nbody_gkan.training.kan_trainer import KANTrainer


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def loss(self, batch, augment=False, augmentation=3.0, lamb=0.0, square=False):
        del batch, augment, augmentation, lamb, square
        return (self.weight - 1.0).pow(2)


def _make_trainer(
    *,
    adam_warmup_epochs: int = 0,
    lamb_schedule: list[float] | None = None,
    checkpoint_dir,
) -> KANTrainer:
    return KANTrainer(
        model=_ToyModel(),
        train_loader=[torch.tensor(0.0)],
        val_loader=None,
        adam_warmup_epochs=adam_warmup_epochs,
        adam_lr=1e-3,
        lbfgs_impl="torch",
        lbfgs_max_iter=2,
        grid_update_freq=0,
        max_grid_updates=0,
        lamb_schedule=lamb_schedule,
        device=torch.device("cpu"),
        checkpoint_dir=checkpoint_dir,
    )


def test_two_phase_switches_to_lbfgs_after_warmup(tmp_path):
    trainer = _make_trainer(adam_warmup_epochs=3, checkpoint_dir=tmp_path)
    assert trainer.current_phase == "Adam"

    for epoch in range(3):
        trainer._on_epoch_start(epoch)
        assert trainer.current_phase == "Adam"

    trainer._on_epoch_start(3)
    assert trainer.current_phase == "LBFGS"


def test_lamb_schedule_applies_uniform_epoch_ramp(tmp_path):
    trainer = _make_trainer(
        adam_warmup_epochs=0,
        lamb_schedule=[1e-4, 1e-3, 5e-3, 1e-2],
        checkpoint_dir=tmp_path,
    )
    trainer._scheduled_total_epochs = 8

    expected = [1e-4, 1e-4, 1e-3, 1e-3, 5e-3, 5e-3, 1e-2, 1e-2]
    for epoch, lamb in enumerate(expected):
        trainer._apply_lamb_schedule(epoch)
        assert trainer.lamb == pytest.approx(lamb)


@pytest.mark.parametrize("bad_schedule", ["[1e-4, 1e-3]", [1e-4, -1e-3], [1e-4, "bad"]])
def test_lamb_schedule_rejects_invalid_values(tmp_path, bad_schedule):
    with pytest.raises(ValueError):
        _make_trainer(
            adam_warmup_epochs=0,
            lamb_schedule=bad_schedule,
            checkpoint_dir=tmp_path,
        )
