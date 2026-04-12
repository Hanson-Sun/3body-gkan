import torch

from nbody_gkan.training.kan_trainer import KANTrainer


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def loss(self, batch, augment=False, augmentation=3.0, lamb=0.0, square=False):
        del batch, augment, augmentation, lamb, square
        return (self.weight - 1.0).pow(2)


def _make_trainer() -> KANTrainer:
    return KANTrainer(
        model=_ToyModel(),
        train_loader=[torch.tensor(0.0)],
        val_loader=None,
        optimizer_mode="alternating",
        alternating_adam_epochs=3,
        lbfgs_rise_tol=0.0,
        adam_lr=1e-3,
        lbfgs_impl="torch",
        lbfgs_max_iter=2,
        grid_update_freq=0,
        max_grid_updates=0,
        device=torch.device("cpu"),
        checkpoint_dir=None,
    )


def _run_initial_adam_window(trainer: KANTrainer):
    trainer._on_epoch_end(0, train_loss=1.0, val_loss=0.0)
    trainer._on_epoch_end(1, train_loss=0.9, val_loss=0.0)
    trainer._on_epoch_end(2, train_loss=0.8, val_loss=0.0)


def test_alternating_switches_to_adam_when_lbfgs_loss_rises():
    trainer = _make_trainer()
    assert trainer.current_phase == "Adam"

    _run_initial_adam_window(trainer)
    assert trainer.current_phase == "LBFGS"

    trainer._on_epoch_end(3, train_loss=0.50, val_loss=0.0)
    assert trainer.current_phase == "LBFGS"

    trainer._on_epoch_end(4, train_loss=0.52, val_loss=0.0)
    assert trainer.current_phase == "Adam"


def test_alternating_returns_to_lbfgs_after_adam_window():
    trainer = _make_trainer()
    _run_initial_adam_window(trainer)

    trainer._on_epoch_end(3, train_loss=0.50, val_loss=0.0)
    trainer._on_epoch_end(4, train_loss=0.52, val_loss=0.0)
    assert trainer.current_phase == "Adam"

    trainer._on_epoch_end(5, train_loss=0.45, val_loss=0.0)
    trainer._on_epoch_end(6, train_loss=0.44, val_loss=0.0)
    assert trainer.current_phase == "Adam"

    trainer._on_epoch_end(7, train_loss=0.43, val_loss=0.0)
    assert trainer.current_phase == "LBFGS"


def test_alternating_scheduler_steps_only_in_adam_phase():
    trainer = _make_trainer()
    assert trainer._should_step_scheduler() is True

    _run_initial_adam_window(trainer)
    assert trainer.current_phase == "LBFGS"
    assert trainer._should_step_scheduler() is False

    trainer._on_epoch_end(3, train_loss=0.50, val_loss=0.0)
    trainer._on_epoch_end(4, train_loss=0.52, val_loss=0.0)
    assert trainer.current_phase == "Adam"
    assert trainer._should_step_scheduler() is True


def test_alternating_clears_adam_state_on_lbfgs_reentry():
    trainer = _make_trainer()
    param = next(trainer.model.parameters())
    trainer._adam_optimizer.state[param] = {
        "step": torch.tensor(3),
        "exp_avg": torch.zeros_like(param),
        "exp_avg_sq": torch.zeros_like(param),
    }
    assert len(trainer._adam_optimizer.state) > 0

    _run_initial_adam_window(trainer)
    assert trainer.current_phase == "LBFGS"

    trainer._on_epoch_end(3, train_loss=0.50, val_loss=0.0)
    trainer._on_epoch_end(4, train_loss=0.52, val_loss=0.0)
    assert trainer.current_phase == "Adam"
    assert len(trainer._adam_optimizer.state) == 0