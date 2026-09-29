"""Train a behavior-cloning policy on the collected demonstrations.

    uv run python scripts/train.py policy=diffusion
    uv run python scripts/train.py policy=flow seed=1 hydra.run.dir=outputs/flow_s1
"""

import os

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from behavior_cloning.checkpoint import save_checkpoint
from behavior_cloning.data import NavigationDataset
from behavior_cloning.logger import Logger
from behavior_cloning.policy import ChunkedPolicy
from behavior_cloning.training import EMA, make_optimizer, make_scheduler, resolve_device, set_seed

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    output_dir = HydraConfig.get().runtime.output_dir
    checkpoint_path = os.path.join(output_dir, "policy.pt")

    dataset = NavigationDataset(
        os.path.join(REPO_ROOT, cfg.task.data_path), cfg.task.obs_horizon, cfg.task.pred_horizon
    )
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True)
    print(f"{dataset}  head={cfg.policy.head}  seed={cfg.seed}  device={device}")

    policy = ChunkedPolicy.from_config(
        cfg, dataset.obs_dim, dataset.action_dim, dataset.obs_normalizer, dataset.action_normalizer
    ).to(device)
    ema = EMA(policy, cfg.train.ema_decay)
    optimizer = make_optimizer(policy, cfg.train)
    scheduler = make_scheduler(
        optimizer, cfg.train.num_epochs * len(loader), cfg.train.lr_warmup_steps
    )
    logger = Logger(
        os.path.join(output_dir, "train_log.csv"), total=cfg.train.num_epochs, unit="epoch"
    )

    for epoch in range(1, cfg.train.num_epochs + 1):
        running = 0.0
        for batch in loader:
            loss = policy.loss({key: value.to(device) for key, value in batch.items()})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            ema.update(policy)
            running += loss.item()

        last = epoch == cfg.train.num_epochs
        if epoch % cfg.train.log_interval == 0 or last:
            logger.log(epoch, loss=running / len(loader), lr=scheduler.get_last_lr()[0])
        if epoch % cfg.train.save_interval == 0 or last:
            save_checkpoint(checkpoint_path, ema.model, cfg, epoch=epoch)

    logger.close()
    print(f"checkpoint {checkpoint_path}")


if __name__ == "__main__":
    main()
