from dataclasses import asdict
from tianshou.data.stats import EpochStats
from tianshou.trainer.base import BaseTrainer
from tianshou.utils import DummyTqdm
from tianshou.utils.net.common import Net
from tqdm import tqdm as _tqdm


def _setup():  # 将项目根节点加入 sys.path
    import sys
    from pathlib import Path

    __FILE = Path(__file__)
    ROOT = __FILE.parents[1]  # /../..
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    return ROOT


ROOT_DIR = _setup()


class Trainer(BaseTrainer):
    pass

    def run(self, reset_prior_to_run: bool = True):
        rst =super().run(reset_prior_to_run=reset_prior_to_run)
        return rst

    def reset(self, reset_collectors: bool = True, reset_buffer: bool = False) -> None:
        return super().reset(reset_collectors, reset_buffer)

    def __next__(self) -> EpochStats:
        BaseTrainer.__next__ # ref
        self.epoch += 1
        self.iter_num += 1

        if self.iter_num > 1:
            # iterator exhaustion check
            if self.epoch > self.max_epoch:
                raise StopIteration

            # exit flag 1, when stop_fn succeeds in train_step or test_step
            if self.stop_fn_flag:
                raise StopIteration

        progress = _tqdm if self.show_progress else DummyTqdm

        # perform n step_per_epoch
        with progress(
            total=self.step_per_epoch, desc=f"Epoch #{self.epoch}", **tqdm_config
        ) as t:
            train_stat: CollectStatsBase
            while t.n < t.total and not self.stop_fn_flag:
                train_stat, update_stat, self.stop_fn_flag = self.training_step()

                if isinstance(train_stat, CollectStats):
                    pbar_data_dict = {
                        "env_step": str(self.env_step),
                        "rew": f"{self.last_rew:.2f}",
                        "len": str(int(self.last_len)),
                        "n/ep": str(train_stat.n_collected_episodes),
                        "n/st": str(train_stat.n_collected_steps),
                    }
                    t.update(train_stat.n_collected_steps)
                else:
                    pbar_data_dict = {}
                    t.update()

                pbar_data_dict = set_numerical_fields_to_precision(pbar_data_dict)
                pbar_data_dict["gradient_step"] = str(self._gradient_step)
                t.set_postfix(**pbar_data_dict)

                if self.stop_fn_flag:
                    break

            if t.n <= t.total and not self.stop_fn_flag:
                t.update()

        # for offline RL
        if self.train_collector is None:
            assert self.buffer is not None
            batch_size = self.batch_size or len(self.buffer)
            self.env_step = self._gradient_step * batch_size

        test_stat = None
        if not self.stop_fn_flag:
            self.logger.save_data(
                self.epoch,
                self.env_step,
                self._gradient_step,
                self.save_checkpoint_fn,
            )
            # test
            if self.test_collector is not None:
                test_stat, self.stop_fn_flag = self.test_step()

        info_stat = gather_info(
            start_time=self.start_time,
            policy_update_time=self.policy_update_time,
            gradient_step=self._gradient_step,
            best_reward=self.best_reward,
            best_reward_std=self.best_reward_std,
            train_collector=self.train_collector,
            test_collector=self.test_collector,
        )

        self.logger.log_info_data(asdict(info_stat), self.epoch)

        # in case trainer is used with run(), epoch_stat will not be returned
        return EpochStats(
            epoch=self.epoch,
            train_collect_stat=train_stat,
            test_collect_stat=test_stat,
            training_stat=update_stat,
            info_stat=info_stat,
        )
