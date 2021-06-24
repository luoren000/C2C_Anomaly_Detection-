"""DeScarGAN experiment suite."""

from pathlib import Path
from typing import Dict, List

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from anomaly_toolbox.datasets.dataset import AnomalyDetectionDataset
from anomaly_toolbox.experiments.experiment import Experiment
from anomaly_toolbox.hps import hparam_parser
from anomaly_toolbox.trainers import DeScarGAN


class DeScarGANExperiment(Experiment):
    """
    DeScarGAN experiment.
    """

    def __init__(self, hparams_path: Path, log_dir: Path):
        super().__init__(hparams_path, log_dir)

        # Get the hyperparameters
        self._hps = hparam_parser(
            hparams_path,
            "descargan",
            self.hyperparameters().union(DeScarGAN.hyperparameters()),
        )

    def experiment(
        self, hps: Dict, log_dir: Path, dataset: AnomalyDetectionDataset
    ) -> None:
        """Experiment execution - architecture specific.
        Args:
            hps: dictionary with the parameters to use for the current run.
            log_dir: where to store the tensorboard logs.
            dataset: the dataset to use for model training and evaluation.
        """
        summary_writer = tf.summary.create_file_writer(str(log_dir / "summaries"))

        dataset.configure(
            anomalous_label=hps["anomalous_label"],
            batch_size=hps["batch_size"],
            new_size=(64, 64),
            output_range=(-1.0, 1.0),  # generator has a tanh in output
            shuffle_buffer_size=hps["shuffle_buffer_size"],
            cache=True,
        )
        summary_writer = tf.summary.create_file_writer(str(log_dir))
        trainer = DeScarGAN(dataset, hps, summary_writer, log_dir)
        trainer.train(
            batch_size=hps["batch_size"],
            epochs=hps["epochs"],
            step_log_frequency=hps["step_log_frequency"],
        )
