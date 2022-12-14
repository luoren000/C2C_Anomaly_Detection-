# Copyright 2021 Zuru Tech HK Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""EGBAD experiments suite."""

from pathlib import Path
from typing import Dict
import sys
sys.path.insert(0,'/projects/care-to-cure/Anomaly-detection-POD1/codes/Renpin_Luo/C2C_Anomaly_Detection-/src/anomaly_toolbox/datasets')
import tensorflow as tf

from anomaly_toolbox.datasets.dataset import AnomalyDetectionDataset
from anomaly_toolbox.experiments.experiment import Experiment
from anomaly_toolbox.hps import hparam_parser
from anomaly_toolbox.trainers import EGBAD
from anomaly_toolbox.hps import grid_search
from COVID19 import COVID19


class EGBADExperiment(Experiment):
    """
    EGBAD experiment.
    """

    def __init__(self, hparams_path: Path, log_dir: Path):
        super().__init__(hparams_path, log_dir)

        # Get the hyperparameters
        self._hps = hparam_parser(
            hparams_path,
            "egbad",
            list(self.hyperparameters().union(EGBAD.hyperparameters())),
        )
        print("check555555555555555555555")

    def experiment(
        self, hps: Dict, log_dir: Path, dataset: AnomalyDetectionDataset
    ) -> None:
        """Experiment execution - architecture specific.
        Args:
            hps: Dictionary with the parameters to use for the current run.
            log_dir: Where to store the tensorboard logs.
            dataset: The dataset to use for model training and evaluation.
        """
        print("Running EGBAD experiment again...")

        summary_writer = tf.summary.create_file_writer(str(log_dir))
        new_size = (28, 28)

        # Create the dataset with the requested sizes (requested by the model architecture)
        dataset.configure(
            anomalous_label=hps["anomalous_label"],
            class_label=hps["class_label"],
            batch_size=hps["batch_size"],
            new_size=new_size,
            shuffle_buffer_size=hps["shuffle_buffer_size"],
        )

        # Create the EGBAD trainer
        trainer = EGBAD(
            dataset=dataset,
            hps=hps,
            summary_writer=summary_writer,
            log_dir=log_dir,
        )
        
        print("not ok")

        # Train the EGBAD model
        '''trainer.train(
            epochs=hps["epochs"],
            step_log_frequency=hps["step_log_frequency"],
        )'''

        # Test on test dataset and put the results in the json file (the same file used inside the
        # training for the model selection)
        trainer.test()


experiment_instance = EGBADExperiment(hparams_path='/projects/care-to-cure/Anomaly-detection-POD1/codes/Renpin_Luo/C2C_Anomaly_Detection-/config/hparams.json', log_dir=Path('/projects/care-to-cure/Anomaly-detection-POD1/codes/Renpin_Luo/C2C_Anomaly_Detection-/logs/EGBADExperiment'))
print("check111111111111111111")
coviddataset = COVID19()
print("check222222222222222")
experiment_instance.run(hparams_tuning=False, hparams_func=grid_search, dataset=coviddataset)
print("check333333333333333")

