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

"""Surface Crack Dataset (https://www.kaggle.com/arunrk7/surface-crack-detection).

The datasets contains images of various concrete surfaces with and without crack.
The image data are divided into two as negative (without crack) and positive
(with crack) in separate folder for image classification.
Each class has 20000 images with a total of 40000 images with 227 x 227 pixels
with RGB channels.
The dataset is generated from 458 high-resolution images (4032x3024 pixel) with
the method proposed by Zhang et al (2016).
High resolution images found out to have high variance in terms of surface finish
and illumination condition.
No data augmentation in terms of random rotation or flipping or tilting is applied.
"""

import zipfile
from functools import partial
from glob import glob
from io import BytesIO
from pathlib import Path
from typing import Tuple, Union
from PIL import Image
import numpy as np
import random

import requests
import tensorflow as tf
import os


from anomaly_toolbox.datasets.dataset import AnomalyDetectionDataset


class COVID19(AnomalyDetectionDataset):
    """COVID19 DATASET.
    20000 images for training (balanced), 5000 images for validation (balanced)
    5000 images for testing (balanced).
    """

    def __init__(self, path: Path = Path("/projects/care-to-cure/Anomaly-detection-POD1/covid19/COVID-19_Radiography_Dataset/")):
        super().__init__()
        self._path = path

        def _read_and_map_fn(label):
            """Closure used in tf.data.Dataset.map for creating
            the correct pair (image, label).
            """

            def fn(filename):
                binary = tf.io.read_file(filename)
                image = tf.image.decode_png(binary)
                return image, label

            return fn

        glob_ext = "*.png"
        all_normal = glob(str(self._path / "Normal/images" / glob_ext))
        all_normal_train = all_normal[:int(len(all_normal) * 0.8)]
        all_normal_test = all_normal[int(len(all_normal) * 0.8):int(len(all_normal) * 0.9)]
        all_normal_validation = all_normal[int(len(all_normal) * 0.9):]
        print(len(all_normal_train), len(all_normal_test), len(all_normal_validation))
        # all_normal_train = all_normal[:10000]
        # all_normal_test = all_normal[10000:15000]
        # all_normal_validation = all_normal[15000:]

        all_anomalous = glob(str(self._path / "COVID/images" / glob_ext))
        all_anomalous += glob(str(self._path / "Lung_Opacity/images" / glob_ext))
        #all_anomalous += glob(str(self._path / "Viral Pneumonia/images" / glob_ext))
        random.shuffle(all_anomalous)
        all_anomalous_train = all_anomalous[:int(len(all_anomalous) * 0.8)]
        all_anomalous_test = all_anomalous[int(len(all_anomalous) * 0.8):int(len(all_anomalous) * 0.9)]
        all_anomalous_validation = all_anomalous[int(len(all_anomalous) * 0.9):]
        print(len(all_anomalous_train), len(all_anomalous_test), len(all_anomalous_validation))

        self._train_raw = (
            tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(all_normal_train))
            .map(_read_and_map_fn(self.normal_label))
            .concatenate(
                tf.data.Dataset.from_tensor_slices(
                    tf.convert_to_tensor(all_anomalous_train)
                ).map(_read_and_map_fn(self.anomalous_label))
            )
        )


        self._test_raw = (
            tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(all_normal_test))
            .map(_read_and_map_fn(self.normal_label))
            .concatenate(
                tf.data.Dataset.from_tensor_slices(
                    tf.convert_to_tensor(all_anomalous_test)
                ).map(_read_and_map_fn(self.anomalous_label))
            )
        )

        self._validation_raw = (
            tf.data.Dataset.from_tensor_slices(
                tf.convert_to_tensor(all_normal_validation)
            )
            .map(_read_and_map_fn(self.normal_label))
            .concatenate(
                tf.data.Dataset.from_tensor_slices(
                    tf.convert_to_tensor(all_anomalous_validation)
                ).map(_read_and_map_fn(self.anomalous_label))
            )
        )
        print(len(all_anomalous_train), len(all_anomalous_test), len(all_anomalous_validation))

        # RGB dataset
        #self._channels = 3

    def configure(
        self,
        batch_size: int,
        new_size: Tuple[int, int],
        anomalous_label: Union[int, str, None] = None,
        class_label: Union[int, str, None] = None,
        shuffle_buffer_size: int = 16819,
        cache: bool = True,
        drop_remainder: bool = True,
        output_range: Tuple[float, float] = (0.0, 1.0),
    ) -> None:
        """Configure the dataset. This makes all the object properties valid (not None).
        Args:
            batch_size: The dataset batch size
            new_size: (H,W) of the input image.
            anomalous_label: If the raw dataset contains label, all the elements with
                             "anomalous_label" are converted to element of
                             self.anomalous_label class.
            class_label: If the raw dataset contains different classes (each one
                         containing both positive and negative samples) we can select
                         only one class to focus on (e.g. a dataset of industrial
                         defects on industrial objects composed of transistors and
                         pills and we are interested only in transistors and not on pills).
            shuffle_buffer_size: Buffer size used during the tf.data.Dataset.shuffle call.
            cache: If True, cache the dataset
            drop_remainder: If True, when the dataset size is not a multiple of the dataset size,
                            the last batch will be dropped.
            output_range: A Tuple (min, max) containing the output range to use
                          for the processed images.
        """
        pipeline = partial(
            self.pipeline,
            new_size=new_size,
            batch_size=batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
            cache=cache,
            drop_remainder=drop_remainder,
            output_range=output_range,
        )


        pipeline_train = partial(pipeline, is_training=True)
        pipeline_test = partial(pipeline, is_training=False)
        is_anomalous = lambda _, label: tf.equal(label, anomalous_label)
        is_normal = lambda _, label: tf.not_equal(label, anomalous_label)

        # Train-data
        self._train_anomalous = (
            self._train_raw.filter(is_anomalous)
            .map(lambda x, _: (x, self.anomalous_label))
            .apply(pipeline_train)
        )
        self._train_normal = (
            self._train_raw.filter(is_normal)
            .map(lambda x, _: (x, self.normal_label))
            .apply(pipeline_train)
        )
        self._train = self._train_raw.apply(pipeline_train)

        # Test-data
        self._validation_anomalous = (
            self._validation_raw.filter(is_anomalous)
            .map(lambda x, _: (x, self.anomalous_label))
            .apply(pipeline_test)
        )
        self._validation_normal = (
            self._validation_raw.filter(is_normal)
            .map(lambda x, _: (x, self.normal_label))
            .apply(pipeline_test)
        )
        self._validation = self._validation_raw.apply(pipeline_test)

        # Test-data
        self._test_anomalous = (
            self._test_raw.filter(is_anomalous)
            .map(lambda x, _: (x, self.anomalous_label))
            .apply(pipeline_test)
        )
        self._test_normal = (
            self._test_raw.filter(is_normal)
            .map(lambda x, _: (x, self.normal_label))
            .apply(pipeline_test)
        )
        self._test = self._test_raw.apply(pipeline_test)

# def read_and_map_fn(label):
#             """Closure used in tf.data.Dataset.map for creating
#             the correct pair (image, label).
#             """
#
#             def fn(filename):
#                 binary = tf.io.read_file(filename)
#                 image = tf.image.decode_png(binary)
#                 return image, label
#
#             return fn
#
# path = Path("/Users/luoren000/Documents/luoren000/Anomaly_detection/COVID-19_Radiography_Dataset/")
# glob_ext = "*.png"
# all_normal = glob(str(path/"COVID/images"/glob_ext))
# #print(list(x.as_numpy_iterator()))
# # print(len(all_normal))
# all_normal+= glob(str(path/"Lung_Opacity/images"/glob_ext))
# all_normal+= glob(str(path/"Viral Pneumonia/images"/glob_ext))
# #all_normal= glob(str(path/"Normal/images"/glob_ext))
# #x = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(all_normal)).map(read_and_map_fn(tf.constant(0)))
#
# train = all_normal[:int(len(all_normal)*0.8)]
# valid = all_normal[int(len(all_normal)*0.8):int(len(all_normal)*0.9)]
# test = all_normal[int(len(all_normal)*0.9):]
# print(len(train), len(valid), len(test))
# #
# # print(len(all_normal))
# #
# # path = Path("surface_cracks")
# # glob_ext = "*.jpg"
# # foldr = os.path.join('src','anomaly_toolbox')
# # all_normal = glob(r'*\*.py')
# # all_normal = os.listdir('/Users/luoren000/Documents/luoren000/Anomaly_detection/COVID-19_Radiography_Dataset/Lung_Opacity/images')
# # print(len(all_normal))
# # all_anomalous = tf.data.Dataset.from_tensor_slices(
# #     tf.convert_to_tensor(all_normal)
# # ).map(read_and_map_fn(tf.constant(0)))
# # all_anomalous = all_anomalous.shuffle(buffer_size=len(all_normal))
# # print(all_anomalous)
# #
# all_normal = glob(str(path/"Normal/images"/glob_ext))
# train = all_normal[:int(len(all_normal)*0.8)]
# print(len(train))
# #
# # img = Image.open('/Users/luoren000/Documents/luoren000/Anomaly_detection/COVID-19_Radiography_Dataset/COVID/images/COVID-1.png')
# # data = np.asarray(img)
# # print(len(data.shape))
'''f = ['COVID','Lung_Opacity','Normal','Viral Pneumonia']
path = Path("/projects/care-to-cure/Anomaly-detection-POD1/covid19/COVID-19_Radiography_Dataset/")
broken = []
for i in f:
    for filename in os.scandir(str(path/'{}/images'.format(i))):
        if filename.is_file():
            img = Image.open(str(path/'{}/images'.format(i)/filename))
            img = np.asarray(img)
            if len(img.shape) != 2:
                os.remove(str(path/'{}/images'.format(i)/filename))
                print(filename)'''

