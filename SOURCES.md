# Component credits and licensing

## BiLSTM and firmware

The BiLSTM implementation and the ESP32 / Arduino MPU-6050 firmware. These components retain their author credits. No license grant for these components is included; the MIT notice in LICENSE applies to the Transformer components and does not establish licensing for the BiLSTM or firmware.

## Transformer

The attention, patch embedding, positional encoding, encoder block, encoder, and supervised classifier components are copyright (c) 2026  and distributed under the MIT license. The complete notice is retained in [LICENSE](LICENSE).

## Dataset

UCI HAR: Human Activity Recognition Using Smartphones, by Davide Anguita et al. Obtain and cite the dataset according to its terms. Raw data is not redistributed in this repository.

## Project configuration

Both models receive 128-step windows with nine channels: body acceleration XYZ, body gyroscope XYZ, and total acceleration XYZ. Data preparation preserves the official training/test partition, reserves training subjects for validation, and fits normalization only on training subjects. Report model performance from saved evaluations for this configuration.
