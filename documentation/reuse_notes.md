# Reuse notes

- Load a model: install nnU-Net v2.8.1 with the custom trainers and use `nnUNetv2_predict -d Dataset001_PulmonaryNodules -c <configuration> -tr <trainer> -p <plans> -chk checkpoint_final.pth`
  with `nnUNet_results` pointing to the corresponding folder in `models/`.
- The splits (`data/nnunet_dataset/splits_final.json`) and reference masks allow direct comparison with other detectors on the same LIDC-IDRI cases.
- Features and statistics of each stage are in `data/stage_outputs_10a_25/stage_<NN>/`; the column definitions are listed in `metadata/data_dictionary.csv`.
