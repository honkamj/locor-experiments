# Generating the pseudo-CT images for evaluation on the CERMEP-IDB-MRXFDG dataset

Start by requesting the CERMEP-IDB-MRXFDG database from the authors of the database (Mérida, Inés, et al. "CERMEP-IDB-MRXFDG: a database of 37 normal adult human brain [18F] FDG PET, T1 and FLAIR MRI, and CT images available for " research." EJNMMI research 11.1 (2021)). After that, follow these steps:

1. Setup the environment for the pseudo-CT generation method (https://github.com/honkamj/non-aligned-i2i) as instructed in the codebase.
2. Generate the "Head MRI to CT synthesis data set" as also instructed in the codebase. Note that the preprocessing script performs registration with elastix for comparison purposes, which is not needed here. To save time one can omit that part in the preprocessing script.
3. The preprocessing script generates the data into three folders "train", "validate", and "test". We want to put all the data in the "train" split to get best quality synthesis since it does not matter for us if the generated pseudo-CT images are from the training set (we are not evaluating them). Hence move all the data to the "train" folder.
4. Run the following command to train the model:

    python -m scripts.training.train_non_aligned_i2i_with_discriminator --config <path_to_config> --target-root <target_root> --model-name CERMEP_pseudo_CT --devices cuda:0 --seed 123456789
    
    Path to the config should point to the config provided in this codebase [here](i2i_config.json).
5. After the training is finished run the inference for the last epoch (15) with the following command:

    python -m scripts.inference.inference_nifti --target-root <target_root> --model-name CERMEP_pseudo_CT --data-set train --epoch 15 --devices cuda:0

Now when running the evaluation with CERMEP dataset for any method as instructed in the [main instructions](../README.md), the script will on the first time ask for the directory where the pseuco-CT images are located. Provide it with the path "<target_root>/CERMEP_pseudo_CT/inference/epoch015/train".