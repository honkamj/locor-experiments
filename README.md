# Repository for reproducing the results in the paper introducing Locor

This repostory contains the code for reproducing the results in the paper introducing the Locor registration method (see the [Publication](#publication) section). The repository for the registration method itself can be found at [https://github.com/honkamj/locor](https://github.com/honkamj/locor "https://github.com/honkamj/locor").

## Environment setup

First install conda (https://docs.conda.io/en/latest/). To setup the enviroment navigate to directory ''devenv'' and execute ''setup.py'' with Python 3:

    python setup.py

The setup script will create a virtual enviroment with required packages installed. To activate the environent navigate to root and run:

    conda activate ./.venv

The repository has been designed to be used with Visual Studio Code (https://code.visualstudio.com/).

This branch (up-to-date-locor) contains version of the experiments compatible with the newest version of Locor. For the original setup, see the main branch.

## Running the experiments

For this section we assume that you have navigated to directory ''src'' and have activated the environment.

The experiments can be ran via the ''run.py'' python script.

To run a hyperparameter optimization for given dataset and method, run the following command:

    python run.py --target-folder <folder_for_the_results> --data-root <data_root_where_the_data_will_be_downloaded> --n-trials <number_of_hyperparameter_optimization_trials> --dataset <name_of_the dataset> <name_of_the_method> optimize_hyperparameters <any_method_specific_arguments>

After the hyperparameter optimization has been done, run the following command to run the evaluation on the test set:

    python run.py --target-folder <folder_for_the_results> --data-root <data_root_where_the_data_will_be_downloaded> --n-trials <number_of_test_trials> --dataset <name_of_the dataset> <name_of_the_method> test --fit-gp-to-quantile 0.4 <any_method_specific_arguments>

Available methods are: "locor", "locor_ablation_study", "NiftyReg_MIND", "NiftyReg_NMI", "ANTs", "corrfield", and "SRWCR".

Available datasets are "CERMEP", "IXI", "CT-MR_Thorax-Abdomen_foreground_mask", "CT-MR_Thorax-Abdomen_roi_mask". The data will be downloaded automatically, if possible.

See the [separate instructions](documentation/CERMEP_data_generation.md) for generating the pseudo-CT images for the evaluation on the CERMEP-IDB-MRXFDG dataset. 

## Publication

If you use the repository, please cite (see [bibtex](citations.bib)):

- **New multimodal similarity measure for image registration via modeling local functional dependence with linear combination of learned basis functions**  
[Joel Honkamaa](https://github.com/honkamj "Joel Honkamaa"), Pekka Marttinen  
Under review ([eprint arXiv:2503.05335](https://arxiv.org/abs/2503.05335 "eprint arXiv:2503.05335"))

## License

The codebase is released under the MIT license.