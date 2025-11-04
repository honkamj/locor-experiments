# Repository for reproducing the results in the paper introducing Locor

This repostory contains the code for reproducing the results in the paper introducing the Locor registration method (see the [Publication](#publication) section). The repository for the registration method itself can be found at [https://github.com/honkamj/locor](https://github.com/honkamj/locor "https://github.com/honkamj/locor").

Note: This branch contains the experimental setup used in the MICCAI 2025 conference publication. An improved setup is documented by the main branch. The following bugs/shortcomings are present in this branch:
- Due to a bug, standard deviation for the Gaussian function defining the local window size is inadvertly fixed to 1.0 for our method (and Locor polynom.) during the hyperparameter optimization.
- The evaluation masks for the Head MRI-CT registration task are accidentally eroded, instead of dilated, from the body masks.
- Inferior hyperparameter optimizer is used.
- Less hyperparameters are included in the optimization.

The conclusions of the conference paper still hold.

## Environment setup

First install conda (https://docs.conda.io/en/latest/). To setup the enviroment navigate to directory ''devenv'' and execute ''setup.py'' with Python 3:

    python setup.py

The setup script will create a virtual enviroment with required packages installed. To activate the environent navigate to root and run:

    conda activate ./.venv

The repository has been designed to be used with Visual Studio Code (https://code.visualstudio.com/).

The version of locor used for the experiments in the paper is installed to the environment. Improvements may have been made to the locor registration tool after that. For the newest software version, see the repository [here](https://github.com/honkamj/locor "here"), and for experimental setup that works with the newest version, see the branch "up-to-date-locor" in this repository.

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

NOTE: Due to an oversight, some baseline results in the MICCAI 2025 conference paper used hyperparameters from the best validation trial instead of those selected via the intended GP-based smoothing. Such approach can be replicated by omitting the "--fit-gp-to-quantile" argument. The resulting numerical differences are minor for the baselines in question and have no effect on the conclusions of the paper.

## Publication

If you use the repository, please cite (see [bibtex](citations.bib)):

- **New multimodal similarity measure for image registration via modeling local functional dependence with linear combination of learned basis functions**  
[Joel Honkamaa](https://github.com/honkamj "Joel Honkamaa"), Pekka Marttinen  
MICCAI 2025 ([Open access](https://papers.miccai.org/miccai-2025/paper/1747_paper.pdf))

## License

The codebase is released under the MIT license.