# _BETA VERSION - IN DEVELOPMENT_

# ESCHR

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/smgoggin10/ESCHR/test.yaml?branch=main
[link-tests]: https://github.com/smgoggin10/ESCHR/actions/workflows/test.yml
[badge-docs]: https://img.shields.io/readthedocs/ESCHR

ESCHR: A hyperparameter-randomized ensemble approach for robust clustering across diverse datasets

## Overview of Algorithm:

![alt text](https://github.com/smgoggin10/ESCHR/blob/main/Images/sccc_fig1_no_letters.png)

## Installation

You need to have Python 3.8 or newer installed on your system. If you don't have
Python installed, we recommend installing [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge).

1. Make sure you have Anaconda installed and functional. [Conda FAQ](https://docs.anaconda.com/anaconda/user-guide/faq/) is a great resource for troubleshooting and verifying that everything is working properly.
2. Open terminal or equivalent command line interface and run `conda create --name <env_name>`
3. Activate the environment by running `conda activate <env_name>`
4. Once environment is activated, run `conda install pip`
5. If you do not have Git installed, run `conda install git`
6. To install ESCHR into your conda environment, run the following line:
   `pip install git+https://github.com/smgoggin10/ESCHR.git`
7. Verify that the Conda environment was created successfully by running `conda list` and verifying that expected packages are installed for this environment. Then either close the environment by running `conda deactivate` or proceed to subsequent optional setup and/or running the method within the environment.

## Getting started

For full documentation, please refer to the [documentation][link-docs] for further details.

### Most basic example run script:

```
import ESCHR as es
import pandas as pd

# Read in data from a csv file.
# The method expects features as columns
# Use commented out ".T" if you have features as rows
# Remove "index_col = 0" if your csv does not have row indices included.
# Also ensure that data has already been preprocessed/scaled/normalized
# as appropriate for your data type.
data_filepath = "/path/to/your/data.csv"
data = pd.read_csv(data_filepath, index_col=0)#.T

# Create the zarr store that will be used for interacting with your data
zarr_loc = "/path/to/your/data.zarr"
es.make_zarr(data=data, zarr_loc=zarr_loc)

# Initialize a ConsensusCluster instance
# (add any optional hyperparameter specifications,
# but bear in mind the method was designed to work for
# diverse datasets with the default settings.
cc_obj = es.ConsensusCluster(zarr_loc=zarr_loc)

# Now you can run the method with your prepped data:
cc_obj.consensus_cluster()

# For most built-in visualizations and/or
# for compatibility with scverse suite of tools,
# you should next generate an AnnData object containing all outputs.
# There is a ConsensusCluster class method for doing this!
# This will add the AnnData object as an attribute to the
# ConsensusCluster object.
cc_obj.make_adata(
    feature_names=data.columns.values,
    sample_names=data.index.values
)

# To extract anndata object for downstream applications, run the following:
adata = cc_obj.adata

# Plot soft membership matrix heatmap visualization
es.pl.make_smm_heatmap(cc_obj, output_path="/where/to/save/figure.png")

# Plot umap visualization
es.pl.plot_umap(cc_obj, output_path="/where/to/save/figure.png")
```

### Setting up to run via command line:

```
conda activate <env_name>
python3
import ESCHR
```

Now you can run code adapted from the example run scripts above or copy and paste lines of code from the tutorial jupyter notebook.

### Setting up to run a Jupyter Notebook on your PC:

(note it is likely only small datasets will be able to run on a PC, but feel free to try large ones!)

1. Open terminal or equivalent command line interface, activate the environment you create above by running `conda activate <env_name>`
2. While in the activated environment, run `conda install -c anaconda ipykernel`
3. Next run `python -m ipykernel install --user --name=<env_name>`.
4. You can then then close the environment by running `conda deactivate`
5. Open Anaconda Navigator and click on the icon for Jupyter Notebooks (this should open an instance of a command line interface, and then open a tab in your default browser to a page containing your PC's file structure)
6. Navigate to where you saved the downloaded tutorial notebook (instructions for downloading tutorial can be found in `Tutorial` section below) and click to open that notebook, or start a fresh notebook if you prefer to work off of the `Quick Start` instructions.
7. Upon opening the notebook, you may be prompted to select a kernel, or if not you can click on the `Kernel` menu from the top navigation bar, and then `Change kernel`. The name of the environment you created should show up as an option for the kernel - select that as the kernel for your notebook.
8. You should now be ready to run! Just click your way through the notebook. You can change output paths for the visualizations when you get to those cells.

### Setting up to run a Jupyter Notebook on HPC OpenOnDemand:

1. Navigate to [UVA OpenOnDemand](https://rivanna-portal.hpc.virginia.edu/pun/sys/dashboard/) if you are a member of the University fo Virginia, otherwise navigate to the equivalent for your institution.
2. Enter the dropdown menu for "Interactive Apps" from the top menu bar and select "JupyterLab"
3. Submit a job - for optimal performance, select 4+ cores and 100GB memory (many cases won't end up needing that much memory, but it is very unlikely anything would ever exceed that - highest I've seen for peak mem is pushing 50 GB)
4. Once your job starts, click `Connect to Jupyter`
5. First, start a terminal window. You can either port in your conda env from your computer to the location of conda envs in your Rivanna storage (home/<your compute id>/.conda/envs - you will have to click `show dot files` to find this). Alternatively, you can upload the requirements.txt file and follow the installation instructions above to create the environment on Rivanna.
6. Either way, you want to end up in this terminal window with your environment activated (`conda activate <env_name>`). While in the activated environment, run `conda install -c anaconda ipykernel`
7. Next run `python -m ipykernel install --user --name=<env_name>`.
8. You can then then close the environment by running `conda deactivate`.
9. Navigate to where you uploaded the tutorial notebook in your rivanna files, or upload it to your current working directory in the file navigation pane to the left of the JupyterLab GUI. (instructions for downloading the tutorial notebook from GitHub can be found in `Tutorial` section below) and click to open that notebook, or start a fresh notebook if you prefer to work off of the `Quick Start` instructions.
10. Once you open the notebook, you can set the kernel to be the environment you created by clicking on the current kernel name (upper right corner, to the left of a gray circle).
11. You should now be ready to run through the notebook!

## Release notes

See the [changelog][changelog].

## Contact

For questions and help requests, you can reach out to Sarah Goggin by [email](mailto:sg4dm@virginia.edu).
If you found a bug, please use the [issue tracker][issue-tracker].

## Citation

> t.b.a

[issue-tracker]: https://github.com/smgoggin10/ESCHR/issues
[changelog]: https://ESCHR.readthedocs.io/latest/changelog.html
[link-docs]: https://ESCHR.readthedocs.io
[link-api]: https://ESCHR.readthedocs.io/latest/api.html
