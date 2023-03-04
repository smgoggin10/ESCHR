# SHaRC

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/smgoggin10/SHaRC/test.yaml?branch=main
[link-tests]: https://github.com/smgoggin10/SHaRC/actions/workflows/test.yml
[badge-docs]: https://img.shields.io/readthedocs/SHaRC

Stable hyperparameter-randomized consensus clustering

## Overview of Algorithm:
![alt text](https://github.com/smgoggin10/SHaRC/blob/main/Images/sccc_fig1_no_letters.png)

## Getting started

Please refer to the [documentation][link-docs] for details. In particular, the

-   [API documentation][link-api].

## Installation

You need to have Python 3.8 or newer installed on your system. If you don't have
Python installed, we recommend installing [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge).

Temporary instructions for private repository:
1. Make sure you have Anaconda installed and functional. [Conda FAQ](https://docs.anaconda.com/anaconda/user-guide/faq/) is a great resource for troubleshooting and verifying that everything is working properly.
2. Download the requirements.txt file from the listed files above (are instructions for this reqd?)
3. Create a token
   1. Go to this link when logged into your github account: https://github.com/settings/tokens/new
   2. Check off "repo" in the settings for your token.
   3. Click generate token and copy/save the provided code (your PAT) somewhere.
4. Open the requirements.txt file and replace `${GITHUB_TOKEN}` in the final line with the token you just generated, save the file
5. Open terminal or equivalent command line interface and run `conda create --name <env_name>` 
6. Activate the environment by running `conda activate <env_name>`  
7. Once environment is activated, run `conda install pip`
8. If you do not have Git installed, run `conda install git`
9. Finally, run `pip install -r <path/to/requirements.txt>`. This will intall this package and the necessary dependencies.
 (optional for max performance: special install nmslib by running `pip install --no-binary :all: nmslib`)
10. Verify that the Conda environment was created successfully by running `conda list` and verifying that expected packages are installed for this environment. Then either close the environment by running `conda deactivate` or proceed to subsequent optional setup and/or running the method within the environment.
 
##### Ignore below instructions, simple pip installation is not yet available.

There are several alternative options to install SHaRC:

<!--
1) Install the latest release of `SHaRC` from `PyPI <https://pypi.org/project/SHaRC/>`_:

```bash
pip install SHaRC
```
-->

1. Install the latest development version:

```bash
pip install git+https://github.com/smgoggin10/SHaRC.git@main
```

## Release notes

See the [changelog][changelog].

## Contact

For questions and help requests, you can reach out to Sarah Goggin by [email](mailto:sg4dm@virginia.edu).
If you found a bug, please use the [issue tracker][issue-tracker].

## Citation

> t.b.a

[issue-tracker]: https://github.com/smgoggin10/SHaRC/issues
[changelog]: https://SHaRC.readthedocs.io/latest/changelog.html
[link-docs]: https://SHaRC.readthedocs.io
[link-api]: https://SHaRC.readthedocs.io/latest/api.html
