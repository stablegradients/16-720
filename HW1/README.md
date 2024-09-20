# Homework 1: Scene Recognition

This repository contains the code for Homework 1 of the 16-720 course. The main focus of this homework is on scene recognition using various hyperparameters.


### How to Run `main.py`

To run the `main.py` script with the default hyperparameters, use the following command:
```bash
python main.py
```

This will run the solution to all queations one after the other sequentially. Feel free to comment the parts not needed.


If you want to customize the hyperparameters, you can do so by providing the appropriate command-line arguments. For example:
```bash
python main.py --data-dir /path/to/data --feat-dir /path/to/feat --out-dir /path/to/output --filter-scales 2 5 --K 50 --alpha 30 --L 2
```

### Hyperparameters

- `--data-dir`: Specifies the data folder (default: `../data`)
- `--feat-dir`: Specifies the feature folder (default: `../feat`)
- `--out-dir`: Specifies the output folder (default: `.`)
- `--filter-scales`: A list of scales for all the filters (default: `[1, 2]`)
- `--K`: Number of words (default: `10`)
- `--alpha`: Using only a subset of alpha pixels in each image (default: `25`)
- `--L`: L + 1 is the number of layers in spatial pyramid matching (SPM) (default: `1`)

### How to Run `ablate.sh`

To run the `ablate.sh` script, use the following command:
```bash
bash ablate.sh
```

This script will perform ablation studies by systematically removing or altering certain components of the model to understand their impact on performance.

If you need to customize the script, you can edit the `ablate.sh` file directly to modify the parameters or the components being ablated.
