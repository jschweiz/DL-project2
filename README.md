# DL-project2

This project is a custom Deep Learning Framework using basic Torch tensor operations realized in the context of the Deep Learning course EE-559 at EPFL.

### Repository organisation

The repository is organized the following way:

```
DL-PROJECT2/
├── README.md             # overview of the project 
├── test.py               # test file requested in subject
├── compare.py            # file to run performance comparision 
├── notebook.ipynb        # development notebook
├── helpers.py            # data generation and training functions
|
├── dlcustomlib/          # package containng the custom framework 
    ├── nn/
        ├── modules/
            └── Linear/Sequential/MSELoss/ReLU/Tanh.py
        └── Activation/Loss/BaseModule.py           
    └── optim/
        ├── Optimizer.py
        └── SGD.py              
|
├── data/                 # comparison plots
```


### Requirements

The framwork only requirement is `Torch`. No additional modules are necessary to run `test.py`.
The `comparison.py` file is generating plots, so `matplotlib.pyplot` is needed.
The `notebok`was used for development and several other libraries may be needed to run its full content. 

### Projet report
Project report is available in `report.pdf`.
