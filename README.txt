# RFN.

## Training
There are two types of example scripts in the scripts folder.
param_search_actor_frame.sh and run_actor_frame_search.sh are examples of the memory saved version, and it can be started by ./scripts/run_actor_frame_search.sh. The parameters can be adjusted in the param_search_actor_frame.sh
param_search_cornell.sh and run_cornell_search.sh are examples of the normal version, and it can be started by ./scripts/run_cornell_search.sh. The parameters can be adjusted in the param_search_cornell.sh


## File Descriptions

`data/`: Datasets
`distribution/`: Hyperbolic Distributions
`kernels/`: Kernel generation
`layers/`: Hyperbolic layers
`manifolds/`: Manifold calculations
`models/`: GNN models
`optim/`: Optimization on manifolds
`utils/`: Utility files
`train.py`: Training scripts

Our Frame method is located in `kernels/kernel_points.py`.
