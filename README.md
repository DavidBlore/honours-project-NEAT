## Credit

NEAT (NeuroEvolution of Augmenting Topologies) is a method developed by Kenneth O. Stanley for evolving arbitrary neural 
networks. The code in this repo comes from [neat-python](https://github.com/CodeReclaimers/neat-python) and is licensed under the [3-clause BSD license](https://opensource.org/licenses/BSD-3-Clause).


## Running NEAT

Ensure you are at the highest level in the directory structure before running anything.
To run NEAT:
```bash
python3 cpg_controller/evolve.py  -nc <num_cores> -ng <num_gens> -nrun <name_of_run> -c <checkpoint_interval> -s <standard_output> -r <restore_checkpoint>
```
E.g:
```bash
python3 cpg_controller/evolve.py  -nc 8 -ng 2500 -nrun 2500 -c 5
```

### Commandline arguments
NEAT takes in the following command line arguments:

|Flag       | Flag (long)           | Description                                     |
|-----------|-----------------------|-------------------------------------------------|
| -nc       | --num_cores           | the number of cores to use                      |
| -ng       | --num_gens            | the number of generations to run for            |
| -nrun     | --name_of_run         | the name of the run                             |
| -c        | --checkpoint_interval | how often to save checkpoints                   |
| -s        | --standard_output     | whether to print to standard output for the run |
| -r        | --restore_checkpoint  | the name of the checkpoint to restore           |