"""Runs NEAT algorithm, using CPPNs to generate high performing CPGControllers which are seeded into MAP-Elites later.

Refer to the research paper for my details.
The best genomes are saved to a file under the `best-genomes` folder

Takes in the following command line arguments:
    Flag    Flag (long)             Description
    _____   _____________________   ____________________________________________
    -nc     --num_cores             : the number of cores to use
    -ng     --num_gens              : the number of generations to run for
    -nrun   --name_of_run           : the name of the run
    -c      --checkpoint_interval   : how often to save checkpoints
    -s      --standard_output       : whether to print to standard output for the run
    -r      --restore_checkpoint    : the name of the checkpoint to restore
"""
from datetime import datetime
import os
from hexapod.controllers.cpg_controller_mouret import CPGControllerMouret
from hexapod.controllers.cpg_controller_mouret import CPGParameterHandler
from hexapod.simulator import Simulator
import neat
import visualize
import argparse


def eval_genome(genome, config, print_if_fitness=0.0):
    """
    Evaluates an individual genome.
    
    Used by the parallel evaluator. Use eval_genomes for a 'non-parallel' evaluator.

    Args:
        genome: the genome to evaluate
        config: the config object to use (based on config  file)
        print_if_fitness: print out params of controller if the fitness is greater than this value
    """
    genome.fitness = 0.0
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    params = get_params(net)
    # try and evaluate controller with these parameters and see how it does
    try:
        my_controller = CPGControllerMouret(params['intrinsic_amplitudes'], params['phase_biases'],seconds=5, velocity=0,crab_angle=0)
        simulator = Simulator(my_controller, follow=False, visualiser=False, collision_fatal=True, failed_legs=[])
        x=0
        while x<(240*5)-1:
            simulator.step()
            x=x+1
        genome.fitness = float(simulator.base_pos()[0]) # distance travelled along x axis
        simulator.terminate()
    except Exception as e:
        # typically this means there was just a leg collison (i.e., the parameters werent great)
        # however, just check it and make sure this is the case and you not catching a real error 
        # and recording a fitness of 0.0 then:
        # print(e)
        pass
    if print_if_fitness>0.0 and genome.fitness >= print_if_fitness:
            print("intrinsic amplitudes: " + str(params['intrinsic_amplitudes'])+"\n")
            print("phase biases: " + str(params['phase_biases'])+"\n\n")
    # print("genome fitness:",genome.fitness)
    return genome.fitness

def run(config_file, num_cores=8, num_gens=2500, name_of_run=datetime.strftime(datetime.now(), '%d-%m_%H-%M'), checkpoint_interval=5, standard_output=True, restore_checkpoint="", visualize=False):
    """
    Runs the NEAT algorithm to train a neural network to find parameters for the CPGController 
    to control the hexapod.

    Args:
        config_file: path to the config file to use for the NEAT algorithm.
        num_cores: number of cores to use for the NEAT algorithm.
        num_gens: number of generations to run the NEAT algorithm for.
        name_of_run: name of the run.
        checkpoint_interval: how often to checkpoint the current best genome.
        standard_output: whether to print to standard output or not.
        restore_checkpoint: path to a checkpoint to restore from.
    """
    # Load configuration
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    p = neat.Population(config) # Create the population, which is the top-level object for a NEAT run.
    # restore state if restoring from a checkpoint file
    if restore_checkpoint != "":
        p = neat.Checkpointer.restore_checkpoint(restore_checkpoint)
        p.config = config
    if standard_output:
        p.add_reporter(neat.StdOutReporter(True)) # Add a stdout reporter to show progress in the terminal.
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(checkpoint_interval,time_interval_seconds=600, filename_prefix=name_of_run+"-checkpoint-"))

    # Run actual algorithm
    pe = neat.ParallelEvaluator(num_cores, eval_genome) # use for parallel
    start_time = datetime.now()                         # start timer
    winner = p.run(pe.evaluate, num_gens)               # Run for up to <num_gens> generations.
    end_time = datetime.now()                           # end timer    
    time_taken = (end_time - start_time)                # time taken to run the algorithm
    print("time taken:",time_taken)                     # print time taken to run
    
    node_names = {-1:'Leg', -2: 'Joint', -3:'Other Leg', -4:'Other Joint', 0:'Intrisic amp', 1:'Phase bias'}
    if visualize:
        visualize.draw_net(config, winner, True, node_names=node_names)
        visualize.plot_stats(stats, ylog=False, view=True)
        visualize.plot_species(stats, view=True)
    # required to ensure saving
    # p = neat.Checkpointer.restore_checkpoint('f-2500-2-08-Aug_12-57-checkpoint-2499')
    # p.run(eval_genomes, 3)

    # Print and save all the best genomes
    print('\nBest genome:\n{!s}'.format(winner))                        # Display the winning genome.
    print('\nOutput:')                                                  # Show output of the most fit genome against training data.

    # FOR DEBUG - print out best genome's parameters
    # winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    # winner_params = get_params(winner_net)
    # print("phase_biases:\n",winner_params['phase_biases'])
    # print("\nintrinsic_amplitudes:\n",winner_params['intrinsic_amplitudes'])

    # get the best genomes from each species and save it to a file
    best_genomes = []
    for s in p.species.species.items():
        best_genome = None
        best_fitness = 0.0
        for member in s[1].members.items():
            try:
                if member[1].fitness > best_fitness:
                    best_fitness = member[1].fitness
                    best_genome = member[1]
            except Exception as e:
                pass
        best_genomes.append(best_genome)
    save_best_genomes(best_genomes,config,filename_prefex=name_of_run)
   
def get_params(net):
    """
    Gets the parameters of the CPGController from the network.

    Args:
        net: the network to get the parameters from

    Returns:
        params: a dictionary containing the parameters of the CPGController {'phase_biases':.., 'intrinsic_amplitudes':..}
    """
    phase_biases = [[[],[],],[[],[],],[[],[],],[[],[],],[[],[],],[[],[],],]
    intrinsic_amplitudes = []
    for leg in range(6):
        for joint in range(2):
            output = net.activate(inputs=(leg,joint,-1,-1)) # -1 means no phase bias / just want intrinsic amplitude
            intrinsic_amplitudes.append(CPGParameterHandler.scale_intrinsic_amplitude(output[0],joint)) # scale intrinsic amplitude to be within the right range for that joint
            # get each parameter by activating the network with the right inputs
            for other_leg in range(6):
                other_joint_bias = []
                for other_joint in range(2):
                    if not(other_leg==leg and other_joint==joint):
                        output = net.activate(inputs=(leg,joint,other_leg,other_joint))
                        other_joint_bias.append(CPGParameterHandler.scale_phase_bias(output[1]))
                    else:
                        other_joint_bias.append(None)
                phase_biases[leg][joint].append(other_joint_bias)
    return {'phase_biases':phase_biases,'intrinsic_amplitudes':intrinsic_amplitudes}

def save_best_genomes(best_genomes, config,detailed=False, num_genomes=5, filename_prefex=""):
    """
    Saves the best unique genomes from the last generation to a file.

    Args:
        best_genomes: the best genomes from each species
        config: the configuration object from the NEAT algorithm
        detailed: whether to save the genomes in a detailed format or not
        num_genomes: the number of best unique genomes to save
        filename_prefex: the prefix of the filename to save the best genomes to

    Returns:
        The best genomes in a list
    """
    num_genomes = len(best_genomes)
    filename = filename_prefex+"-best_genomes.txt"
    filename = os.path.join(local_dir, "..","best-genomes", filename)
    file = open(filename,"w")
    # iterate through each genome and save it to a file
    for i, genome in enumerate(best_genomes):
        if genome is not None:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            params = get_params(net)

            file.write("{rank}:\n".format(rank=i))
            if detailed:
                file.write("{!s}\n".format(genome))
            else:
                file.write("Fitness: {fitness}\n".format(fitness=genome.fitness))
            file.write("\nphase_biases:\n{phase_biases}\n".format(phase_biases=params['phase_biases']))
            file.write("intrinsic_amplitudes:\n{intrinsic_amplitudes}\n\n\n".format(intrinsic_amplitudes=params['intrinsic_amplitudes']))
            file.write("-------------------------------------------------------\n")
            file.flush()
        else:
            num_genomes -= 1
    file.close()
    print("Saved best {n} genomes to '{filename}'".format(n=num_genomes,filename=filename))
    return best_genomes

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run NEAT on CPG controller.')
    parser.add_argument('-nc','--num_cores',            required=True, type=int, default=8, help='the number of cores to use')
    parser.add_argument('-ng','--num_gens' ,            required=True, type=int, default=2500, help='the number of generations to run for')
    parser.add_argument('-nrun','--name_of_run',        required=True, type=str, default="", help='the name of the run')
    parser.add_argument('-c','--checkpoint_interval',   required=False, type=int, default=5, help='how often to save checkpoints')
    parser.add_argument('-s','--standard_output',       required=False, type=bool, default=True, help='whether to print to standard output for the run')
    parser.add_argument('-r','--restore_checkpoint',    required=False, type=str, default="", help='the name of the checkpoint to restore')
    args = parser.parse_args()    
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-cpg')
    # Run the NEAT algorithm
    run(
        config_path, 
        name_of_run=args.name_of_run+datetime.strftime(datetime.now(), '-(%d-%b_%H-%M)'), 
        num_cores=args.num_cores, 
        num_gens=args.num_gens,
        checkpoint_interval=args.checkpoint_interval,
        standard_output=args.standard_output,
        restore_checkpoint=args.restore_checkpoint
    )