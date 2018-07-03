import treenome
import random
import numpy as np

class Genome(object):
    """complete genome with body and brain and mutation operators"""
    def __init__(self, depth,
                       hidden_per_branch,
                       connectivity=0.5,
                       mag=3.0,
                       prob_weight=0.01,
                       prob_joint=0.05,
                       prob_expression=0.05,
                       joint_sigma=0.1):
        # set class attributes
        self.depth = depth
        self.nbranches = (2 ** (depth + 1)) - 1
        self.nleaves = 2 * depth
        self.nsensors = self.nleaves
        self.nmotors = self.nbranches
        self.nhidden = hidden_per_branch * self.nbranches
        self.nneurons = self.nsensors + self.nhidden + self.nmotors

        # set random joint ranges
        self.joint_ranges = [random.random() for i in range(self.nbranches)]
        # create random adjacency matrix
        self.adj_matrix = (np.random.random((self.nneurons, self.nneurons - self.nsensors)) * (2.0 * mag)) - mag

        # create matrix specify which weights to use
        self.weight_expression = np.random.random((self.nneurons, self.nneurons - self.nsensors))
        self.weight_expression[self.weight_expression < 1 - connectivity] = 0
        self.weight_expression[self.weight_expression >= 1 - connectivity] = 1

        # set more attributes
        self.prob_weight = prob_weight
        self.prob_expression = prob_expression
        self.prob_joint = prob_joint
        self.joint_sigma = joint_sigma

        self.neuron_positions = np.zeros(self.nneurons, dtype=np.int32)
        branch_ids = treenome.Tree.generate_branch_ids(depth)

        # sensor positions in relation to tree branch
        for i, index in enumerate(branch_ids[-1]):
            self.neuron_positions[i] = index

        # hidden positions
        for i in range(self.nbranches):
            branch_id = i % self.nbranches
            index = self.nsensors + (i * hidden_per_branch)
            # self.neuron_positions[index] = branch_id
            self.neuron_positions[index:(index + hidden_per_branch)] = [branch_id] * hidden_per_branch

        # motor positions
        self.neuron_positions[-self.nmotors:] = range(self.nmotors)

    @classmethod
    def from_config(cls, config):
        tree_params = config['Tree']
        network_params = config['Network']
        mutation_params = config['Mutation']

        genome = cls(depth=int(tree_params['depth']),
                     hidden_per_branch=int(network_params['hidden_per_branch']),
                     connectivity=float(network_params['connectivity']),
                     mag=float(network_params['magnitude']),
                     prob_weight=float(mutation_params['weight_probability']),
                     prob_joint=float(mutation_params['joint_probability']),
                     prob_expression=float(mutation_params['expression_probability']),
                     joint_sigma=float(mutation_params['joint_sigma'])
                     )

        return genome

    def mutate_weights(self):
        """mutates weight value"""
        r_weights = np.random.random((self.nneurons, self.nneurons - self.nsensors))
        indexes = np.where(r_weights < self.prob_weight)
        if len(indexes[0]) > 0:
            self.adj_matrix[indexes] += np.random.normal(0, np.abs(self.adj_matrix[indexes]), len(self.adj_matrix[indexes]))

    def mutate_expression(self):
        """mutates whether weight is expressed"""
        r_expression = np.random.random((self.nneurons, self.nneurons - self.nsensors))
        indexes = np.where(r_expression < self.prob_expression)
        if len(indexes[0]) > 0:
            # toggle mutation
            self.weight_expression[indexes] = np.abs(self.weight_expression[indexes] - 1)

    def mutate_joints(self):
        """mutates joints"""
        r_joints = np.random.random(self.nbranches)
        indexes = np.where(r_joints < self.prob_joint)
        if len(indexes[0]) > 0:
            # gaussian mutation of joint rang
            for index in indexes[0]:
                self.joint_ranges[index] += np.random.normal(0, self.joint_sigma)

    def mutate(self, weights=True, expression=True, joints=True):
        """mutates tree"""
        if weights:
            self.mutate_weights()
        if expression:
            self.mutate_expression()
        if joints:
            self.mutate_joints()

    def send_to_simulator(self, sim, tree):
        """sends tree to simulator"""
        tree.set_joint_ranges(self.joint_ranges)

        sensors = tree.send_to_simulator(sim)
        for sensor_type in sensors:
            for i, sensor in enumerate(sensors[sensor_type]):
                sim.send_sensor_neuron(i)

        for i in range(self.nhidden):
            sim.send_hidden_neuron()

        motors = []
        for i in range(self.nmotors):
            motors.append(sim.send_motor_neuron(i))


        rows, cols = np.shape(self.adj_matrix)

        for source in range(rows):
            for sink in range(cols):
                if self.weight_expression[source, sink] != 0:
                    sim.send_synapse(source, sink + self.nsensors, self.adj_matrix[source, sink])


    def calc_connection_cost(self):
        """total sum of connections"""
        return np.sum(self.weight_expression) / np.size(self.weight_expression)

    def calc_joint_cost(self):
        """average normalized joint range"""
        return np.average(self.joint_ranges)

    def calc_tree_joint_cost(self):
        """weighting joints based on effort"""
        ids_at_depth = treenome.Tree.generate_branch_ids(self.depth)

        weights = np.zeros(self.nbranches)
        branches_per_depth = np.zeros(self.nbranches)
        branches_per_depth[0] = self.nbranches
        for i in range(1, self.depth + 1):
            branches_per_depth[i] = (branches_per_depth[i-1] - 1) / 2

        for curr_depth in range(len(ids_at_depth)):
            for index in ids_at_depth[curr_depth]:
                weights[index] = branches_per_depth[curr_depth]

        cost = np.average(self.joint_ranges, weights=weights)
        return cost

    def calc_tree_connection_cost(self, branch_cost_matrix=None):
        """hierarchical weighting of connections"""

        if branch_cost_matrix == None:
            branch_cost_matrix = treenome.Tree.generate_tree_connection_cost(self.depth)

        rows, cols = np.shape(self.weight_expression)

        cost_matrix = np.zeros_like(self.weight_expression)

        for i in range(rows):
            for j in range(cols):
                source_pos = self.neuron_positions[i]
                sink_pos = self.neuron_positions[j + self.nsensors]
                cc = branch_cost_matrix[source_pos, sink_pos]
                if self.weight_expression[i, j] == 1:
                    cost_matrix[i, j] = cc
        cost = np.sum(cost_matrix)
        return cost

if __name__ == '__main__':
    import configtools
    import pyrosim
    import treenome

    config = configtools.config_from_file('base-tree.config')

    genome = Genome.from_config(config)
    tree_kwargs = configtools.tree_from_config(config)
    tree = treenome.Tree(**tree_kwargs)

    sim = pyrosim.Simulator(debug=False, play_paused=True)
    genome.send_to_simulator(sim, tree)

    sim.start()
    sim.wait_to_finish()
