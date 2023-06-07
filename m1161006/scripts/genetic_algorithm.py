import random
import json
import os
import copy
import numpy as np

file_path = os.path.dirname(os.path.abspath(__file__))

class GeneticAlgorithm(object):
    """ Genetic algorithm framework that handles robot params, full generations,
    saving, crossover/mutations, and fitness function calculation.
    """

    def __init__(self, max_time, load=True):
        self.max_time = max_time
        self.generation_size = 100
        self.generation_num = 10
        self.generation = []

        # used to test only the best out of a generation for our final results
        self.test_best = False

        self.n_crossover_mean = self.generation_size * 2
        self.n_crossover_swap = self.generation_size * 2
        self.n_mutation = self.generation_size * 2

        # all the predator params
        self.param_keys = [
            "prey_weight",
            "parallel_weight",
            "away_weight",
            "prey_only_pixel_percent",
            "min_turn_only_angle",
            "base_speed",
            "scaled_speed",
            "angle_adjust_rate"
        ]

        if load:
            self.load(self.generation_num)
            if self.test_best:
                self.reduce_to_best()

            self.choose_next()
        
        self.required_tries = 3
        if self.test_best:
            self.required_tries = 10

        self.try_count = 0
        self.capture_count = 0


    def get_filename(self, generation_num):
        return f'generation_{generation_num}.json'


    def random_params(self):
        params = {}
        for key in self.param_keys:
            params[key] = np.random.uniform()
        return params


    def init_and_save_generation(self):
        """ Saves a randomly generated generation to start out
        """
        for i in range(self.generation_size):
            v = {
                "params": self.random_params(),
                "score": 1 / self.generation_size,
                "tested": False
            }
            self.generation.append(v)

        self.save()


    def count_tested(self):
        c = 0
        for v in self.generation:
            if v["tested"]:
                c += 1
        return c


    def print_progress(self):
        tested = self.count_tested()
        size = self.generation_size
        print(f'Generation {self.generation_num} Progress: {tested}/{size}')


    def set_subject(self):
        """ The 'subject' is the current parameter set that we are testing for the predator
        """
        for v in self.generation:
            if not v["tested"]:
                self.subject = v
                return True
        
        return False


    def get_params(self):
        return self.subject["params"]


    def set_score_by_capture(self, captured):
        """ Fitness function to set the score of predator params
        """
        self.try_count += 1
        if captured:
            self.capture_count += 1

        if self.try_count == self.required_tries:
            if self.capture_count == 0:
                self.set_score(0.05)
            else:
                self.set_score(self.capture_count / self.try_count)


    def set_score_by_time(self, time):
        time_param = self.max_time * 1.03
        score = min(1, max(0, (time_param - time) / time_param))
        self.set_score(score)


    def choose_next(self):
        """ Choose the next subject and reset the try counter for determining fitness
        """
        self.try_count = 0
        self.capture_count = 0

        if not self.set_subject():
            if self.test_best:
                exit()

            self.generate_next_generation()
            self.set_subject()


    def set_score(self, score):
        self.subject["score"] = score
        self.subject["tested"] = True

        print(f'Score: {score}')

        self.save()

        self.choose_next()


    def regularize_scores(self):
        # scores should total 1 like in particle filter
        scores_tot = 0

        for v in self.generation:
            scores_tot += v["score"]
        
        for v in self.generation:
            v["score"] = v["score"] / scores_tot


    def crossover_mean(self, v1, v2):
        key = np.random.choice(self.param_keys)
        p1 = v1["params"][key]
        p2 = v2["params"][key]
        avg = (p1 + p2) / 2
        v1["params"][key] = avg
        v2["params"][key] = avg


    def crossover_swap(self, v1, v2):
        key = np.random.choice(self.param_keys)
        temp = v1["params"][key]
        v1["params"][key] = v2["params"][key]
        v2["params"][key] = temp


    def mutation(self, v):
        key = np.random.choice(self.param_keys)
        v["params"][key] += np.random.uniform(-.1, .1)


    def generate_next_generation(self):
        """ Create the next generation using the current, doing copying, picking new gen
        based on scores from fitness function, then doing crossover/mutation
        """
        self.save()
        self.regularize_scores()

        self.generation_num += 1
        # randomly select based on scores, then do mutation and crossover
        probs = list(map(lambda x: x["score"], self.generation))

        next_generation = np.random.choice(self.generation, size=self.generation_size, p=probs).tolist()

        for i in range(len(next_generation)):
            next_generation[i] = copy.deepcopy(next_generation[i])
            next_generation[i]["tested"] = False
            next_generation[i]["score"] = 1 / self.generation_size

        for i in range(self.n_crossover_mean):
            v1 = np.random.choice(next_generation)
            v2 = np.random.choice(next_generation)
            self.crossover_mean(v1, v2)
        
        for i in range(self.n_crossover_swap):
            v1 = np.random.choice(next_generation)
            v2 = np.random.choice(next_generation)
            self.crossover_swap(v1, v2)
        
        for i in range(self.n_mutation):
            v = np.random.choice(next_generation)
            self.mutation(v)

        self.generation = next_generation
        return next_generation


    def save(self):
        # we don't want to save anything when testing the best params
        if self.test_best:
            return

        filename = self.get_filename(self.generation_num)
        full_data = {
            "generation": self.generation,
            "generation_num": self.generation_num
        }
        with open(file_path + '/data/' + filename, 'w') as f:
            json.dump(full_data, f)


    def load(self, generation_num=0):
        filename = self.get_filename(generation_num)
        with open(file_path + '/data/' + filename, 'r') as f:
            data = json.load(f)
            self.generation = data["generation"]
            self.generation_num = data["generation_num"]


    def modify_gen(self):
        # simple helper made to reduce generation size to 100 from 200
        for v in self.generation:
            v["tested"] = False
        self.generation = self.generation[:100]


    def avg_capture_rate(self):
        tot = 0
        for v in self.generation:
            score = v["score"]
            if score != .05:
                tot += score
        
        return tot / self.generation_size
    

    def reduce_to_best(self):
        new_generation = []
        for v in self.generation:
            v["tested"] = False
            if v["score"] > 0.9:
                new_generation.append(v)
        self.generation = new_generation
        self.generation_size = len(self.generation)


if __name__ == '__main__':
    ga = GeneticAlgorithm(0, False)
    # ga.load(0)
    # ga.modify_gen()
    # ga.save()

    ga.init_and_save_generation()
    # ga.generate_next_generation()
    # ga.generate_next_generation()
