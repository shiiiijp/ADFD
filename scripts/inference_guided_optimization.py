import os, sys
import json
import pprint

sys.path.append(".")
sys.path.append("..")

from options.test_options import TestOptions
from utils.guided_optimization import GuidedOptimization


def main():
	opts = TestOptions().parse()
	os.makedirs(opts.exp_dir, exist_ok=True)

	coach = GuidedOptimization(opts)
	coach.optimize()

if __name__ == '__main__':
	main()
