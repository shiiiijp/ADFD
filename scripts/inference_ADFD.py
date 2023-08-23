import os, sys
import json
import pprint

sys.path.append(".")
sys.path.append("..")

from options.test_options import TestOptions
from utils.ADFD import ADFD


def main():
	opts = TestOptions().parse()
	os.makedirs(opts.exp_dir, exist_ok=True)

	coach = ADFD(opts)
	coach.diversify()

if __name__ == '__main__':
	main()
