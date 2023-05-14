import os
import json
import sys
import pprint

sys.path.append(".")
sys.path.append("..")

from options.test_options import TestOptions
from guided_optimization import guided_optimization

def main():
    opts = TestOptions().parse()
    os.makedirs(opts.exp_dir, exist_ok=True)
    
    opts_dict = vars(opts)
    pprint.pprint(opts_dict)
    with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
        json.dump(opts_dict, f, indent=4, sort_keys=True)

    low_age_transformation = guided_optimization(opts)
    low_age_transformation.optimize()

    if __name__ == '__main__':
        main()
