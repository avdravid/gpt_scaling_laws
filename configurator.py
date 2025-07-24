"""
Poor Man's Configurator. Probably a terrible idea. Example usage:
$ python train.py config/override_file.py --batch_size=32
this will first run config/override_file.py, then override batch_size to 32

The code in this file will be run as follows from e.g. train.py:
>>> exec(open('configurator.py').read())

So it's not a Python module, it's just shuttling this code away from train.py
The code in this script then overrides the globals()

I know people are not going to love this, I just really dislike configuration
complexity and having to prepend config. to every single variable. If someone
comes up with a better simple Python solution I am all ears.
"""

import sys
from ast import literal_eval

for arg in sys.argv[1:]:
    if '=' not in arg:
        # assume it's the name of a config file
        assert not arg.startswith('--')
        config_file = arg
        print(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            print(f.read())
        exec(open(config_file).read())
    else:
        # assume it's a --key=value argument
        assert arg.startswith('--')
        key, val = arg.split('=')
        key = key[2:]
        if key in globals():
            try:
                # attempt to eval it it (e.g. if bool, number, or etc)
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                # if that goes wrong, just use the string
                attempt = val
            # ensure the types match ok
            assert type(attempt) == type(globals()[key])
            # cross fingers
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")
            
# when scaling, update variables that depend on other variables, such as N, learning rate, etc.
scaling = globals()['scaling']
assert scaling in ['', 'Kaplan', 'Chinchilla-1', 'Chinchilla-2'], "invalid scaling_type"

if scaling == 'Kaplan':
    import math
    # retrieve scale_N, scale_D, estimate_B_crit from globals(); assert that at least one of them is True when scaling=Kaplan
    scale_N, scale_D, estimate_B_crit = globals()['scale_N'], globals()['scale_D'], globals()['estimate_B_crit']
    assert scale_N or scale_D or estimate_B_crit, "at least one of scale_N, scale_D, or estimate_B_crit must be True when scaling=Kaplan."
    
    # retrieve n_layer, n_embd, fraction_of_data; and compute N and D
    n_layer, n_embd, fraction_of_data = globals()['n_layer'], globals()['n_embd'], globals()['fraction_of_data']
    N = 12 * n_layer * n_embd**2 
    D = int(fraction_of_data*9035582198)
    
    # Set learning rate, wandb-run-name and out-dir name; these depend on N and/or D
    if scale_N or scale_D:
        learning_rate =  0.003239 - 0.0001395 * math.log(N) # use equation D.1 of Kaplan et al to set maximum learning rate
        wandb_run_name = f'D-{D:.2e}-N-{N:.2e}' if scale_D else f'N-{N:.2e}'
    else: # when estimating critical batch size, learning rate is set by hand so we do not need to compute it
        wandb_run_name = f"N-{N:.2e}-batch-{globals()['batch_size']*globals()['gradient_accumulation_steps']}-lr-{globals()['learning_rate']}"
    
    #out_dir = 'out-'+wandb_run_name

elif scaling == 'Chinchilla-1' or scaling == 'Chinchilla-2':
    raise NotImplementedError("work in progress")
