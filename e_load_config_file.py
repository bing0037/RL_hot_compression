import yaml

# Different block size for different layers:
def load_config_file(config_file):
    with open(config_file, "r") as stream:
        try:
            raw_dict = yaml.load(stream)
            prune_ratios = raw_dict['prune_ratios']
        except yaml.YAMLError as exc:
            print(exc)
    return prune_ratios