import os
import configparser

######################################################################
# Default configuration: read or create config file
config = configparser.ConfigParser()
config_file = os.sep.join((os.environ['HOME'], '.config', 'dust', 'config.ini'))
if os.path.exists(config_file):
    config.read(config_file)
else:
    path = os.sep.join((os.environ['HOME'], 'Projects', 'src', 'thermal-dust-model', 'data'))
    config['fit-idl-save'] = {'Path': path}
    del path

    i = 0
    while i >= 0:
        if i != 0:  # no need to create a root directory!
            if not os.path.isdir(config_file[:i]):
                os.mkdir(config_file[:i])
        i = config_file.find(os.sep, i + 1)

    with open(config_file, 'w') as outf:
        config.write(outf)

del config_file
del configparser
del os
