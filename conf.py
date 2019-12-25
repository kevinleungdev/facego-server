# -*- coding: utf-8 -*-

import ConfigParser
import os

# global configuration
config_parser = ConfigParser.ConfigParser()

config_filename = os.path.expandvars('conf.ini')

# load config file
config_parser.read(config_filename)


def get_prop(section, option):
    return config_parser.get(section, option)
