import warnings
from collections import OrderedDict
from os import path
from os.path import basename, splitext

import yaml
from easydict import EasyDict


class SectionDict(EasyDict):
    def __init__(self, section, dict_value=None, **kwargs):
        self.__verbose__ = False
        self.__section__ = section
        super().__init__(dict_value, **kwargs)
        self.__verbose__ = True

    def __setattr__(self, name, value):
        if not (name.startswith('__') and name.endswith('__')):
            if isinstance(value, (list, tuple)):
                value = [EasyDict(x) if isinstance(x, dict) else x for x in value]
            elif isinstance(value, dict) and not isinstance(value, self.__class__):
                value = EasyDict(value)
            if self.__verbose__:
                if name in self:
                    print('Config operation "change", section: %s, option: %s, value: %s => %s'
                          % (self.__section__, name, str(getattr(self, name)), str(value)))
                else:
                    print('Config operation "adding", section: %s, option: %s, value: %s'
                          % (self.__section__, name, str(value)))
            super(EasyDict, self).__setattr__(name, value)
            super(EasyDict, self).__setitem__(name, value)
        else:
            super(EasyDict, self).__setattr__(name, value)


class Configurer():
    def __init__(self, config_file, verbose=False):
        config = OrderedDict()
        with open(config_file, 'r') as cf:
            config.update(yaml.safe_load(cf))
        self.path = path.abspath(config_file)
        self.name = splitext(basename(config_file))[0]
        config["TEMP"] = OrderedDict({"config_path": self.path, "config_name": self.name})
        self.sections = tuple(config.keys())
        self.DATASET = SectionDict(section="DATASET", dict_value=config.pop("DATASET"))
        self.HPARAM = SectionDict(section="HPARAM", dict_value=config.pop("HPARAM"))
        self.TRAIN = SectionDict(section="TRAIN", dict_value=config.pop("TRAIN"))
        self.PROCESSING = SectionDict(section="PROCESSING", dict_value=config.pop("PROCESSING"))
        self.OTHER = SectionDict(section="OTHER", dict_value=config.pop("OTHER"))
        self.TEMP = SectionDict(section="TEMP", dict_value=config.pop("TEMP"))
        for sec in config.keys():
            setattr(self, sec, SectionDict(section=sec, dict_value=config[sec]))
        self.config_sanity_check()
        self.verbose = verbose
        self.is_initialized = True

    def config_sanity_check(self):
        for sec in self.sections:
            assert sec.isupper()
        if hasattr(self, 'DATASET'):
            assert self.DATASET.color_mode in ['gray', 'rgb']
            if hasattr(self.DATASET, 'normalization'):
                assert isinstance(self.DATASET.normalization, str)

    def __setattr__(self, name, value):
        if hasattr(self, 'is_initialized') and self.is_initialized:
            with warnings.catch_warnings():
                warnings.simplefilter("always")
                warnings.warn('Nothing has changed in the configuration. '
                              'If you want to modify any item in configure, please specify the "section" name!')
        super().__setattr__(name, value)

    def print_configuration(self):
        print('{0:+^64}'.format('CFG'))
        print(self)
        print('{0:+^64}'.format('CFG'))

    def __repr__(self):
        str_list = []
        for sec_name in self.sections:
            str_list.append('{0:-^32}'.format(sec_name))
            for k, v in getattr(self, sec_name).items():
                if v is not None:
                    str_list.append('%s: %s' % (str(k), str(v)))
        return '\n'.join(str_list)

    def __iter__(self):
        for _, value in self.__dict__.items():
            if isinstance(value, SectionDict):
                yield from value


if __name__ == '__main__':
    cp = Configurer('DEFAULT.yaml', verbose=True)
    cp.print_configuration()
    cp.config_sanity_check()
    cp.pattern_id = 3
    cp.test_patch_size = 2
    pass
