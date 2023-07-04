# from abc import 
from enum import Enum, auto
from mimetypes import init
from multiprocessing.sharedctypes import Value
from optparse import Values
from typing import Dict, List, Set
from ruamel.yaml import YAML, round_trip_dump as yaml_dump
import sys
import os
import pwd
import getpass
from pathlib import Path

from KeyConfig import KeyGroupConfig
from ValueConfig import ValueConfig

yaml = YAML(typ='safe')

class UserConfig:

    __storable = False

    def __init__(self) -> None:
        pass

class UserConfig:

    __storable = False
    alias: str = None
    usernames: List[str] = []

    def __init__(self, dict: dict) -> None:
        if dict:
            self.load_dict(dict)

    def load_dict(self, dict: dict):
        assert len(dict.keys()) == 1, 'Se encontraron multiples claves para un diccionario de usuario'
        assert type(dict.values()[0]) == list, 'Sección de usuarios no es una lista'


        self.alias = dict.keys()[0]
        self.usernames = list(set(dict[self.alias]))
        self.__storable = True

    def dump(self):
        return {
            self.alias: self.usernames
        }

    @classmethod
    def new_config(cls, alias: str) -> 'UserConfig':
        config = {
            alias: []
        }

        return UserConfig().load_dict(config)


class Config:

    __storable = False
    file_dest = 'switchconf.yml'
    keys: KeyGroupConfig = None
    users: List[str] = []

    def __init__(self) -> None:
        pass

    def dump(self):
        return {
            'Keys': self.keys.dump(),
            'Users': self.users
        }

    def load_dict(self, dict: dict):
        self.keys = KeyGroupConfig(dict.get('Keys'))
        self.users = dict.get('Users')
        self.__storable = True

        return self

    def add_user(self, user: str):
        if user not in (self.users.keys() or []):
            self.users[user] = [user]

    def add_user_alias(self, user: str, alias: str):
        assert user in self.users.keys(), f'El usuario ({user}) no existe'
        self.users[user].append(alias)
    
    def add_key(self, key: str) -> None:
        self.keys.add(key)

    def add_kv(self, key: str, path: str, user: str):
        self.keys.add_kv(key, ValueConfig(user, path))
        # assert self.keys.has(key), f'La clave del recurso ({key}) no existe'
        # assert user in self.users.keys(), f'El usuario ({user}) no existe'
        # self.keys[key].append({
        #     'path': path,
        #     'user': user
        # })

    @classmethod
    def new_config(cls) -> 'Config':
        config = {
            'Keys': {},
            'Users': {}
        }

        return Config().load_dict(config)

    @classmethod
    def from_dict(cls, config: dict) -> 'Config':
        assert 'Keys' in config.keys() and type(config.get('Keys')) == dict 
        assert 'Users' in config.keys() and type(config.get('Users')) == dict

        return Config().load_dict(config)

    def gen(self, path: str = None): 
        data = yaml_dump(self.dump())
        if not path:
            path = self.file_dest
        file = open(path, "w")
        file.write(data)
        
        file.close()

    @classmethod
    def from_file(cls, path: str):
        raw_data = Path.read_text(path).replace('\n', '')
        data = yaml.load(raw_data)
        return Config.from_dict(data)


class ConfigSwitch:

    DEFAULT_CONFIG_FILENAME = 'SwitchFile.yaml'
    current_user: str = None
    
    class UserSelection(Enum):
        USERNAME_EXPLICIT = auto()
        ALIAS_EXPLICIT = auto()
        AUTO = auto()

    def __init__(self, user_key: str = None, selection: UserSelection = UserSelection.AUTO) -> None:
        self.current_user 


conf = Config.new_config()
conf.add_user('alam')
conf.add_user('jon')

# conf.gen()

conf.add_user_alias('alam', 'aast')
conf.add_user_alias('alam', 'aast1')
# conf.gen()

conf.add_key('total')
conf.add_key('test')
print(conf.keys.dump())

conf.add_kv('total', '/path/to', 'alam')
conf.add_kv('total', '/path/to', 'jon')
conf.add_kv('test', '/path/to', 'jon')
conf.gen()

# print(getpass.getuser())

# Switchconfig(currentuser='alam' | AUTO )
# get_resource('useralias', 'key')
# if not: your current user does not belong to any alias, add to one? yes / no
# if not: create new user with current linux usernames
# get_resource_('key')