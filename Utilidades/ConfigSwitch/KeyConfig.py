# from abc import 
from optparse import Values
from typing import Dict, List, Set
from ruamel.yaml import YAML, round_trip_dump as yaml_dump
from pathlib import Path
from ValueConfig import ValueConfig


class KeyConfig:

    name: str = None
    values: List[ValueConfig] = []
    user_values: List[str] = [] 

    def __init__(self, name: str, user_values: List[str]) -> None:
        self.name = name
        self.user_values = user_values.copy()
        self.values = []
        assert self.validate()
    
    def add_value(self, value: ValueConfig):
        print('ADDING', value.dump(), 'to ', self.name)
        self.values.append(value)
        print('AFTER', [k.dump() for k in self.values])
        assert self.validate()

    def validate(self):
        return True
        # users = [val.user for val in self.values]
        # users_set = set()
        # duplicates = [u for u in users if u in users_set or users_set.add(x)]    
        # assert len(duplicates) == 0, f'Los usuarios [{duplicates}] se repiten en las entradas de la clave {self.name}'

        # assert all([user in self.user_values for user in users_set])

    def dump(self):
        return {
            self.name: [value.dump() for value in self.values]
        }

    @classmethod
    def from_dict(cls, dict: dict) -> 'KeyConfig':
        assert len(dict.keys()) == 1, 'Se encontraron multiples claves para un diccionario de clave'
        
        name = list(dict.keys())[0]
        values = [ValueConfig(val) for val in dict[name]]        

        return KeyConfig(name, values)

    @classmethod
    def new_config(cls, key: str) -> 'KeyConfig':
        config = {
            key: []
        }

        return KeyConfig.from_dict(config)

class KeyGroupConfig:

    configs: Dict[str, KeyConfig] = {}

    def __init__(self, configs: dict) -> None:
        print('conf', configs)
        self.configs = {key: KeyConfig.from_dict(configs[key]) for key in configs.keys()}
        print('xd', self.configs)

    def has(self, key: str) -> None:
        return key in self.configs.keys()
    
    def add(self, key: str) -> None:
        if not self.has(key):
            self.configs[key] = KeyConfig.new_config(key)

    def get(self, key: str) -> KeyConfig:
        return self.configs[key]
    
    def add_kv(self, key: str, value: ValueConfig) -> None:
        print('ADDING', value, 'TO ', self.configs.get(key))
        print(self.configs.get(key))
        self.configs.get(key).add_value(value)

    def dump(self) -> List[dict]:
        # for key in self.configs.values():
        #     print(key)
        #     print(key.values)
        #     print('----------')

        return [key.dump() for key in self.configs.values()]
