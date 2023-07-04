class ValueConfig:

    user: str = None
    value: str = None

    def __init__(self, user: str, value: str) -> None:
        self.user = user
        self.value = value

    def dump(self):
        return {
            'value': self.value,
            'user': self.user
        }

    @classmethod
    def from_dict(cls, config: dict) -> 'ValueConfig':
        assert 'value' in dict.keys()
        assert 'user' in dict.keys()

        user = config['user']
        value = config['value']

        return ValueConfig(user, value)

    @classmethod
    def new_config(cls) -> 'ValueConfig':
        config = {
            'value': None,
            'user': None
        }

        return ValueConfig.from_dict(config)
