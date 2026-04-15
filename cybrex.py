import importlib
import yaml

class Cybrex:
    def __init__(self):
        self._registry = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for module in self._registry.values():
            if hasattr(module, "stop"):
                module.stop()

    def register(self, capability: str, module):
        self._registry[capability] = module

    def get(self, capability: str):
        try:
            return self._registry[capability]
        except KeyError:
            raise KeyError(f"No module registered for capability '{capability}'")

    @classmethod
    def from_config(cls, path: str = "config.yaml") -> "Cybrex":
        api = cls()
        
        with open(path) as f:
            config = yaml.safe_load(f)

        for capability, module_name in config.items():
            mod = importlib.import_module(f"modules.{capability}.{module_name}")
            instance = mod.Module()
            api.register(capability, instance)

        return api