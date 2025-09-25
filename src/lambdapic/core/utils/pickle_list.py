from numba import typed

class PickleableTypedList:
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        for k in state:
            if k.endswith('_list'):
                if isinstance(state[k], list):
                    continue
                if not isinstance(state[k], typed.typedlist.List):
                    raise TypeError(f"{k} of {self.__class__.__name__} is not a numba typed list")
                state[k] = list(state[k])
        return state
    
    def __setstate__(self, state: dict):
        for k in state:
            if k.endswith('_list'):
                state[k] = typed.typedlist.List(state[k])
        self.__dict__.update(state)