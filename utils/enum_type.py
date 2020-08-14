from enum import Enum


class ModelType(Enum):
    GENERAL = 1
    SEQUENTIAL = 2
    CONTEXT = 3
    KNOWLEDGE = 4
    SOCIAL = 5


class DataLoaderType(Enum):
    NEGSAMPLE = 1
    FULL = 2


# can not use Enum Type , Enum Type can't be store in config object
class EvaluatorType:
    RANKING = 'ranking'
    INDIVIDUAL = 'loss'


class InputType(Enum):
    POINTWISE = 1
    PAIRWISE = 2
    LISTWISE = 3
