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
