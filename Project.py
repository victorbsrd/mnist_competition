from pathlib import Path
from dataclasses import dataclass

@dataclass
class Project:
    """
    this class represents our project. It stores usefull information for the project.
    """
    base_dir: Path = Path(__file__).parents[0]
    data_dir = base_dir / 'dataset'
    checkpoint_dir = base_dir / 'checkpoint'
    savings_dir = base_dir / 'saves'

    def __post_init__(self):
        #creates directiory if it doesn't exists yet
        self.data_dir.mkdir(exist_ok = True)
        self.checkpoint_dir.mkdir(exist_ok = True)
        self.savings_dir.mkdir(exist_ok = True)

project = Project()
