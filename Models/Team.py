from neontology import BaseNode, BaseRelationship, init_neontology
from typing import ClassVar, Optional, List

class Team(BaseNode):
    __primaryproperty__: ClassVar[str] = "teamname"
    __primarylabel__: ClassVar[str] = "Team"
    teamname: str
    slogan: str = "Better than the rest!"