from neontology import BaseNode, BaseRelationship, init_neontology
from typing import ClassVar, Optional, List

class TeamMemberNode(BaseNode):
    __primaryproperty__: ClassVar[str] = "nickname"
    __primarylabel__: ClassVar[str] = "TeamMember"
    nickname: str