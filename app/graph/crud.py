from datetime import datetime, timezone
from typing import Optional

# Import modules from FastAPI
from fastapi import APIRouter, Depends, HTTPException, status

# Import internal utilities for database access, authorisation, and schemas
from app.utils.db import neo4j_driver
from app.authorisation.auth import get_current_active_user
from app.utils.schema import User, Node, Nodes, Relationship

# Set the API Router
router = APIRouter()

# List of acceptable node labels and relationship types
# Modify these to add constraints
query = "CALL db.labels()"
result = neo4j_driver.session().run(query=query)
data = result.data()
node_labels = []
for label in data:
    node_labels.append(label['label'])

query = "CALL db.relationshipTypes()"
result = neo4j_driver.session().run(query=query)
data = result.data()
relationship_types = data

# Used for validation to ensure they are not overwritten
base_properties = ['created_by', 'created_time']


# GRAPH PROPERTIES
# Labels
@router.get('/labels')
async def get_all_labels():
    query = "CALL db.labels()"
    with neo4j_driver.session() as session:
        result = session.run(query=query)
        data = result.data()
    return data


# Property Keys
@router.get('/propertykeys')
async def get_all_property_keys():
    query = "CALL db.propertyKeys()"
    with neo4j_driver.session() as session:
        result = session.run(query=query)
        data = result.data()
    return data


# Properties of Label
@router.get('/propertykeys/{label}')
async def get_all_property_keys(label: str):
    query = f"""MATCH (n:{label})
        WITH n LIMIT 25
        UNWIND keys(n) as key
        RETURN distinct key"""
    # renvoie la liste des propriétés/champ de donnée des ressources (c'est à dire les noeuds ayant le label ns0__record dans le cas du projet HUMANE)
    with neo4j_driver.session() as session:
        result = session.run(query=query, )
        data = result.data()
    return data


# NODES
from pydantic import BaseModel


# CREATE new node
@router.post('/create_node', response_model=Node)
async def create_node(label: str, node_attributes: dict,
                      current_user: User = Depends(get_current_active_user)):
    # Check that node is not User
    if label == 'User':
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Operation not permitted, cannot create a User with this method.",
            headers={"WWW-Authenticate": "Bearer"})

    print(label)
    print(node_labels)
    # Check that node has an acceptable label
    if label not in node_labels:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Operation not permitted, node label is not accepted.",
            headers={"WWW-Authenticate": "Bearer"})

    # Check that attributes dictionary does not modify base fields
    for key in node_attributes:
        if key in base_properties:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                                detail="Operation not permitted, you cannot modify those fields with this method.",
                                headers={"WWW-Authenticate": "Bearer"})

    unpacked_attributes = ""
    for (key, value) in node_attributes.items():
        if value != None:
            unpacked_attributes += f"\nSET new_node.{key} = "
            if isinstance(value, str):
                # protected_string = original_string.replace("'", "\\'")
                value = value.replace("'", "\\'")
                unpacked_attributes += f"\'{value}\' "
            else:
                unpacked_attributes += f"{value} "

    # unpacked_attributes = 'SET ' + ', '.join(f'new_node.{key}=\'{value}\'' for (key, value) in node_attributes.items())

    cypher = f"CREATE (new_node:{label}) "
    cypher += f"\nSET new_node.created_by = $created_by "
    cypher += f"\nSET new_node.created_time = $created_time "
    cypher += f"{unpacked_attributes}"
    cypher += f"\nRETURN new_node, LABELS(new_node) as labels, ID(new_node) as id"

    print(cypher)

    with neo4j_driver.session() as session:
        result = session.run(
            query=cypher,
            parameters={
                'created_by': current_user.username,
                'created_time': str(datetime.now(timezone.utc)),
                'attributes': node_attributes,
            },
        )

        node_data = result.data()[0]

    return Node(node_id=node_data['id'],
                labels=node_data['labels'],
                properties=node_data['new_node'])


# READ data about a node in the graph by ID
@router.get('/read/{node_id}', response_model=Node)
async def read_node_id(node_id: int, current_user: User = Depends(get_current_active_user)):
    """
    **Retrieves data about a node in the graph, based on node ID.**

    :param **node_id** (str) - node id, used for indexed search

    :returns: Node response, with node id, labels, and properties.
    """

    cypher = """
    MATCH (node)
    WHERE ID(node) = $node_id
    RETURN ID(node) as id, LABELS(node) as labels, node
    """

    with neo4j_driver.session() as session:
        result = session.run(query=cypher,
                             parameters={'node_id': node_id})

        node_data = result.data()[0]

    # Check node for type User, and send error message if needed
    if 'User' in node_data['labels']:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Operation not permitted, please use User endpoints to retrieve user information.",
            headers={"WWW-Authenticate": "Bearer"})

    # Return Node response
    return Node(node_id=node_data['id'],
                labels=node_data['labels'],
                properties=node_data['node'])


# READ data about a collection of nodes in the graph
@router.get('/read_node_collection', response_model=Nodes)
async def read_nodes(search_node_property: str, node_property_value: str,
                     current_user: User = Depends(get_current_active_user)):
    """
    Retrieves data about a collection of nodes in the graph, based on node property.

    :param **node_property** (str) - property to search in nodes

    :param **node_property_value** (str) - value of property, to select the correct node

    :returns: Node response, with node id, labels, and properties. Returns only first response.
    """

    cypher = f"""
        MATCH (node)
        WHERE node.{search_node_property} = '{node_property_value}'
        RETURN ID(node) as id, LABELS(node) as labels, node;
        """

    with neo4j_driver.session() as session:
        result = session.run(query=cypher)

        collection_data = result.data()

    node_list = []
    for node in collection_data:
        # Create node for each result in query
        node = Node(node_id=node['id'],
                    labels=node['labels'],
                    properties=node['node'])

        # Append each node result into Nodes list
        node_list.append(node)

    # Return Nodes response with collection as list
    return Nodes(nodes=node_list)


# UPDATE properties of node in the graph
@router.put('/update/{node_id}')
async def update_node(node_id: int, attributes: dict):
    # Check that property to update is not part of base list
    for key in attributes:
        if key in base_properties:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Operation not permitted, that property field cannot be updated.",
                headers={"WWW-Authenticate": "Bearer"})

    cypher = '''MATCH (node) WHERE ID(node) = $id
                SET node += $attributes
                RETURN node, ID(node) as id, LABELS(node) as labels'''

    with neo4j_driver.session() as session:
        result = session.run(query=cypher,
                             parameters={'id': node_id, 'attributes': attributes})

        node_data = result.data()[0]

    # Return Node response
    return Node(node_id=node_data['id'],
                labels=node_data['labels'],
                properties=node_data['node'])


# DELETE node in the graph
@router.post('/delete/{node_id}')
async def delete_node(node_id: int):
    cypher = """
    MATCH (node)
    WHERE ID(node) = $node_id
    DETACH DELETE node
    """

    with neo4j_driver.session() as session:
        result = session.run(query=cypher,
                             parameters={'node_id': node_id})

        node_data = result.data()

    # Confirm deletion was completed by empty response
    return node_data or {
        'response': f'Node with ID: {node_id} was successfully deleted from the graph.'
    }


# RELATIONSHIPS
# Create new relationship between two nodes
@router.post('/create_relationship', response_model=Relationship)
async def create_relationship(attributes: dict, current_user: User = Depends(get_current_active_user)):
    relationship_type = attributes['relationship_type']
    print(relationship_type)
    relationship_attributes = attributes['relationship_attributes']
    print(relationship_attributes)
    source_node = attributes['source_node']
    print(source_node)
    target_node = attributes['target_node']
    print(target_node)

    # Check that relationship has an acceptable type
    # if relationship_type not in relationship_types:
    #     raise HTTPException(
    #         status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
    #         detail="Operation not permitted, relationship type is not accepted.",
    #         headers={"WWW-Authenticate": "Bearer"})

    # Check that attributes dictionary does not modify base fields
    for key in relationship_attributes:
        if key in base_properties:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                                detail="Operation not permitted, you cannot modify those fields with this method.",
                                headers={"WWW-Authenticate": "Bearer"})

    if relationship_attributes:
        unpacked_attributes = 'SET ' + ', '.join(
            f'relationship.{key}=\'{value}\'' for (key, value) in relationship_attributes.items())
        unpacked_attributes += '\n'
    else:
        unpacked_attributes = ''

    cypher = f"""MATCH (nodeA:{source_node["label"]}) WHERE id(nodeA) = $nodeA_ID\n"""
    cypher += f"""MATCH (nodeB:{target_node["label"]}) WHERE id(nodeB) = $nodeB_ID\n"""
    cypher += f"""CREATE (nodeA)-[relationship:{relationship_type}]->(nodeB)\n"""
    cypher += f"""SET relationship.created_by = $created_by\n"""
    cypher += f"""SET relationship.created_time = $created_time\n"""
    cypher += f"{unpacked_attributes}"
    cypher += (f"RETURN nodeA, nodeB, LABELS(nodeA), LABELS(nodeB), ID(nodeA), ID(nodeB), ID(relationship), "
               f"TYPE(relationship), PROPERTIES(relationship)\n")

    with neo4j_driver.session() as session:
        result = session.run(
            query=cypher,
            parameters={
                'created_by': current_user.username,
                'created_time': str(datetime.now(timezone.utc)),
                'nodeA_ID': source_node["id"],
                'nodeB_ID': target_node["id"],
            },
        )

        relationship_data = result.data()
    print("result data: ")
    print(relationship_data)
    if len(relationship_data) > 0:
        relationship_data = relationship_data[0]

        # Organise the data about the nodes in the relationship
        source_node = Node(node_id=relationship_data['ID(nodeA)'],
                           labels=relationship_data['LABELS(nodeA)'],
                           properties=relationship_data['nodeA'])

        target_node = Node(node_id=relationship_data['ID(nodeB)'],
                           labels=relationship_data['LABELS(nodeB)'],
                           properties=relationship_data['nodeB'])

        # Return Relationship response
        return Relationship(relationship_id=relationship_data['ID(relationship)'],
                            relationship_type=relationship_data['TYPE(relationship)'],
                            properties=relationship_data['PROPERTIES(relationship)'],
                            source_node=source_node,
                            target_node=target_node)
    else:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Error while creating the relationship.",
                            headers={"WWW-Authenticate": "Bearer"})

# READ data about a relationship
@router.get('/read_relationship/{relationship_id}', response_model=Relationship)
async def read_relationship(relationship_id: int):
    cypher = """
        MATCH (nodeA)-[relationship]->(nodeB)
        WHERE ID(relationship) = $rel_id
        RETURN nodeA, ID(nodeA), LABELS(nodeA), relationship, ID(relationship), TYPE(relationship), nodeB, ID(nodeB), LABELS(nodeB), PROPERTIES(relationship)
        """

    with neo4j_driver.session() as session:
        result = session.run(query=cypher,
                             parameters={'rel_id': relationship_id})

        relationship_data = result.data()[0]

    # Organise the data about the nodes in the relationship
    source_node = Node(node_id=relationship_data["ID(nodeA)"],
                       labels=relationship_data["LABELS(nodeA)"],
                       properties=relationship_data["nodeA"])

    target_node = Node(node_id=relationship_data["ID(nodeB)"],
                       labels=relationship_data["LABELS(nodeB)"],
                       properties=relationship_data["nodeB"])

    # Return Relationship response
    return Relationship(relationship_id=relationship_data["ID(relationship)"],
                        relationship_type=relationship_data["TYPE(relationship)"],
                        properties=relationship_data["PROPERTIES(relationship)"],
                        source_node=source_node,
                        target_node=target_node)


@router.get('/read_relationship_node_label/')
async def read_relationship_node_label(node_id: int, _label: str):
    cypher = f"""
        MATCH (nodeA)-[relationship]->(n2:{_label})
        WHERE id(nodeA) = {node_id}
        RETURN nodeA, ID(nodeA), LABELS(nodeA), relationship, ID(relationship), TYPE(relationship), PROPERTIES(relationship)
        """

    with neo4j_driver.session() as session:
        result = session.run(query=cypher)

        relationship_data = result.data()

    if len(relationship_data) < 0:
        return -1

    # Organise the data about the nodes in the relationship
    source_node = Node(node_id=relationship_data["ID(nodeA)"],
                       labels=relationship_data["LABELS(nodeA)"],
                       properties=relationship_data["nodeA"])

    target_node = Node(node_id=relationship_data["ID(nodeB)"],
                       labels=relationship_data["LABELS(nodeB)"],
                       properties=relationship_data["nodeB"])

    # Return Relationship response
    return Relationship(relationship_id=relationship_data["ID(relationship)"],
                        relationship_type=relationship_data["TYPE(relationship)"],
                        properties=relationship_data["PROPERTIES(relationship)"],
                        source_node=source_node,
                        target_node=target_node)


@router.get('/read_relationship_btwn_node/')
async def read_relationship_btwn_node(node_id1: int, node_id2: int):
    cypher = """
        MATCH (nodeA)-[relationship]->(nodeB)
        WHERE id(nodeA) = $node1_id AND id(nodeB) = $node2_id
        RETURN nodeA, ID(nodeA), LABELS(nodeA), relationship, ID(relationship), TYPE(relationship), nodeB, ID(nodeB), LABELS(nodeB), PROPERTIES(relationship)
        """

    with neo4j_driver.session() as session:
        result = session.run(query=cypher,
                             parameters={
                                 'node1_id': node_id1,
                                 'node2_id': node_id2,
                             })

        relationship_data = result.data()

    if len(relationship_data) > 0:
        relationship_data = relationship_data[0]
    else:
        return -1

    # Organise the data about the nodes in the relationship
    source_node = Node(node_id=relationship_data["ID(nodeA)"],
                       labels=relationship_data["LABELS(nodeA)"],
                       properties=relationship_data["nodeA"])

    target_node = Node(node_id=relationship_data["ID(nodeB)"],
                       labels=relationship_data["LABELS(nodeB)"],
                       properties=relationship_data["nodeB"])

    # Return Relationship response
    return Relationship(relationship_id=relationship_data["ID(relationship)"],
                        relationship_type=relationship_data["TYPE(relationship)"],
                        properties=relationship_data["PROPERTIES(relationship)"],
                        source_node=source_node,
                        target_node=target_node)


# Update data about a relationship
@router.put('/update_relationship/{relationship_id}', response_model=Relationship)
async def update_relationship(relationship_id: int, attributes: dict):
    cypher = """
    MATCH (nodeA)-[relationship]->(nodeB)
    WHERE ID(relationship) = $rel_id
    SET relationship += $attributes
    RETURN nodeA, ID(nodeA), LABELS(nodeA), relationship, ID(relationship), TYPE(relationship), nodeB, ID(nodeB), LABELS(nodeB), PROPERTIES(relationship)
    """

    with neo4j_driver.session() as session:
        result = session.run(query=cypher,
                             parameters={'rel_id': relationship_id,
                                         'attributes': attributes})

        relationship_data = result.data()[0]

    # Organise the data about the nodes in the relationship
    source_node = Node(node_id=relationship_data['ID(nodeA)'],
                       labels=relationship_data['LABELS(nodeA)'],
                       properties=relationship_data['nodeA'])

    target_node = Node(node_id=relationship_data['ID(nodeB)'],
                       labels=relationship_data['LABELS(nodeB)'],
                       properties=relationship_data['nodeB'])

    # Return Relationship response
    return Relationship(relationship_id=relationship_data['ID(relationship)'],
                        relationship_type=relationship_data['TYPE(relationship)'],
                        properties=relationship_data['PROPERTIES(relationship)'],
                        source_node=source_node,
                        target_node=target_node)


# DELETE relationship in the graph
@router.post('/delete_relationship/{relationship_id}')
async def delete_relationship(relationship_id: int):
    cypher = """
            MATCH (nodeA)-[relationship]->(nodeB)
            WHERE ID(relationship) = $rel_id
            RETURN nodeA, ID(nodeA), LABELS(nodeA), relationship, ID(relationship), TYPE(relationship), nodeB, ID(nodeB), LABELS(nodeB), PROPERTIES(relationship)
            """

    with neo4j_driver.session() as session:
        result = session.run(query=cypher,
                             parameters={'rel_id': relationship_id})

        relationship_data = len(result.data())

    if relationship_data == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Operation impossible, relationship doesn't exist.",
            headers={"WWW-Authenticate": "Bearer"})

    cypher = """
        MATCH (a)-[relationship]->(b)
        WHERE ID(relationship) = $relationship_id
        DELETE relationship
        """

    with neo4j_driver.session() as session:
        result = session.run(query=cypher,
                             parameters={'relationship_id': relationship_id})

        relationship_data = result.data()

    # Confirm deletion was completed by empty response
    return relationship_data or {
        'response': f'Relationship with ID: {relationship_id} was successfully deleted from the graph.'
    }

@router.post('/delete_all_relationship/{node_id}', summary="Delete all relationship of the node of given ID, Node Label in request body")
async def delete_all_relationship(node_id: int, attributes: dict):
    cypher = f"""
        MATCH (n:{attributes['label']})-[r]-()
        WHERE ID(n) = $node_id
        DELETE r
        """

    with neo4j_driver.session() as session:
        result = session.run(query=cypher,
                             parameters={'node_id': node_id})

        relationship_data = result.data()

    # Confirm deletion was completed by empty response
    return relationship_data or {
        'response': f'Relationship of node with ID: {node_id} were successfully deleted from the graph.'
    }
