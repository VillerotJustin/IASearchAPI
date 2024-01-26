# Import modules from FastAPI
import configparser
import json
import re

# Other Libs
import pandas as pd
from fastapi import APIRouter
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk

# Import internal utilities for database access, authorisation, and schemas
from app.utils.db import neo4j_driver

# Set the API Router
router = APIRouter()

# Used for validation to ensure they are not overwritten
base_properties = ['created_by', 'created_time']

data = {
    'invalid_filters': [],
    'to_numerous_field': [],
    'invalid_filter': [],
    'restrictive_data_field': [],
    'node_label': []
}

config_file = configparser.ConfigParser()


###
# Différentes fonctions permettant de traiter le texte des ressources et de réaliser la phase de "pré-processing"
###

# Function for removing NonAscii characters
def _removeNonAscii(s):
    return "".join(i for i in s if ord(i) < 128)


# Function for converting into lower case
def make_lower_case(text):
    return text.lower()


# Function for removing stop words
def remove_stop_words(text):
    text = text.split()
    stops = set(stopwords.words('french'))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return text


# Function for removing punctuation
def remove_punctuation(text):
    tokenizer = RegexpTokenizer(r'\w+')
    text = tokenizer.tokenize(text)
    text = " ".join(text)
    return text


# Function for removing the html tags
def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)


def remove_letter_e_and_s_from_words(phrase):
    # Divise la phrase en mots
    mots = phrase.split()

    # Parcourt chaque mot et supprime les "e" et "s" à la fin de chaque mot
    mots_modifies = [mot.rstrip('es') for mot in mots]

    # Rejoint les mots modifiés pour former la nouvelle phrase
    nouvelle_phrase = ' '.join(mots_modifies)

    return nouvelle_phrase


###
# DB Info
###

def get_all_labels():
    query = "CALL db.labels()"
    with neo4j_driver.session() as session:
        result = session.run(query=query)
        data = result.data()
    return data


def get_every_property_keys_no_async():
    query = "CALL db.propertyKeys()"
    with neo4j_driver.session() as session:
        result = session.run(query=query)
        data = result.data()
    return data


def get_all_property_keys_of_label(label):
    query = f"""MATCH (n:{label})
        WITH n LIMIT 25
        UNWIND keys(n) as key
        RETURN distinct key"""
    # renvoie la liste des propriétés/champ de donnée des ressources
    with neo4j_driver.session() as session:
        result = session.run(query=query)
        properties_key = result.data()
    properties_key = [element['key'] for element in properties_key]
    return properties_key


def number_ressources():  # number of ressources
    query = """
    MATCH (n)
    RETURN count(n) as count
    """
    with neo4j_driver.session() as session:
        result = session.run(query=query)
        data = result.data()
    return data


def number_ressources_label(label: str):  # number of ressources for a particular label
    query = f"""
    MATCH (n:{label})
    RETURN count(n) as count
    """
    with neo4j_driver.session() as session:
        result = session.run(query=query)
        data = result.data()
    return data


###
# IA
# is it worth it, i don't know
###

data['node_label'] = get_all_labels()
data['property_keys'] = get_every_property_keys_no_async()


# Matières = ns0__setSpec

# API request
@router.post('/search')
async def AISearch(attributes: dict):
    ## Json print for debugging
    # json_object = json.loads(json.dumps(attributes))
    # json_formatted_str = json.dumps(json_object, indent=2)
    # print(json_formatted_str)

    ###
    #   Filter
    ###

    target_node_class = attributes["config_file"]["label"]
    filters = attributes["config_file"]["filters"]

    filters_cypher = ""  # f"filtre1:{filter_node_class}"
    where_cypher = ""  # f"WHERE(split(filtre1.uri, '#')[1] CONTAINS 'PHY')"
    i = 0
    for filter_node_class in filters:
        filter_value = filters[filter_node_class]
        print(filter_node_class, filter_value)
        filters_cypher += f"-[: {filter_value[1]}]->(filtre{i}:{filter_node_class})"
        where_cypher += f"WHERE(split(filtre{i}.uri, '#')[1] CONTAINS '{filter_value[0]}')"
        if i + 1 < len(filters):
            where_cypher += "|\n|"
        i += 1

    query_part1 = f"""
    MATCH(ressource: {target_node_class}){filters_cypher}
    {where_cypher} 
    WITH collect(ressource) AS filtered_{target_node_class}
    """

    with_parameters = attributes["config_file"]["WITH"]
    with_cypher = "WITH\n"
    i = 0
    for element in with_parameters:
        value = with_parameters[element]
        print(element, value)
        with_cypher += f"    {element} AS {value}"
        if i + 1 < len(with_parameters):
            with_cypher += ",\n"
        i += 1

    return_parameters = attributes["config_file"]["RETURN"]
    return_cypher = "RETURN\n"
    for element in return_parameters:
        return_cypher += f"    {element}"
        if i + 1 < len(return_parameters):
            return_cypher += ",\n"
        i += 1

    taxon = attributes["config_file"]["taxon"]
    taxon_cypher = ""
    if taxon != {}:
        for element in taxon:
            value = taxon[element]
            taxon_cypher += f"-[r: {element}]->(taxon:{value})"

    query_part2 = f"""
    MATCH(ressource: {target_node_class}){taxon_cypher}
    WHERE ressource IN filtered_{target_node_class}
    {with_cypher}
    RETURN
        ID, 
        title,
        description,
        reduce(s='', phrase IN phrases | s + phrase) AS concatenated_taxon
    """

    query = query_part1 + query_part2

    print(query)

    with neo4j_driver.session() as session:
        query_result = session.run(query=query)
        data['query_result'] = query_result.data()
        # print(f"data {data['query_result']}\n")

    filtered_nodes = data['query_result']

    ###
    #  Recommandation
    ###

    # TODO recommendation

    return {
        "number_of_results": len(filtered_nodes),
        "result": filtered_nodes
    }


def essential_filter_request(attributes):
    pre_query = f"""MATCH (n:{config_file['Resources']['focused_resource']} LIMIT 3)
    MATCH (n)-[r]-() RETURN n,r"""
    with neo4j_driver.session() as session:
        query_result = session.run(query=pre_query)
        relationships = query_result.data()
        print(relationships)

    query = f"MATCH (ressource:{config_file['Resources']['focused_resource']})\n"

    data['query_param'] = {}
    i = 0
    for (value) in (config_file["essential_filters"]):
        if value not in data['invalid_filters'] and value not in data['to_numerous_field']:
            data['query_param'][f'{value}'] = attributes[i]
            i += 1
        else:
            data['query_param'][f'{value}'] = ''
    # query += "WHERE"
    for value in data['query_param']:
        # print(value)
        query += f""

    query += "RETURN ressource"

    print("------ query ------\n", query, "\n-------------------")
    print("======\npre query")
    with neo4j_driver.session() as session:
        query_result = session.run(query=query, parameters=data['query_param'])
        data['query_result'] = query_result.data()
        # print(f"data {data['query_result']}\n")
    print("post query\n======")

    print(f"le nombre de ressource restant après première filtration sur les "
          f"filtres essentiels est de {len(data['query_result'])}")

    return data['query_result']


# requete -> ti_df (calcul suggestion)

# construction du fichier de configuration contenant les paramètres de la requête utilisateur
def build_fichier_config(config_file_data):
    # config_file = configparser.ConfigParser()

    config_file.optionxform = lambda option: option

    config_file["Project"] = {}
    config_file["Resources"] = {}

    # définir les sections et la clé/valeur en spécifiant le nom exact du champ de donnée de la propriété des nodes
    # liées aux ressources ou le nom exact de la donnée (appelée label sur neo4j) dans l'autre cas
    config_file["essential_filters"] = {}
    config_file["preferential_filters"] = {}

    # écriture des données dans le fichier de configuration en fonction des choix de l'utilisateur dans l'application

    config_file.set("Project", "Project_name", config_file_data["project_name"])
    config_file.set("Resources", "focused_resource", config_file_data["label"])

    # écriture des filtres essentiels dans le fichier de configuration qui seront plus tard utilisé pour la
    # recommandation
    i = 0
    for element in config_file_data["filters"]:
        # print(f"{element} : {config_file_data['filters'][element]}")
        config_file.set("essential_filters", element, config_file_data["filters"][element])
        i += 1

    # Enregistrer le fichier de configuration
    # Sous windows remplacez .conf par .ini, sur linux remplacez .ini par .conf

    with open("fichier.conf", 'w') as f:
        config_file.write(f)
    # Explicitement fermer le fichier
    f.close()
    # print("Le fichier de configuration fichier.conf a été créé")


def add_filters(key, value, query):
    nb_critere = len(data['query_param'].get(key))
    if nb_critere > 1:
        # ajout des critères sur les types de ressource à la requête, nécessite une
        # concatenation car le nombre de critères varie en fonction des requêtes

        for j in range(nb_critere - 1):
            # concaténation des critères à rechercher dans la requête cypher
            ajout = f"""OR ressource.{key} CONTAINS "{value[j + 1]}" """
            query = query + ajout

        query += ")"
    else:
        query += ")"

    query += f""" WITH  collect(ressource) AS filter_{key} """
    return query


# Requête cypher pour faire une première filtration sur les ressources selon les critères essentiels et les plus
# importants afin de réduire le nombre de ressources parmi lesquelles trouvé notre ressource à recommander et réduire
# le temps de calcul
def requete_neo4j_sur_filtre_essentiel(choix_critere):
    ###
    # Requête cypher sur neo4j la plus générique possible qui ne provoque pas une erreur si certaine syntaxe sont
    # incorrects en évitant simplement de considérer la donnée avec la syntaxe incorrecte
    ###
    # construction du dictionnaire de paramètre de la requête générique
    ###

    data['query_param'] = {}
    i = 0  # permet de parcourir la liste des critères renseignés par l'utilisateur
    for (value) in (config_file["essential_filters"]):
        if value not in data['invalid_filters'] and value not in data['to_numerous_field']:
            # dataframe['query_param'][f'{value}'] = dataframe['choix_critere'][i]
            data['query_param'][f'{value}'] = choix_critere[i]
            i += 1
        else:
            data['query_param'][f'{value}'] = ''
            # dataframe['query_param'][f'{value}'] = None

    print(f"query_param = {data['query_param']}")
    # st.write("dictionnaire paramètre de la requete : ")
    # for key, value in dataframe['query_param'].items():
    # st.write(f"{key}: {value}")

    data['query_param']['ressource'] = config_file.items("Resources")[0][1]
    # Ajout du nom de la ressource dans le dictionnaire (le nom du label associé aux ressources dans neo4j)

    # il serait possible de rajouter filtre lié au taxon pour le projet HUMANE, par exemple sur le niveau
    # d'enseignement ou bien préciser dans la barre de recherche comment écrire le niveau d'enseignement pour
    # respecter la bonne syntaxe sinon la méthode TF IDF ne sait pas reconnaitre la différence entre 2 mots écrit
    # d'une façon différente (en dehors du cas lié aux accents, au e et s excessif en fin de mot qui sont pris
    # en compte grâce à la fonction de text preprocessing recites plus bas)
    # https://www.reseau-canope.fr/scolomfr/data/scolomfr-9-1/fr/page/scolomfr-voc-022-num-121
    # https://www.reseau-canope.fr/scolomfr/data/scolomfr-9-1/fr/page/scolomfr-voc-022-num-626 ne pas forcément se
    # restreindre à un niveau d'enseignement pour avoir des résultats plus large/exhaustif

    # ## Construction de la requête permettant de filtrer selon les critères essentiels puis de trouver puis
    # concaténer tous les taxons de chaque ressource (fiche pédagogique ici) filtrée selon les critères essentiels
    # recherche dans neo4j à l'aide de "CONTAINS" pour voir si la propriété d'une node contient le mot clé en
    # question renseigné par l'utilisateur ##

    # st.write(f"query param = {dataframe['query_param']}")

    # la ligne WHERE filtre1.uri est specific au projet HUMANE compteur nécessaire car pour le premier élément du
    # dictionnaire, il faut faire une requete légèrement différente qui n'est pas générique avec le reste de la
    # requete que nous allons concaténer
    i = 0
    query = ""
    last_key = None
    for key, value in data['query_param'].items():  # parcours-les clé/valeur du dictionnaire de critères
        print(f"\n======================================\nkey:|{key}|  value:|{value}|\n")
        if i == 0:  # écriture spécifique lié au début de la requête cypher
            query, last_key = request_first_iteration(key, value, query, last_key)
        else:
            print(f"\nBoucle {i} start", f"|{query}|", f"|{last_key}|", f"|{i}|")
            print("===============")
            query, last_key, i = request_boucle(key, value, query, last_key, i)
            print("===============")
            print(f"Boucle {i} end", f"|{query}|", f"|{last_key}|", f"|{i}|")
        i += 1
        # on récupère la clé actuelle dans la boucle car nous l'utilisons par la suite pour
        # renvoyer la recherche dans le filtre précèdent
        # i += 1 last_key = key # on récupère la clé actuelle dans la boucle, car nous l'utilisons en
        # suite pour renvoyer la recherche dans le filtre précèdent

    # Ecriture de la fin de la requête, spécifique à chaque projet en fonction de comment sont nommés les champs de
    # données de la ressource du projet en question

    # if "HUMANE" == config_file.items("Project")[0][1]:
    #     # concanténation de tous les taxons de chaque ressource (fiche pédagogique) filtrée selon les critères
    #     # essentiels (spécifique au projet HUMANE)
    #     query += f"""
    #         MATCH (ressource:{data['query_param']['ressource']})-[r:ns0__has_taxon]->(taxon:ns0__taxon)
    #         WHERE ressource
    #         IN filter_{last_key}
    #
    #         WITH
    #         ressource.ns0__identifier as ID,
    #         ressource.ns0__title_string AS title,
    #         ressource.ns0__description_string as description,
    #         collect(taxon.ns0__entry_string) AS phrases
    #         RETURN
    #             ID,
    #             title,
    #             description,
    #             reduce(s = "", phrase IN phrases | s + ' ' + phrase) AS concatenated_taxon
    #     """

    # ## elif("EASING" == config.items("Project")[0][1]): si le projet en question est EASING alors, il faut spécifier
    # le nom des champs de données du type de node ressource à retourner, car ils n'auront pas nécessairement le même
    # nom que ceux du projet HUMANE ##

    # else: modèle de requête générique, mais qui nécessite cependant d'adapter le nom des champs de données du
    # type de node à retourner

    query += f"""                  
    MATCH (ressource:{data['query_param'][last_key]})
    WITH 
    ressource.ns0__identifier as ID, 
    ressource.ns0__title_string AS title, 
    ressource.ns0__description_string as description
    RETURN 
        ID,
        title,
        description
    """

    print(f"query: {query}")
    # penser à gérer le cas ou aucune ressource (fiches pedagogique) n'est trouvé

    # print(query)
    # Do query
    # data['query_result'] = data['gds'].run_cypher(query, data['query_param']) old query

    # MATCH(ressource: ns0__record)-[: ns0__has_setSpec]->(filtre1:ns0__setSpec)
    # WHERE(split(filtre1.uri, "#")[1]
    #   CONTAINS "PHY"
    # ) WITH collect(ressource) AS filter_ns0__setSpec
    #
    # MATCH(ressource: ns0__record)-[r: ns0__has_taxon]->(taxon:ns0__taxon)
    # WHERE ressource IN filter_ns0__setSpec
    # WITH
    #   ressource.ns0__identifier as ID,
    #   ressource.ns0__title_string
    # AS
    #   title,
    #   ressource.ns0__description_string as description,
    #   collect(taxon.ns0__entry_string)
    # AS phrases
    # RETURN
    #   ID,
    #   title,
    #   description,
    #   reduce(s="", phrase IN phrases | s + ' ' + phrase) AS concatenated_taxon

    print("pre query")
    with neo4j_driver.session() as session:
        query_result = session.run(query=query, parameters=data['query_param'])
        data['query_result'] = query_result.data()
        print(f"data {data['query_result']}\n")

    print("post query")

    print(f"le nombre de ressource restant après première filtration sur les "
          f"filtres essentiels est de {len(data['query_result'])}")
    # print(df)


def request_boucle(key, value, query, last_key, i):
    # écriture générique de la suite de la requête en fonction d'un nombre de critères et du nombre de choix
    # pour chaque critère le code qui suit est très similaire au code présent dans le "if" précèdent pour
    # construire le début de la requête

    print(f"iteration {i} de la requete\n")

    if key not in data['invalid_filter'] and key != 'ressource':
        # on vérifie qu'on ne traite pas une donnée qui renvoie une erreur sur la base de donnée neo4j car
        # elle possèderait un problème de syntaxe dans son label ou sur l'un de ces champs de données
        # accedes par exemple, on vérifie aussi qu'on ne traite pas la donnée ressource dans notre
        # dictionnaire, parce que ce n'est pas un filtre

        if ((any(element in data['restrictive_data_field'] for element in value)
             or any(element in data['restrictive_data_field'] for element in value)
             and not all(element in data['restrictive_data_field'] for element in value)
             and len(value) > 1)):
            # on vérifie aussi que le champ choisi par l'utilisateur ne fait pas partie de la liste des
            # éléments des champs de donnée trop restrictive, car cela restrain trop les ressources
            # renvoyées par la requête, s'il n'en fait pas partie, on passe à la suite, dans le cas ou un
            # élément du champ de donnée est trop restrictif, mais que d'autres éléments ne le sont pas dans le
            # meme champ de donnée alors, on passe à la suite aussi cependant si tous les éléments du champ de
            # donnée font partie de des éléments des champs de donnée trop restrictive ainsi, on ne passe pas à
            # la suite et le filtre est enlevé

            labels = []

            [labels.append(element['label']) for element in data['node_label']]

            if any(labels) == key:

                print("labels = key")

                # si le filtre essentiel que l'on est en train de parcourir
                # dans notre dictionnaire de paramètre correspond au label d'un certain type de node dans neo4j
                if key == "ns0__setSpec":
                    # spécifique au projet HUMANE et a la node setSpec dont la matière est écrit d'une
                    # manière spéciale dans le champ de donnée sous la forme
                    # http://edubase_ontologie/Ontologie.owl#MATHS ou il faut donc récupérer le dernier
                    # caractère du champ de donnée après le "#"
                    property = "has_" + key.replace("ns0__", "")  # nom de la relation/propiété dans neo4J
                    query += f""" 
                                    MATCH (ressource:{data['query_param']['ressource']})-[:ns0__{property}]->(filtre{i + 1}:{key}) 
                                    WHERE ressource 
                                    IN filter_{last_key} 
                                    AND ( split(filtre{i + 1}.uri, "#")[1] 
                                    CONTAINS "{value[0]}"
                                    """

                    # lorsque plusieurs champs de données sont sélectionnés pour une même donnée
                    nb_critere = len(data['query_param'].get(key))
                    if nb_critere > 1:
                        # ajout des critères sur les types de ressource à la requête, nécessite une
                        # concatenation car le nombre de critères varie en fonction des requêtes

                        for j in range(nb_critere - 1):
                            # concaténation des critères à rechercher dans la requête cypher
                            ajout = f"""OR split(filtre{i + 1}.uri, "#")[1] CONTAINS "{value[j + 1]}" """
                            query = query + ajout

                        query += ")"
                    else:
                        query += ")"

                    query += f""" WITH  collect(ressource) AS filter_{key} """
                    last_key = key
                else:
                    property = "has_" + key.replace("ns0__", "")  # nom de la relation/propiété dans neo4J
                    query += f"""
                                MATCH(ressource:{data['query_param']['ressource']})-[:ns0__{property}]->(filtre{i + 1}:{key}) 
                                WHERE ressource 
                                IN filter_{last_key} 
                                AND (filtre{i + 1}.{key}_label 
                                CONTAINS "{value[0]}"
                                """

                    # lorsque plusieurs champs de données sont sélectionnés pour une même donnée
                    nb_critere = len(data['query_param'].get(key))
                    # ajout des critères sur les types de ressource à la requête, nécessite une
                    # concanténation car le nombre de critères varie en fonction des requêtes
                    query = add_filters(key, value, query)
                    last_key = key
            else:
                properties = []
                [properties.append(element['propertyKey']) for element in data['property_keys']]
                if any(properties) == key:
                    # si le filtre essentiel que l'on est en train de parcourir dans notre dictionnaire de
                    # paramètre correspond au nom d'une propriété/champ de donnée d'un type de node dans neo4j

                    if key not in data['to_numerous_field']:
                        query += f""" 
                                    MATCH (ressource:{data['query_param']['ressource']})
                                    WHERE ressource IN filter_{last_key} 
                                    AND (ressource.{key}
                                    CONTAINS "{value[0]}"
                                    """

                        # lorsque plusieurs champs de données sont sélectionnés pour une même donnée
                        query = add_filters(key, value, query)
    else:
        print(
            f"{key} ne peut pas être examiner dans la base de donnée neo4j , le problème provient "
            f"probablement de la syntaxe dans son label ou sur l'un de ces champs de données accedées par "
            f"exemple"
        )
    return query, last_key, i


def request_first_iteration(key, value, query, last_key):
    # on vérifie qu'on ne traite pas une donnée qui renvoie une erreur sur la base de donnée neo4j car elle
    # possèderait un problème de syntaxe dans son label ou sur l'un de ces champs de données accedes par
    # exemple, on vérifie aussi qu'on ne traite pas la donnée ressource dans notre dictionnaire, parce que ce
    # n'est pas un filtre
    print("\nPremière iteration de la requete\n")
    i = 0
    if key not in data['invalid_filter'] and key != 'ressource':
        print(f"not an invalid filter {key not in data['invalid_filter']} "
              f"or key is not ressource {key != 'ressource'}")
        restrictive_data_fields = [element in data['restrictive_data_field'] for element in value]
        if ((not any(restrictive_data_fields)
             or any(restrictive_data_fields)
             and not all(restrictive_data_fields)
             and len(value) > 1)
        ):
            print("2")
            # on vérifie aussi que le champ choisi par l'utilisateur ne fait pas partie de la liste des
            # éléments des champs de donnée trop restrictive, car cela restreindrait trop les ressources
            # renvoyées par la requête, s'il n'en fait pas partie, on passe à la suite, dans le cas ou un
            # élément du champ de donnée est trop restrictif, mais que d'autres éléments ne le sont pas dans le
            # meme champ de donnée alors, on passe à la suite aussi cependant si tous les éléments du champ de
            # donnée font partie de des éléments des champs de donnée trop restrictive, on ne passe pas à
            # la suite et le filtre est enlevé

            labels = []
            for element in data['node_label']:
                labels.append(element['label'])
            print(f"==\n\n{labels}\n|{key}|\n{key in labels}\n\n==")
            if any(labels) == key:
                print("3rd if")
                # cas général ou le nom du champ de donnée est label et dont l'extraction du champ de donnée
                # est générique
                _property = "has_" + key.replace("ns0__", "")  # nom de la relation/propriété dans neo4J
                query = f"""
                MATCH(ressource:{data['query_param']['ressource']})-[:ns0__{_property}]->(filtre{i + 1}:{key}) 
                WHERE (filtre{i + 1}.{key}_label 
                CONTAINS "{value[0]}"
                """

                # lorsque plusieurs champs de données sont sélectionnés pour une même donnée
                nb_critere = len(data['query_param'].get(key))
                # ajout des critères sur les types de ressource à la requête, nécessite une concanténation
                # car le nombre de critères varie en fonction des requêtes
                if nb_critere > 1:
                    print("4th if")
                    # concaténation des critères à rechercher dans la requête cypher
                    for j in range(nb_critere - 1):
                        ajout = f""" OR filtre{i + 1}.{key}_label  CONTAINS {value[j + 1]}"""
                        query = query + ajout

                    query += ")"
                else:
                    query += ")"

                print(f"WITH collect(ressource) AS filter_{key}")
                query += f""" WITH collect(ressource) AS filter_{key}"""
            else:
                print(6)
                properties = []
                for element in data['property_keys']:
                    properties.append(element['propertyKey'])
                if any(properties) == key:
                    # si le filtre essentiel que l'on est en train de parcourir dans notre dictionnaire de
                    # paramètre correspond au nom d'une propriété/champ de donnée d'un type de node dans neo4j
                    if key not in data['to_numerous_field']:
                        query = f""" 
                        MATCH (ressource:{data['query_param']['ressource']}) 
                        WHERE (ressource.{key}
                        CONTAINS "{value[0]}"
                        """

                        # lorsque plusieurs champs de données sont sélectionnés pour une même donnée
                        query = add_filters(key, value, query)
            last_key = key

    else:
        print(
            f"{key} ne peut pas être examiner dans la base de donnée neo4j , le problème provient "
            f"probablement de la syntaxe dans son label ou sur l'un de ces champs de données accedées par "
            f"exemple")
    return query, last_key


# calcul recomandation
def tf_idf(ressource):
    # cas général : on concatène le titre et la description de la ressource
    concatenated_tittle_desc = ressource.title + ' ' + ressource.description

    # ## ajout de la requête utilisateur dans le dataframe des ressources filtrées jusqu'à là pour comparer ces
    # requêtes à la description et titres et mots clés (taxons) des ressources déjà filtrées ##

    # Créer une série avec des valeurs nulles sauf pour la dernière colonne
    nouvelle_ligne = pd.Series(
        [None] * (len(st.session_state['df'].columns) - 1) + [st.session_state['requete_utilisateur']],
        index=st.session_state['df'].columns)

    # Ajoute la nouvelle ligne au DataFrame en concaténant le DataFrame original avec la nouvelle ligne
    st.session_state['df'] = pd.concat([st.session_state['df'], nouvelle_ligne.to_frame().T], ignore_index=True)
    # st.write(f"df = {st.session_state['df']}")

    # ## importance de la syntaxe des mots : si un mot n'est pas écrit avec la même orthographe comme il l'est écrit
    # dans les ressource alors il n'est pas pris en compte dans la recherche comme le montre l'exemple ci dessous qui
    # ne renvoie pas le même résultat user_request = "trigonométrie 3e cosinus sinus géométrie "  # renvoie la
    # ressource : Cosinus et sinus sous GeoGebra Construction des courbes des fonctions sinus et cosinus à partir du
    # cercle trigonométrique. user_request = "trigonométrie 3e cosinu sinu géométrie "   # renvoie la ressource :
    # Apprendre la trigonométrie , Séquence de découverte des fonctions trigonométriques en classe de 3e, ##

    # ============================================== StreamLite =============================================


    # from nltk.corpus import stopwords

    ###
    #   affichage des graphiques sur la distribution du nombre de mot des ressources
    #   ainsi que les bigram/trigram si l'utilisateur le souhaite
    ###
    if st.session_state['visualisation_donnee'] == 'Oui':
        # Calculating the word count for ressources description
        df_countword = pd.DataFrame()
        df_countword['word_count'] = st.session_state['df']['combined'].apply(lambda x: len(
            str(x).split()))
        # prend la colonne 'combined' et compte le nombre de mot  de chaque ligne de cette
        # colonne pour créer le dataframe countword
        # Plotting the word count
        fig, ax = plt.subplots(figsize=(12, 8))
        df_countword['word_count'].plot(kind='hist', bins=50, ax=ax)
        # Configure les étiquettes et le titre
        ax.set_xlabel("Number of Words")
        ax.set_ylabel("Number of Resources")
        ax.set_title("Word Count Distribution for Resource Descriptions + Title")

        # Affichez le plot avec st.pyplot()
        st.subheader(
            "Affichage de la distribution du nombre de mot de chaque ressource à la suite de la concaténation du titre, de la description et des taxon de chaque ressource")
        st.pyplot(fig)

        # Converting text descriptions into vectors using TF-IDF using Bigram

        nltk.download('stopwords')
        # print(stopwords.words('french'))

        final_stopwords_list = stopwords.words(
            'french')  # permet d'éviter que la méthode TF IDF ne compte les termes redondant (le,la,les,un,une ect...) qui n'apporte pas d'information sur la ressource
        # final_stopwords_list = stopwords.words('english') + stopwords.words('french')

        tf = TfidfVectorizer(  # max_df=0.8,
            # max_features=200000,
            # min_df=0.2,
            stop_words=final_stopwords_list,
            # use_idf=True,
            # tokenizer=tokenize_and_stem,
            ngram_range=(2, 2))  # ngram_range=(3,3) pour avoir les trigram

        tfidf_matrix = tf.fit_transform(st.session_state['df']['combined'])
        total_words = tfidf_matrix.sum(axis=0)
        # Finding the word frequency
        freq = [(word, total_words[0, idx]) for word, idx in tf.vocabulary_.items()]
        freq = sorted(freq, key=lambda x: x[1], reverse=True)
        # converting into dataframe
        bigram = pd.DataFrame(freq)
        bigram.rename(columns={0: 'bigram', 1: 'count'}, inplace=True)
        # Taking first 20 records
        bigram = bigram.head(20)
        # Plotting the bigram distribution
        # bigram.plot(x ='bigram', y='count', kind = 'bar', title = "Bigram disribution for the top 20 words in the ressources description", figsize = (15,7), )
        fig, ax = plt.subplots(figsize=(15, 7))
        bigram.plot(x='bigram', y='count', kind='bar',
                    title="Bigram Distribution for the Top 20 Words in the Resource Descriptions", ax=ax)
        # Configurez les étiquettes et le titre
        ax.set_xlabel("Bigram")
        ax.set_ylabel("Count")
        # Affichez le graphique avec st.pyplot()
        st.subheader(" Distribution des bigram avant pre processing des textes concanténés des ressources")
        st.pyplot(fig)

    else:  # initialisation des variables pour notre modèle de recommandation dans le cas ou l'on n'affiche pas la visualisation des données
        final_stopwords_list = stopwords.words(
            'french')  # permet d'éviter que la méthode TF IDF ne compte les termes redondant (le,la,les,un,une ect...) qui n'apporte pas d'information sur la ressource

    if (st.session_state['methode_calcul_recommandation'] == 'TF-IDF'):
        ###
        # Applying all the functions and storing as a cleaned_desc
        # crée une nouvelle colonne 'cleaned_des' dans le dataframe contenant la concaténation des données utilisés pour TF IDF
        # df['cleaned_desc'] = df['combined'].apply(_removeNonAscii)
        # df['cleaned_desc'] = df.cleaned_desc.apply(func = make_lower_case)
        # df['cleaned_desc'] = df.cleaned_desc.apply(func = remove_stop_words)
        # df['cleaned_desc'] = df.cleaned_desc.apply(func=remove_punctuation)
        # df['cleaned_desc'] = df.cleaned_desc.apply(func=remove_html)
        # df['cleaned_desc'] = df.cleaned_desc.apply(func=remove_letter_e_and_s_from_words)
        ###

        # Applying all the functions and storing as a cleaned_desc
        # remplace la colonne dans le dataframe contenant la concaténation des données utilisés pour TF IDF au lieu d'en crée une nouvelle et de dédoubler nos données
        # évite d'avoir une nouvelle colonne dans le dataframe pour alléger le nombre de données
        st.session_state['df']['combined'] = st.session_state['df']['combined'].apply(_removeNonAscii)
        st.session_state['df']['combined'] = st.session_state['df'].combined.apply(func=make_lower_case)
        st.session_state['df']['combined'] = st.session_state['df'].combined.apply(func=remove_stop_words)
        st.session_state['df']['combined'] = st.session_state['df'].combined.apply(func=remove_punctuation)
        st.session_state['df']['combined'] = st.session_state['df'].combined.apply(func=remove_html)
        st.session_state['df']['combined'] = st.session_state['df'].combined.apply(
            func=remove_letter_e_and_s_from_words)

        # print(f" Dataset des description nettoyées : \n\n {df['combined']}")    # les accents sont aussi enlevé
        # print(f" Dataset des description nettoyées : \n\n {df['cleaned_desc']}")    # les accents sont aussi enlevé

    # affichage de la distribution des bigram après processing des textes concanténés des ressources
    if st.session_state['visualisation_donnee'] == 'Oui' and st.session_state[
        'methode_calcul_recommandation'] == 'TF-IDF':
        tf = TfidfVectorizer(  # max_df=0.8,
            # max_features=200000,
            # min_df=0.2,
            stop_words=final_stopwords_list,
            # use_idf=True,
            # tokenizer=tokenize_and_stem,
            ngram_range=(2, 2))  # (3,3) pour avoir les trigram

        tfidf_matrix = tf.fit_transform(st.session_state['df']['combined'])
        total_words = tfidf_matrix.sum(axis=0)
        # Finding the word frequency
        freq = [(word, total_words[0, idx]) for word, idx in tf.vocabulary_.items()]
        freq = sorted(freq, key=lambda x: x[1], reverse=True)
        # converting into dataframe
        bigram = pd.DataFrame(freq)
        bigram.rename(columns={0: 'bigram', 1: 'count'}, inplace=True)
        # Taking first 20 records
        bigram = bigram.head(20)
        # Plotting the bigram distribution
        # bigram.plot(x ='bigram', y='count', kind = 'bar', title = "Bigram disribution for the top 20 words in the ressources description", figsize = (15,7), )
        fig, ax = plt.subplots(figsize=(15, 7))
        bigram.plot(x='bigram', y='count', kind='bar',
                    title="Bigram Distribution for the Top 20 Words in the Resource Descriptions", ax=ax)
        # Configurez les étiquettes et le titre
        ax.set_xlabel("Bigram")
        ax.set_ylabel("Count")
        # Affichez le graphique avec st.pyplot()
        st.subheader(" Distribution des bigram après pre processing des textes concanténés des ressources")
        st.pyplot(fig)

    ###
    # Recommandation basé sur la description et le titre de la ressource :
    # Converting the title and description into vectors and used bigram
    ###

    if (st.session_state['methode_calcul_recommandation'] == 'TF-IDF'):
        # Doc de la fonction https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=1, stop_words=final_stopwords_list,
                             lowercase=True)  # ngram_range = (2,2) ne renvoie que des 0 dans la matrice de similarité
        # tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df = 1, max_df = 0.7,  lowercase = True)                         # possibilité de mettre une valeur forte pour max_df entre 0.7 et 1 à la place  de la liste de stop_word

        # tfidf_matrix = tf.fit_transform(df['cleaned_desc'])
        st.session_state['tfidf_matrix'] = tf.fit_transform(st.session_state['df']['combined'])
        # st.write(st.session_state['tfidf_matrix'])

    if (st.session_state['methode_calcul_recommandation'] == 'TF-IDF+Word2Vec'):

        # Applying all the functions and storing as a cleaned_desc
        # le pre processing ici est moins stricte car le modèle Word2vec permet
        # d'avoir une certaine compréhension du contexte de la phrase donc il ne faut pas dénaturer les textes de notre dataframe
        # st.session_state['df']['combined'] = st.session_state['df']['combined'].apply(_removeNonAscii)
        st.session_state['df']['combined'] = st.session_state['df'].combined.apply(func=make_lower_case)
        st.session_state['df']['combined'] = st.session_state['df'].combined.apply(func=remove_stop_words)
        st.session_state['df']['combined'] = st.session_state['df'].combined.apply(func=remove_punctuation)
        st.session_state['df']['combined'] = st.session_state['df'].combined.apply(func=remove_html)
        # st.session_state['df']['combined'] = st.session_state['df'].combined.apply(func=remove_letter_e_and_s_from_words)

        _ = '''# Chemin relatif vers le dossier du script
        script_folder = os.path.dirname(__file__)

        # Remonter d'un niveau pour accéder au dossier contenant le modèle
        model_folder = os.path.join(script_folder, '..')  # '..' représente le dossier parent

        # Chemin vers le fichier modèle
        model_path = os.path.join(model_folder, 'AI model', 'cc.fr.300.bin')
        '''

        from gensim.models import KeyedVectors

        # Charger le modèle
        with st.spinner("chargement du modèle de NLP pré-entrainé"):

            # https://fasttext.cc/docs/en/crawl-vectors.html
            # pre-trained word vectors for 157 languages, trained on Common Crawl and Wikipedia
            # using fastText. These models were trained using CBOW with position-weights,
            # in dimension 300, with character n-grams of length 5, a window of size 5 and 10 negatives.
            # with three new word analogy datasets, for French, Hindi and Polish.

            # model = FastText.load_fasttext_format(model_path)  # ce modèle est trop gourmand en mémoire pour être utilisé

            frWac_model_path = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.vec.gz"
            ###
            # vocabulaire limité à 50 000 mots, cela couvre une grande partie du vocabulaire courant et est souvent suffisant pour de nombreuses applications .
            # La plupart des mots fréquemment utilisés dans le langage quotidien, ainsi que beaucoup de termes spécialisés, sont généralement inclus
            # dans un modèle avec un vocabulaire de cette taille. Pour de nombreuses tâches, l'utilisation d'un modèle avec un vocabulaire plus important peut ne pas apporter d'amélioration significative, tout en augmentant les exigences en mémoire.
            # possibilité d'augmenter ou réduire le nombre de mots dans le paramètre limit=50 000
            ###
            # model = KeyedVectors.load_word2vec_format(frWac_model_path, binary=False, limit=50000)
            if 'model' not in st.session_state:  # évite de recharger le modèle entre chaque interaction une fois l'application lancé
                st.session_state['model'] = KeyedVectors.load_word2vec_format(frWac_model_path, binary=False,
                                                                              limit=50000)
                # st.session_state['model'] = KeyedVectors.load_word2vec_format(frWac_model_path, binary=False, limit=100000)
                # st.session_state['model'] = KeyedVectors.load_word2vec_format(frWac_model_path, binary=False, limit=200000) # 2 minute pour charger ce modèle à 200 000 mot

            ###
            # la nécessité de ré entrainer le modèle avec notre dataset est à étudier car ce n'est pas forcément nécessaire cependant
            # il n'est pas possible de ré-entrainer ce modèle en question avec notre dataset , Si l'ensemble de données est assez différent du corpus sur lequel le modèle pré-entraîné a été construit, la ré-entraînement peut améliorer les performances du modèle pour votre tâche particulière
            # un ré entrainement peut notamment ajouter de nouveaux mots si notre dataset contient des mots qui n'existent pas dans le modele

            # pour réaliser un ré entrainement il faut aussi prendre en compte la taille de notre dataset, s'il est trop grand il faut prendre en compte les ressources nécessaires qui peuvent être gourmande pour le ré entrainer
            # , s'il est trop petit alors le modèle va se surajuster à notre dataset et ne généralisera pas bien à de nouvelle donnée
            #          => dans la mesure ou l'objectif est de réaliser un système de recommandation générique qui doit fonctionner sur plusieurs projets avec des données différentes entre chaque projet
            #               alors il ne semble pas nécessaire de ré entrainer le modèle et de simplement garder le modèle d'origine qui est plus général
            ###

        st.write('Chargement du modèle pré-entrainé réussi')

        # Building TFIDF model and calculate TFIDF score
        tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=2, stop_words=final_stopwords_list,
                                lowercase=True)
        # st.session_state['tfidf_matrix'] = tf.fit(st.session_state['df']['combined'])
        # tfidf.fit(st.session_state['df']['combined'])
        st.session_state['tfidf_matrix'] = tfidf.fit_transform(st.session_state['df']['combined'])

        # Getting the words from the TF-IDF model

        tfidf_list = dict(zip(tfidf.get_feature_names_out(), list(tfidf.idf_)))
        tfidf_feature = tfidf.get_feature_names_out()  # tfidf words/col-names

        # detailed tutorial of the computation : https://pub.towardsai.net/content-based-recommendation-system-using-word-embeddings-c1c15de1ef95

        # splitting the description into words
        corpus = []
        for words in st.session_state['df']['combined']:
            corpus.append(words.split())

        len_dataframe = len(st.session_state['df']['combined'])
        mot_absent_du_vocabulaire_du_modele = []  # mots qui ne sont pas dans le vocabulaire du modèle mais présent dans requête utilisateur
        st.session_state[
            'mot_non_present_dans_le_vocabulaire'] = False  # variable pour identifier si des mots de la requête utilisateur ne sont pas présent dans le vocabulaire du modèle

        # Building TF-IDF Word2Vec

        # Storing the TFIDF Word2Vec embeddings
        tfidf_vectors = []
        line = 0
        # for each book description
        # for i , desc in zip(len_dataframe , corpus):
        for i, desc in enumerate(corpus):
            # for desc in corpus:
            # Word vectors are of zero length (Used 300 dimensions)
            sent_vec = np.zeros(300)
            # num of words with a valid vector in the book description
            weight_sum = 0
            # for each word in the book description
            for word in desc:
                if word in st.session_state['model'].key_to_index and word in tfidf_feature:
                    vec = st.session_state['model'][word]
                    tf_idf = tfidf_list[word] * (desc.count(word) / len(desc))

                    if tf_idf < 1:  # garanti que le score tf_idf est > à 1 pour que exp(tf_idf^2) augmente significativement le score tf_idf
                        tf_idf += 1
                    tf_idf = np.exp(tf_idf ** 2)

                    # if tf_idf < 1 :
                    # tf_idf = tf_idf**5
                    # else:
                    # tf_idf = tf_idf**5

                    if i == len_dataframe - 1:
                        st.write(f"tf_idf du mot '{word}' = {tf_idf}")

                    sent_vec += (vec * tf_idf)
                    weight_sum += tf_idf
                else:
                    if i == len_dataframe - 1:  # lorsque l'on traite la requête utilisateur, c'est à dire la dernière ligne du dataframe

                        ###
                        # Traitement du cas ou un mot de la requête utilisateur n'est pas reconnu (lorsque le mot n'est pas dans le vocabulaire du modèle)
                        # ce qui peut réduire grandement la précision des résultats
                        ###

                        st.session_state['mot_non_present_dans_le_vocabulaire'] = True

                        mot_absent_du_vocabulaire_du_modele.append(word)

                        # Si le mot n'est pas reconnu, obtiens le mot le plus proche
                        # mots_proches = st.session_state['model'].similar_by_word(word, topn=1)
                        # mots_proches = st.session_state['model'].similar_by_vector(st.session_state['model'][word] , topn=1)
                        # mot_proche, similarite = mots_proches[0]
                        # print(f"Le mot le plus proche du mot non reconnu '{word}' est '{mot_proche}' avec une similarité de {similarite}")

            if weight_sum != 0:
                sent_vec /= weight_sum
            else:  # si aucun mot n'a été reconnu
                st.write(
                    "aucun mot de la requête n'a été reconnu provoquant des résultats imprécis, précisez davantage votre requête en ajoutant de nouveaux mots")
                st.write("weight sum = 0")
                from numpy import random
                sent_vec = random.randint(2, size=(
                    300))  # le vecteur de la requete utilisateur est remplacé par un vecteur de nombre aléatoire

            tfidf_vectors.append(sent_vec)
            line += 1

        st.session_state['tfidf_matrix+Word2Vec'] = tfidf_vectors
        st.write(
            f"Mots de la requête utilisateur suivante :   ''' {st.session_state['df']['combined'].iloc[-1]} '''   qui ne sont pas présent dans le vocabulaire du modèle et donc ne sont pas pris en compte dans le calcul de la recommandation et réduit la précision : {mot_absent_du_vocabulaire_du_modele}")

