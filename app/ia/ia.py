# Import modules from FastAPI
from fastapi import APIRouter

# Other Libs
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from gensim.models import KeyedVectors
import nltk
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
import re
import configparser

# Import internal utilities for database access, authorisation, and schemas
from app.utils.db import neo4j_driver

# Set the API Router
router = APIRouter()

# Used for validation to ensure they are not overwritten
base_properties = ['created_by', 'created_time']

dataframe = pd.DataFrame

config_file = None


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
###

# Matieres = ns0__setSpec

# API request
@router.post('/search')
async def AISearch():
    # Parrameter parsing

    config_file_data = {
        "project_name": "HUMANE",
        "label": "ns0__setSpec",
        "filter": {
            "name",
        },
        "opt_filter": {
            "",
        },
    }
    build_fichier_config(config_file_data)
    # Neo4J Request
    # requeteneo4j_sur_filtre_essentiel()

    # Suggestion Processing

    return {"message": "TODO"}


# requete -> ti_df (calcul suggestion)

# construction du fichier de configuration contenant les paramètres de la requête utilisateur
def build_fichier_config(config_file_data):
    x = 0
    print(x)
    config_file = configparser.ConfigParser()

    x += 1
    print(x)
    config_file["Project"] = {}
    config_file["Ressource"] = {}

    x += 1
    print(x)
    # définir les sections et les clé/valeur en spécifiant le nom exact du champ de donnée de la propriété des nodes liées aux ressources
    # ou le nom exact de la donnée (appelée label sur neo4j) dans l'autre cas
    config_file["essential_filters"] = {}
    config_file["preferential_filters"] = {}

    # écriture des données dans le fichier de configuration en fonction des choix de l'utilisateur dans l'application

    x += 1
    print(x)
    config_file.set("Project", "Project_name", config_file_data["project_name"])
    x += 1
    print(x)
    config_file.set("Ressource", "Ressource_a_recommander", config_file_data["label"])

    # /\ problem changing line to self.config["TV"]["apiv"] = str(api_version)

    x += 1
    print(x)
    # écriture des filtres essentiels dans le fichier de configuration qui seront plus tard utilisé pour la recommandation
    i = 0
    for element in config_file_data["filter"]:
        config_file.set("essential_filters", f"filtre{i}", element)
        i += 1

    x += 1
    print(x)
    # Enregistrer le fichier de configuration
    # Sous windows remplacez .conf par .ini, sur linux remplacez .ini par .conf
    x += 1
    print(x)
    with open("fichier.conf", 'w') as f:
        config_file.write(f)
        x += 1
        print(x)
    # Explicitement fermer le fichier
    f.close()
    print("Le fichier de configuration fichier.conf a été créé")


# Requête cypher pour faire une première filtration sur les ressources selon les critères essentiels et les plus importants
# afin de réduire le nombre de ressource parmi lesquelles trouvé notre ressource a recommander et réduire le temps de calcul
def requeteneo4j_sur_filtre_essentiel(choix_critere):
    ###
    # Requête cypher sur neo4j la plus générique possible qui ne provoque pas une erreur si certains syntaxe sont incorrect en évitant simplement de considérer
    # la donnée avec la syntaxe incorrect
    ###

    ###
    # construction du dictionnaire de paramètre de la requête générique
    ###
    dataframe['query_param'] = {}
    i = 0  # permet de parcourir la liste des critères renseignés par l'utilisateur
    for (name, value) in (config_file.items("Filtres essentiels")):

        if value not in dataframe['invalid_filters'] and value not in dataframe['list_donnee_trop_nombreuse']:
            # dataframe['query_param'][f'{value}'] = dataframe['choix_critere'][i]
            dataframe['query_param'][f'{value}'] = choix_critere[i]
            i += 1
        else:
            dataframe['query_param'][f'{value}'] = ''
            # dataframe['query_param'][f'{value}'] = None

    # st.write("dictionnaire paramètre de la requete : ")
    # for key, value in dataframe['query_param'].items():
    # st.write(f"{key}: {value}")

    dataframe['query_param']['ressource'] = config_file.items("Ressources")[0][
        1]  # Ajout du nom de la ressource dans le dictionnaire (le nom du label associé au ressource dans neo4j)

    # il serait possible de rajouter filtre liée au taxon pour le projet HUMANE , par exemple sur le niveau d'enseignement ou bien préciser dans la barre de recherche comment écrire le niveau d'enseignement
    # pour respecter la bonne syntaxe sinon la méthode TF IDF ne sait pas reconnaitre la différence entre 2 mots écrit d'une façon différentes
    # (en dehors de la casse lié aux accents , au e et s excessif en fin de mot qui sont pris en compte grâce au fonction de text preprocessing ecrites plus bas)
    #  https://www.reseau-canope.fr/scolomfr/data/scolomfr-9-1/fr/page/scolomfr-voc-022-num-121 https://www.reseau-canope.fr/scolomfr/data/scolomfr-9-1/fr/page/scolomfr-voc-022-num-626
    # ne pas forcément se restreindre à un niveau d'enseignement pour avoir des résultats plus large/exhaustif

    ###
    # Construction de la requête permettant de filtrée selon les critères essentiels puis de trouver puis concanténer tous les taxons de chaque ressource (fiche pédagogique ici) filtrée selon les critères essentiels
    # recherche dans neo4j a l'aide de "CONTAINS" pour voir si la propriété d'une node contient le mot clé en question renseigné par l'utilisateur
    ###

    # st.write(f"query param = {dataframe['query_param']}")

    # la ligne WHERE filtre1.uri est spécifice au projet HUMANE
    i = 0  # compteur nécessaire car pour le premier élément du dictionnaire il faut faire une requete légèrement différente qui n'est pas générique avec le reste de la requete que nous allons concaténer
    for key, value in dataframe['query_param'].items():  # parcours les clé/valeur du dictionnaire de critères
        if i == 0:  # écriture spécifique lié au début de la requête cypher

            if (key not in dataframe[
                'list_filtre_invalide'] and key != 'ressource'):  # on vérifie qu'on ne traite pas une donnée qui renvoie une erreur sur la base de donnée neo4j car elle possèderait un problème de syntaxe dans son label ou sur l'un de ces champs de données accedées par exemple
                # on vérifie aussi qu'on ne traite pas la donnée ressource dans notre dictionnaire car ce n'est pas un filtre

                if ((any(element in dataframe['list_champ_donne_trop_restrictif'] for element in value) == False
                     or any(element in dataframe['list_champ_donne_trop_restrictif'] for element in
                            value) == True
                     and all(element in dataframe['list_champ_donne_trop_restrictif'] for element in
                             value) == False
                     and len(value) > 1)):
                    # on vérifie aussi que le champ choisi par l'utilisateur ne fait pas partie de la liste des éléments des champs de donnée
                    # trop restrictif car cela restricterait trop les ressources renvoyés par la requête , s'il n'en fait pas
                    # partie on passe a la suite , dans le cas ou un élément du champ de donnée est trop restrictif
                    # mais que d'autre éléments ne le sont pas dans le meme champ de donnée alors on passe a la suite aussi
                    # cependant si tous les éléments du champ de donnée font partie de des éléments des champs de donnée
                    # trop restrictif alors on ne passe pas à la suite et le filtre est enlevé

                    if (any(dataframe['node_label'][
                                'label'] == key) == True):  # si le filtre essentiel que l'on est entrain de parcourir
                        # dans notre dictionnaire de paramètre correspond au label d'un certain type de node dans neo4j

                        if ("HUMANE" == config_file.items("Project")[0][
                            1]):  # ne s'applique que si le nom de projet du fichier de configuration est "HUMANE"
                            if (
                                    key == "ns0__setSpec"):  # spécifique au projet HUMANE et a la node setSpec dont la matière est écrit d'une manière spécial dans le champ de donnée
                                # sous la forme http://edubase_ontologie/Ontologie.owl#MATHS ou il faut donc récuper les dernière caractère du champ de donnée après le "#"
                                property = "has_" + key.replace("ns0__", "")  # nom de la relation/propiété dans neo4J
                                query = f""" 
                                MATCH (ressource:{dataframe['query_param']['ressource']})-[:ns0__{property}]->(filtre{i + 1}:{key}) 
                                WHERE ( split(filtre{i + 1}.uri, "#")[1] 
                                CONTAINS "{value[0]}"
                                """

                                # lorsque plusieurs champs de données sont sélectionnés pour une même donnée
                                nb_critere = len(dataframe['query_param'].get(key))
                                if (
                                        nb_critere > 1):  # ajout des critères sur le types de ressource à la requête , nécessite une concanténation car le nombre de critère varie en fonction des requêtes

                                    for j in range(
                                            nb_critere - 1):  # concaténation des critères à rechercher dans la requête cypher
                                        ajout = f"""OR split(filtre{i + 1}.uri, "#")[1] CONTAINS "{value[j + 1]}" """
                                        query = query + ajout

                                    query += ")"
                                else:
                                    query += ")"

                                query += f""" WITH  collect(ressource) AS filter_{key} """

                                i += 1
                                last_key = key  # on récupère la clé actuel dans la boucle car nous l'utilisons par la suite pour renvoyer la recherche dans le filtre précèdent

                                continue

                        ###
                        # elif("EASING" == config.items("Project")[0][1]):       # comme précèdement si le projet EASING doit traiter un champ de donnée qui n'est pas générique
                        # rajouter ici le code du cas spécifique au projet EASING si nécessaire en remplacant uri dans filtre{i+1}.uri par le nom du champ de donnée correspondant
                        # et en adaptant à la représentation de ce champ de donnée qui ne nécessite pas forcément d'utiliser la fonction split de la meme facon
                        ###

                        # cas général ou le nom du champ de donnée est label et dont l'extraction du champ de donnée est générique
                        property = "has_" + key.replace("ns0__", "")  # nom de la relation/propiété dans neo4J
                        query = f"""
                        MATCH(ressource:{dataframe['query_param']['ressource']})-[:ns0__{property}]->(filtre{i + 1}:{key}) 
                        WHERE (filtre{i + 1}.{key}_label 
                        CONTAINS "{value[0]}"
                        """

                        # lorsque plusieurs champs de données sont sélectionnés pour une même donnée
                        nb_critere = len(dataframe['query_param'].get(key))
                        if (
                                nb_critere > 1):  # ajout des critères sur le types de ressource à la requête , nécessite une concanténation car le nombre de critère varie en fonction des requêtes

                            for j in range(
                                    nb_critere - 1):  # concaténation des critères à rechercher dans la requête cypher
                                ajout = f"""OR filtre{i + 1}.{key}_label  CONTAINS "{value[j + 1]}" """
                                query = query + ajout

                            query += ")"
                        else:
                            query += ")"

                        query += f""" WITH collect(ressource) AS filter_{key}"""

                        i += 1
                        last_key = key  # on récupère la clé actuel dans la boucle car nous l'utilisons par la suite pour renvoyer la recherche dans le filtre précèdent


                    else:
                        if (any(dataframe['champ_donnee_de_toutes_les_nodes'][
                                    'propertyKey'] == key) == True):  # si le filtre essentiel que l'on est entrain de parcourir
                            # dans notre dictionnaire de paramètre correspond au nom d'une propriété/champ de donnée d'un type de node dans neo4j

                            if key not in dataframe['list_donnee_trop_nombreuse']:
                                query = f""" 
                                MATCH (ressource:{dataframe['query_param']['ressource']}) 
                                WHERE (ressource.{key}
                                CONTAINS "{value[0]}"
                                """

                                # lorsque plusieurs champs de données sont sélectionnés pour une même donnée
                                nb_critere = len(dataframe['query_param'].get(key))
                                if (
                                        nb_critere > 1):  # ajout des critères sur le types de ressource à la requête , nécessite une concanténation car le nombre de critère varie en fonction des requêtes

                                    for j in range(
                                            nb_critere - 1):  # concaténation des critères à rechercher dans la requête cypher
                                        ajout = f"""OR ressource.{key} CONTAINS "{value[j + 1]}" """
                                        query = query + ajout

                                    query += ")"
                                else:
                                    query += ")"

                                query += f""" WITH  collect(ressource) AS filter_{key} """

                                i += 1
                                last_key = key  # on récupère la clé actuel dans la boucle car nous l'utilisons par la suite pour renvoyer la recherche dans le filtre précèdent

            else:
                print(
                    f"{key} ne peut pas être examiner dans la base de donnée neo4j , le problème provient probablement de la syntaxe dans son label ou sur l'un de ces champs de données accedées par exemple")

        else:  # écriture générique de la suite de la requête en fonction d'une nombre de critère et du nombre de choix pour chaque critère
            # le code qui suit est très similaire au code présent dans le "if" précèdent pour construire le début de la requête

            if (key not in dataframe[
                'list_filtre_invalide'] and key != 'ressource'):  # on vérifie qu'on ne traite pas une donnée qui renvoie une erreur sur la base de donnée neo4j car elle possèderait un problème de syntaxe dans son label ou sur l'un de ces champs de données accedées par exemple
                # on vérifie aussi qu'on ne traite pas la donnée ressource dans notre dictionnaire car ce n'est pas un filtre

                if ((any(element in dataframe['list_champ_donne_trop_restrictif'] for element in value) == False
                     or any(element in dataframe['list_champ_donne_trop_restrictif'] for element in
                            value) == True
                     and all(element in dataframe['list_champ_donne_trop_restrictif'] for element in
                             value) == False
                     and len(value) > 1)):
                    # on vérifie aussi que le champ choisi par l'utilisateur ne fait pas partie de la liste des éléments des champs de donnée
                    # trop restrictif car cela restricterait trop les ressources renvoyés par la requête , s'il n'en fait pas
                    # partie on passe a la suite , dans le cas ou un élément du champ de donnée est trop restrictif
                    # mais que d'autre éléments ne le sont pas dans le meme champ de donnée alors on passe a la suite aussi
                    # cependant si tous les éléments du champ de donnée font partie de des éléments des champs de donnée
                    # trop restrictif alors on ne passe pas à la suite et le filtre est enlevé

                    if (any(dataframe['node_label'][
                                'label'] == key) == True):  # si le filtre essentiel que l'on est entrain de parcourir
                        # dans notre dictionnaire de paramètre correspond au label d'un certain type de node dans neo4j
                        if (
                                key == "ns0__setSpec"):  # spécifique au projet HUMANE et a la node setSpec dont la matière est écrit d'une manière spécial dans le champ de donnée
                            # sous la forme http://edubase_ontologie/Ontologie.owl#MATHS ou il faut donc récuper les dernière caractère du champ de donnée après le "#"
                            property = "has_" + key.replace("ns0__", "")  # nom de la relation/propiété dans neo4J
                            query += f""" 
                                MATCH (ressource:{dataframe['query_param']['ressource']})-[:ns0__{property}]->(filtre{i + 1}:{key}) 
                                WHERE ressource 
                                IN filter_{last_key} 
                                AND ( split(filtre{i + 1}.uri, "#")[1] 
                                CONTAINS "{value[0]}"
                                """

                            # lorsque plusieurs champs de données sont sélectionnés pour une même donnée
                            nb_critere = len(dataframe['query_param'].get(key))
                            if (
                                    nb_critere > 1):  # ajout des critères sur le types de ressource à la requête , nécessite une concanténation car le nombre de critère varie en fonction des requêtes

                                for j in range(
                                        nb_critere - 1):  # concaténation des critères à rechercher dans la requête cypher
                                    ajout = f"""OR split(filtre{i + 1}.uri, "#")[1] CONTAINS "{value[j + 1]}" """
                                    query = query + ajout

                                query += ")"
                            else:
                                query += ")"

                            query += f""" WITH  collect(ressource) AS filter_{key} """

                            i += 1
                            last_key = key  # on récupère la clé actuel dans la boucle car nous l'utilisons par la suite pour renvoyer la recherche dans le filtre précèdent

                        else:
                            property = "has_" + key.replace("ns0__", "")  # nom de la relation/propiété dans neo4J
                            query += f"""
                            MATCH(ressource:{dataframe['query_param']['ressource']})-[:ns0__{property}]->(filtre{i + 1}:{key}) 
                            WHERE ressource 
                            IN filter_{last_key} 
                            AND (filtre{i + 1}.{key}_label 
                            CONTAINS "{value[0]}"
                            """

                            # lorsque plusieurs champs de données sont sélectionnés pour une même donnée
                            nb_critere = len(dataframe['query_param'].get(key))
                            if (
                                    nb_critere > 1):  # ajout des critères sur le types de ressource à la requête , nécessite une concanténation car le nombre de critère varie en fonction des requêtes

                                for j in range(
                                        nb_critere - 1):  # concaténation des critères à rechercher dans la requête cypher
                                    ajout = f"""OR filtre{i + 1}.{key}_label  CONTAINS "{value[j + 1]}" """
                                    query = query + ajout

                                query += ")"
                            else:
                                query += ")"

                            query += f""" WITH collect(ressource) AS filter_{key}"""

                            i += 1
                            last_key = key  # on récupère la clé actuel dans la boucle car nous l'utilisons par la suite pour renvoyer la recherche dans le filtre précèdent

                    else:
                        if (any(dataframe['champ_donnee_de_toutes_les_nodes'][
                                    'propertyKey'] == key) == True):  # si le filtre essentiel que l'on est entrain de parcourir
                            # dans notre dictionnaire de paramètre correspond au nom d'une propriété/champ de donnée d'un type de node dans neo4j

                            if key not in dataframe['list_donnee_trop_nombreuse']:
                                query += f""" 
                                MATCH (ressource:{dataframe['query_param']['ressource']})
                                WHERE ressource IN filter_{last_key} 
                                AND (ressource.{key}
                                CONTAINS "{value[0]}"
                                """

                                # lorsque plusieurs champs de données sont sélectionnés pour une même donnée
                                nb_critere = len(dataframe['query_param'].get(key))
                                if (
                                        nb_critere > 1):  # ajout des critères sur le types de ressource à la requête , nécessite une concanténation car le nombre de critère varie en fonction des requêtes

                                    for j in range(
                                            nb_critere - 1):  # concaténation des critères à rechercher dans la requête cypher
                                        ajout = f"""OR ressource.{key} CONTAINS "{value[j + 1]}" """
                                        query = query + ajout

                                    query += ")"
                                else:
                                    query += ")"

                                query += f""" WITH  collect(ressource) AS filter_{key} """

                                i += 1
                                last_key = key  # on récupère la clé actuel dans la boucle car nous l'utilisons par la suite pour renvoyer la recherche dans le filtre précèdent

                    # i += 1
                    # last_key = key  # on récupère la clé actuel dans la boucle car nous l'utilisons par la suite pour renvoyer la recherche dans le filtre précèdent

            else:
                print(
                    f"{key} ne peut pas être examiner dans la base de donnée neo4j , le problème provient probablement de la syntaxe dans son label ou sur l'un de ces champs de données accedées par exemple")

    # Ecriture de la fin de la requête, spécifique à chaque projet en fonction de comment sont nommés les champs de données de la ressource du projet en question

    if ("HUMANE" == config_file.items("Project")[0][1]):
        # concanténation de tous les taxons de chaque ressource (fiche pédagogique) filtrée selon les critères essentiels (spécifique au projet HUMANE)
        query += f"""                  
        MATCH (ressource:{dataframe['query_param']['ressource']})-[r:ns0__has_taxon]->(taxon:ns0__taxon)
        WHERE ressource 
        IN filter_{last_key}

        WITH 
        ressource.ns0__identifier as ID, 
        ressource.ns0__title_string AS title, 
        ressource.ns0__description_string as description,  
        collect(taxon.ns0__entry_string) AS phrases
        RETURN 
            ID,
            title,
            description,
            reduce(s = "", phrase IN phrases | s + ' ' + phrase) AS concatenated_taxon
        """

    ###
    # elif("EASING" == config.items("Project")[0][1]):
    #  si le projet en question est EASING alors il faut spécifier le nom des champs de données du type de node ressource a retourner car ils n'auront pas nécessairement le même nom que ceux du projet HUMANE
    ###

    else:  # modèle de requête générique mais qui nécessite cependant d'adapter le nom des champs de données du type de node à retourner

        query += f"""                  
        MATCH (ressource:{dataframe['query_param']['ressource']})
        WHERE ressource 
        IN filter_{last_key}

        WITH 
        ressource.ns0__identifier as ID, 
        ressource.ns0__title_string AS title, 
        ressource.ns0__description_string as description
        RETURN 
            ID,
            title,
            description
        """

    # penser à gérer le cas ou aucune ressource (fiches pedagogique) n'est trouvé

    # print(query)
    dataframe['df'] = dataframe['gds'].run_cypher(query, dataframe['query_param'])
    st.write(
        f"le nombre de ressource restant après première filtration sur les filtres essentiels est de {dataframe['df'].shape[0]} ")
    # print(df)


# calcul recomandation
def tf_idf(title: str, description: str, columns, user_request: list):
    # concaténation du titre et de la description de la ressource et des mots clés (taxon) pour ensuite utilisé TF IDF pour la recommandation basé sur le titre et la description et les taxons de la ressource
    # IL est possible de concanténer d'autre type de données de la ressource afin d'augmenter la précision du modèle dans la mesure ou l'utilisateur utilise les mêmes termes avec la même syntaxe que les données que l'on concatène
    # cas général : on concatène le titre et la description de la ressource
    combined = title + ' ' + description

    ###
    # ajout de la requête utilisateur dans le dataframe des ressources filtrées jusque la pour comparer cette requêtes aux description et titres et mots clés (taxons)
    # des ressources déjà filtrées
    ###

    # Créer une série avec des valeurs nulles sauf pour la dernière colonne
    new_line = pd.Series(
        [None] * (len(columns) - 1) + user_request, index=columns)

    # Ajoute la nouvelle ligne au DataFrame en concaténant le DataFrame original avec la nouvelle ligne
    # dataframe =  pd.concat([dataframe, new_line.to_frame().T], ignore_index=True)
    # st.write(f"df = {dataframe['df']}")

    ###
    # importance de la syntaxe des mots : si un mot n'est pas écrit avec la même orthographe comme il l'est écrit dans les ressource alors il n'est
    #  pas pris en compte dans la recherche comme le montre l'exemple ci dessous qui ne renvoie pas le même résultat
    # user_request = "trigonométrie 3e cosinus sinus géométrie "  # renvoie la ressource : Cosinus et sinus sous GeoGebra Construction des courbes des fonctions sinus et cosinus à partir du cercle trigonométrique.
    # user_request = "trigonométrie 3e cosinu sinu géométrie "   # renvoie la ressource : Apprendre la trigonométrie , Séquence de découverte des fonctions trigonométriques en classe de 3e,
    ###
