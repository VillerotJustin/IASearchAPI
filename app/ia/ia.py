# Import modules from FastAPI
import configparser
import re

# Other Libs
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from numpy import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from starlette import status
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel

# Import internal utilities for database access, authorisation, and schemas
from app.utils.db import neo4j_driver
from app.utils.model import loaded_model

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
    ### Json print for debugging
    # json_object = json.loads(json.dumps(attributes))
    # json_formatted_str = json.dumps(json_object, indent=2)
    # print(json_formatted_str)

    ###
    #   Filter
    # affiche formulaire 1
    # affiche formulaire 2 ?
    ###

    filter_nodes(attributes["filter_config"])

    filtered_nodes = data['filtered_node']

    ###
    #  Recommandation
    ###

    recomandation(filtered_nodes, attributes["recomandation_config"])

    return {
        "number_of_results": len(data["recommended_ressources"]),
        "result": data["recommended_ressources"]
    }


def filter_nodes(filter_params):
    target_node_class = filter_params["label"]
    filters = filter_params["filters"]

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

    with_parameters = filter_params["WITH"]
    with_cypher = "WITH\n"
    i = 0
    for element in with_parameters:
        value = with_parameters[element]
        # print(element, value)
        with_cypher += f"    {element} AS {value}"
        if i + 1 < len(with_parameters):
            with_cypher += ",\n"
        i += 1

    return_parameters = filter_params["RETURN"]
    return_cypher = "RETURN\n"
    for element in return_parameters:
        return_cypher += f"    {element}"
        if i + 1 < len(return_parameters):
            return_cypher += ",\n"
        i += 1

    taxon = filter_params["taxon"]
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

    # print(query)

    with neo4j_driver.session() as session:
        query_result = session.run(query=query)
        data['filtered_node'] = query_result.data()
        # print(f"data {data['query_result']}\n")


def recomandation(filtered_nodes, recomandation_config):
    # initialisation des variables pour notre modèle de recommandation dans le cas ou l'on n'affiche pas la
    # visualisation des données
    # permet d'éviter que la méthode TF IDF ne compte les termes redondant (le,la,les,un,une ect...) qui n'apporte
    # pas d'information sur la ressource
    final_stopwords_list = stopwords.words('french')
    # On concatène le titre et la description de la ressource

    ###
    #   Traitement donné
    ###

    if recomandation_config["recomandation_method"] == "TF-IDF":
        for node in filtered_nodes:
            combined = f"{node['title']} - {node['description']}"
            node["cleaned_desc"] = _removeNonAscii(combined)
            node["cleaned_desc"] = make_lower_case(node["cleaned_desc"])
            node["cleaned_desc"] = remove_stop_words(node["cleaned_desc"])
            node["cleaned_desc"] = remove_punctuation(node["cleaned_desc"])
            node["cleaned_desc"] = remove_html(node["cleaned_desc"])
            node["cleaned_desc"] = remove_letter_e_and_s_from_words(node["cleaned_desc"])
    elif recomandation_config["recomandation_method"] == "TF-IDF+Word2Vec":
        for node in filtered_nodes:
            # le pre processing ici est moins stricte car le modèle Word2vec permet d'avoir une certaine
            # compréhension du contexte de la phrase donc il ne faut pas dénaturer les textes de notre dataframe
            combined = f"{node['title']} - {node['description']}"
            node["cleaned_desc"] = make_lower_case(combined)
            node["cleaned_desc"] = remove_stop_words(node["cleaned_desc"])
            node["cleaned_desc"] = remove_punctuation(node["cleaned_desc"])
            node["cleaned_desc"] = remove_html(node["cleaned_desc"])
    else:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid or missing recomandation method, valid: TF-IDF or TF-IDF+Word2Vec",
            headers={"WWW-Authenticate": "Bearer"})

    ###
    # Recommandation basé sur la description et le titre de la ressource :
    # Converting the title and description into vectors and used bigram
    ###

    data['df'] = pd.DataFrame.from_dict(filtered_nodes)

    if recomandation_config["recomandation_method"] == "TF-IDF":
        # Doc de la fonction :
        # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

        tf = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 3),
            min_df=1,
            stop_words=final_stopwords_list,
            lowercase=True
        )
        # ngram_range = (2,2) ne renvoie que des 0 dans la matrice de similarité
        # tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df = 1, max_df = 0.7,  lowercase = True)
        # possibilité de mettre une valeur forte pour max_df entre 0.7 et 1 à la place  de la liste de stop_word
        # tfidf_matrix = tf.fit_transform(df['cleaned_desc'])

        data['tfidf_matrix'] = tf.fit_transform(data['df']['cleaned_desc'])
    if recomandation_config["recomandation_method"] == "TF-IDF+Word2Vec":

        # Building TFIDF model and calculate TFIDF score
        tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=2, stop_words=final_stopwords_list,
                                lowercase=True)
        # st.session_state['tfidf_matrix'] = tf.fit(st.session_state['df']['combined'])
        # tfidf.fit(st.session_state['df']['combined'])
        tfidf_matrix = tfidf.fit_transform(data['df']['cleaned_desc'])

        # Getting the words from the TF-IDF model

        tfidf_list = dict(zip(tfidf.get_feature_names_out(), list(tfidf.idf_)))
        tfidf_feature = tfidf.get_feature_names_out()  # tfidf words/col-names

        # detailed tutorial of the computation : https://pub.towardsai.net/content-based-recommendation-system-using-word-embeddings-c1c15de1ef95

        # splitting the description into words
        corpus = []
        for words in data['df']['cleaned_desc']:
            corpus.append(words.split())

        len_dataframe = len(data['df']['cleaned_desc'])
        word_not_in_model_vocabulary = []
        # mots qui ne sont pas dans le vocabulaire du modèle, mais présent dans requête utilisateur
        data['are_word_not_in_model_vocabulary'] = False
        # variable pour identifier si des mots de la requête utilisateur ne sont pas présent dans le vocabulaire du
        # modèle

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
                if word in loaded_model.key_to_index and word in tfidf_feature:
                    vec = loaded_model[word]
                    tf_idf = tfidf_list[word] * (desc.count(word) / len(desc))

                    # garanti que le score tf_idf est > à 1 pour que exp(tf_idf^2) augmente significativement le
                    # score tf_idf
                    if tf_idf < 1:
                        tf_idf += 1
                    tf_idf = np.exp(tf_idf ** 2)

                    # if tf_idf < 1 :
                    # tf_idf = tf_idf**5
                    # else:
                    # tf_idf = tf_idf**5

                    if i == len_dataframe - 1:
                        print(f"tf_idf du mot '{word}' = {tf_idf}")

                    sent_vec += (vec * tf_idf)
                    weight_sum += tf_idf
                else:
                    if i == len_dataframe - 1:
                        # lorsque l'on traite la requête utilisateur, c'est-à-dire la dernière ligne du dataframe

                        # ## Traitement du cas ou un mot de la requête utilisateur n'est pas reconnu (lorsque le mot
                        # n'est pas dans le vocabulaire du modèle) ce qui peut réduire grandement la précision des
                        # résultats ##

                        data['are_word_not_in_model_vocabulary'] = True

                        word_not_in_model_vocabulary.append(word)

                        # Si le mot n'est pas reconnu, obtiens le mot le plus proche
                        # mots_proches = st.session_state['model'].similar_by_word(word, topn=1)
                        # mots_proches = st.session_state['model'].similar_by_vector(st.session_state['model'][word] , topn=1)
                        # mot_proche, similarite = mots_proches[0]
                        # print(f"Le mot le plus proche du mot non reconnu '{word}' est '{mot_proche}' avec une similarité de {similarite}")

            if weight_sum != 0:
                sent_vec /= weight_sum
            else:  # si aucun mot n'a été reconnu
                print("aucun mot de la requête n'a été reconnu provoquant des résultats imprécis, "
                      "précisez davantage votre requête en ajoutant de nouveaux mots")
                print("weight sum = 0")

                sent_vec = random.randint(2, size=300)
                # le vecteur de la requete utilisateur est remplacé par un vecteur de nombre aléatoire

            tfidf_vectors.append(sent_vec)
            line += 1

        data['tfidf_matrix+Word2Vec'] = tfidf_vectors
        print(f"Mots de la requête utilisateur suivante :"
              f"   ''' {data['df']['cleaned_desc'].iloc[-1]} '''   "
              f"qui ne sont pas présent dans le vocabulaire du modèle et donc ne sont pas pris en compte"
              f" dans le calcul de la recommandation et réduit la précision : {word_not_in_model_vocabulary}"
              )

    # print(tfidf_matrix)

    ###
    # Result recovery
    ###

    result_recovery(recomandation_config)


def result_recovery(recomandation_config):
    top_n_resultat = recomandation_config["number_of_recomandation"]
    similarity_method = recomandation_config["similarity_method"]
    # choix de la méthode de calcul des similarités entre les ressources et la requête

    ###
    #   Utilisation de différentes métriques de similarité
    ###

    # documentation des métriques de similarité
    # https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics.pairwise
    # https://scikit-learn.org/stable/modules/metrics.html#metrics

    if similarity_method == "cosine similarity":
        # Calculating the similarity measures based on Cosine Similarity
        if recomandation_config["recomandation_method"] == 'TF-IDF':

            tf_idf_similarity_matrix = cosine_similarity(
                data['tfidf_matrix'],
                data['tfidf_matrix']
            )
            # print(cosine_similarity[-1,-1])
            tf_idf_similarity_matrix[-1, -1] = 0
            # correspond a la similarité entre la requete utilisateur et elle meme qui vaut donc 1 et fausse le résultat
            # que l'on cherche en la mettant à 0 elle n'interfere plus avec le résultat de la meilleur recommandation
            # print(type(cosine_similarity)) print(cosine_similarity) print(cosine_similarity.shape)
            top_n_resultat = 1

        if recomandation_config["recomandation_method"] == 'TF-IDF+Word2Vec':
            tf_idf_Word2Vec_similarity_matrix = cosine_similarity(
                data['tfidf_matrix+Word2Vec'],
                data['tfidf_matrix+Word2Vec']
            )
            # print(cosine_similarity[-1,-1])
            tf_idf_Word2Vec_similarity_matrix[-1, -1] = 0
            top_n_resultat = 4  # nombre de ressource recommandée

    elif similarity_method == "euclidean distance":
        # Calcul de similarité avec la distance euclidienne (inverse de la similarité)
        tf_idf_similarity_matrix = 1 / (1 + euclidean_distances(data['tfidf_matrix'], data['tfidf_matrix']))
        tf_idf_similarity_matrix[-1, -1] = 0
        # correspond a la similarité entre la requete utilisateur et elle meme qui vaut donc 1 et fausse le résultat
        # que l'on cherche
        top_n_resultat = 1

        if recomandation_config["recomandation_method"] == 'TF-IDF+Word2Vec':
            tf_idf_Word2Vec_similarity_matrix = 1 / (1 + euclidean_distances(
                data['tfidf_matrix+Word2Vec'],
                data['tfidf_matrix+Word2Vec']
            )
                                                     )
            # print(cosine_similarity[-1,-1])
            tf_idf_Word2Vec_similarity_matrix[-1, -1] = 0
            top_n_resultat = 4  # nombre de ressources recommandé
    elif similarity_method == "dot product":
        # Calcul de similarité avec le produit scalaire
        tf_idf_similarity_matrix = linear_kernel(data['tfidf_matrix'], data['tfidf_matrix'])
        tf_idf_similarity_matrix[-1, -1] = 0
        # correspond a la similarité entre la requete utilisateur et elle meme qui vaut donc 1 et fausse le résultat
        # que l'on cherche
        top_n_resultat = 1

        if recomandation_config["recomandation_method"] == 'TF-IDF+Word2Vec':
            tf_idf_Word2Vec_similarity_matrix = linear_kernel(
                data['tfidf_matrix+Word2Vec'],
                data['tfidf_matrix+Word2Vec']
            )
            # print(cosine_similarity[-1,-1])
            tf_idf_Word2Vec_similarity_matrix[-1, -1] = 0
            top_n_resultat = 4  # nombre de ressource recommandée
    else:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid or missing similarity method, valid: |cosine similarity| or |euclidean distance| or |dot "
                   "product|",
            headers={"WWW-Authenticate": "Bearer"})

    # retour des résultats

    data["recommended_ressources"] = []

    if recomandation_config["recomandation_method"] == 'TF-IDF':
        affichage_resultat_de_la_recommandation(data['df'], similarity_method,
                                                tf_idf_similarity_matrix, 5)

    if recomandation_config["recomandation_method"] == 'TF-IDF+Word2Vec':
        if 'mot_non_present_dans_le_vocabulaire' in data:
            if data['are_word_not_in_model_vocabulary']:
                # affichage du meilleur résultat provenant de la méthode TF_IDF dans le cas ou des mots de les
                # requêtes utilisateurs ne sont pas présent dans le vocabulaire du modèle NLP
                affichage_resultat_de_la_recommandation(
                    data['df'], similarity_method,
                    tf_idf_similarity_matrix,
                    1
                )

            # affichage des autres résultats de la méthode Word2Vec+TF_IDF
            affichage_resultat_de_la_recommandation(
                data['df'],
                similarity_method,
                tf_idf_Word2Vec_similarity_matrix,
                top_n_resultat
            )


def affichage_resultat_de_la_recommandation(df, nom_de_la_methode, matrice_similarite, n_meilleur_resultats):
    # Récupérez la dernière ligne du tableau
    last_row = matrice_similarite[-1]

    # Obtenez les indices triés de la dernière ligne
    # sorted_indices = np.argsort(last_row)
    # top_n = 5  renvoie les n meilleure ressource
    sorted_indices = np.argsort(last_row)[::-1][:n_meilleur_resultats + 5]
    # [::-1] reverses the array returned by argsort() and [:n] gives that last n elements , n+5 au cas ou il soit
    # nécessaire d'avoir besoin de plus de ressource à recommander

    # Trie la dernière ligne en fonction des indices triés
    sorted_last_row = last_row[sorted_indices]

    print("Dernière ligne triée :", sorted_last_row)
    print("Indices d'origine triés :", sorted_indices)
    # print(sorted_last_row[0])

    print(f"Nom de la technique pour calculé la similarité utilisé : {nom_de_la_methode} \n ")

    print(f"similarity matric = {matrice_similarite}")
    print(f"Dernière ligne triée = {sorted_last_row}")
    print(f"liste indices trié = {sorted_indices}")

    for i in range(n_meilleur_resultats):

        # st.write(f"{i+1} ième ressource")

        n = i
        # si la ressource en question a déjà été recommandé on propose celle qui suit dans la liste, si l'indice
        # correspond à la requête utilisateur dans la matrice de similarité car la ligne est rempli de 0 (dans le cas
        # il n'y pas ou trop peu de résultat car des mots ne sont pas reconnus) alors on passe aussi à l'indice suivant
        while (n < (n_meilleur_resultats + 5) and sorted_indices[n] in data["recommended_ressources"]) or (sorted_indices[n] == (df.shape[0] - 1)):
            n += 1
            # st.write(f"df shape = {st.session_state['df'].shape[0]}  et sorted_indices[{n}] = {sorted_indices[n]}")

        print(f"TITRE : {df['title'][sorted_indices[n]]} \nDESCRIPTION : {df['description'][sorted_indices[n]]} \n")


