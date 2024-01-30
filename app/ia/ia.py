# Import modules from FastAPI
import configparser
import re

# Other Libs
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import linear_kernel
from starlette import status

# Import internal utilities for database access, authorisation, and schemas
from app.utils.db import neo4j_driver
from app.utils.environment import settings
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

    # Parcourt chaque mot et supprime-les "e" et "s" à la fin de chaque mot
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


###
# IA
# is it worth it, I don't know
###

data['node_label'] = get_all_labels()
data['property_keys'] = get_every_property_keys_no_async()


# Matières = ns0__setSpec

# API request
@router.post('/search')
async def AISearch(attributes: dict):
    # Parameter Recovery
    AISearch_part1(attributes)

    # Filtering nodes
    AISearch_part2()

    # Recommandation Setup
    df = AISearch_part3()

    # Recommandation Calculation
    AISearch_part4(df)
    # print("pre return")
    # print(len(data["recommended_ressources"]))
    # print(data["recommended_ressources"])
    return {
        "number_of_results": len(data["recommended_ressources"]),
        "result": data["recommended_ressources"]
    }


def AISearch_part1(attributes):
    # Sélectionnez le label de la node symbolisant les ressources dans la base de donnée neo4j, c'est-à-dire les nœuds
    # avec un titre et une description comme champ de donnée de la ressource | Ressource = ['choix_node_ressource']

    data['ressource'] = attributes['ressource']
    data['ressource_id'] = attributes['ressource_id']
    data['sub_ressource'] = attributes['sub_ressource']

    # Sélectionnez les champs de donnée de la ressource pour vos filtres essentiels | ?

    data['choice_of_resource_data_fields'] = attributes['choice_of_resource_data_fields']

    # Sélectionnez le label des nœuds pour vos filtres essentiels (choisissez au moins 1 paramètre) : Filters

    data['essential_filters'] = attributes['essential_filters']

    # Choisir le nombre maximal de critères pour chaque donnée,
    # c'est-à-dire le nombre de choix dans le menu déroulant pour chaque filtre essentiel :

    data['nb_criteria_for_each_data'] = attributes['nb_criteria_for_each_data']
    data['threshold_percentage_max_filter'] = attributes['threshold_percentage_max_filter']

    # Sélectionnez le critère du filtre essentiel ns0__setSpec = PHY

    data['filters_criteria'] = attributes['filters_criteria']

    # check if no error in json and there is the good number of criteria
    if len(data['filters_criteria']) != len(data['essential_filters']):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid number of criteria",
            headers={"WWW-Authenticate": "Bearer"})

    data["WITH_RETURN"] = attributes['WITH_RETURN']
    data['special_return'] = attributes['special_return']

    # Entrez votre requête, insérez tous les mots que doit contenir la ressource qui correspond à vos besoins
    # Recomandation → |la trajectoire de la courbe est parabolique| ← ['requete_utilisateur']

    data['user_request'] = attributes['user_request']

    # Methode de calcul = TF-IDF+Word2Vec

    data['method'] = attributes['method']

    # Number of result

    data['n_result'] = attributes['n_result']

    # Similarity method

    data['similarity_method'] = attributes['similarity_method']
    similarity_method_possible_value = ["cosine", "euclidean", "dot"]
    if data['similarity_method'] not in similarity_method_possible_value:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid similarity method, possible values:{similarity_method_possible_value}",
            headers={"WWW-Authenticate": "Bearer"})


# Filter Ressource with filters
def AISearch_part2():
    query = ""
    last_filter = ""
    for i in range(len(data['essential_filters'])):
        print(f"Filter{i}")
        print(data['essential_filters'][i])
        print(data['filters_criteria'][i])
        filteri = data['essential_filters'][i]
        filter_criteria = data['filters_criteria'][i]
        relationship = settings.DB_PREFIX + "has_" + filteri.replace(settings.DB_PREFIX, "")
        filter_query = f"MATCH(ressource: {data['ressource']})-[: {relationship}]->(filtre{i}:{filteri})\nWHERE "

        if last_filter != "":
            filter_query += f"ressource IN {last_filter} AND ("

        if len(filter_criteria) > 1:
            filter_query += "("
            for criteria in filter_criteria:
                filter_query += f"(split(filtre{i}.uri, '#')[1] CONTAINS '{criteria}') OR "
            filter_query = filter_query[:-4] + ")"
        else:
            filter_query += f"split(filtre{i}.uri, '#')[1] CONTAINS '{filter_criteria[0]}'"

        if last_filter != "":
            filter_query += ")"

        filter_query += f"\nWITH collect(ressource) AS filter_{filteri}\n\n"
        last_filter = f"filter_{filteri}"
        query += filter_query

    sub_ressource = (f"-[r: {settings.DB_PREFIX + 'has_' + data['sub_ressource'].replace(settings.DB_PREFIX, '')}]"
                     f"->(taxon:{data['sub_ressource']})")

    final_query = f"MATCH(ressource: {data['ressource']}){sub_ressource}\nWHERE ressource IN {last_filter}\n"

    with_cypher = "WITH\n"
    return_cypher = "RETURN\n"
    i = 0
    for element in data['WITH_RETURN']:
        value = data['WITH_RETURN'][element]
        # print(element, value)
        with_cypher += f"    {element} AS {value}"
        return_cypher += f"    {value}"
        if i + 1 < len(data['WITH_RETURN']):
            with_cypher += ",\n"
            return_cypher += ",\n"
        else:
            with_cypher += "\n"
        i += 1

    final_query += with_cypher + return_cypher

    query += final_query

    # print(f"query : \n{query}")

    with neo4j_driver.session() as session:
        query_result = session.run(query=query)
        data['filtered_node'] = query_result.data()


def AISearch_part3():
    data['final_stopwords_list'] = stopwords.words('french')

    df = pd.DataFrame.from_dict(data['filtered_node'])

    df['combined'] = df['title'] + ' ' + df['description']

    ###
    # ajout de la requête utilisateur dans le dataframe des ressources filtrées jusqu'à là pour comparer ces
    # requêtes à la description et titres et mots clés (taxons) des ressources déjà filtrées ##
    ###

    # Créer une série avec des valeurs nulles sauf pour la dernière colonne
    nouvelle_ligne = pd.Series(
        [None] * (len(df.columns) - 1) + [data['user_request']],
        index=df.columns
    )

    # Ajoute la nouvelle ligne au DataFrame en concaténant le DataFrame original avec la nouvelle ligne
    df = pd.concat([df, nouvelle_ligne.to_frame().T], ignore_index=True)

    if data['method'] == 'TF-IDF':
        return TF_IDF(df)
    elif data['method'] == 'TF-IDF+Word2Vec':
        return TF_IDF_Word2Vec(df)
    else:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid or missing recomandation method, valid: TF-IDF or TF-IDF+Word2Vec",
            headers={"WWW-Authenticate": "Bearer"})


def TF_IDF(df):
    # Applying all the functions and storing as a cleaned_desc remplace la colonne dans le dataframe contenant la
    # concaténation des données utilisés pour TF IDF au lieu d'en créer une nouvelle et de dédoubler nos données évite
    # d'avoir une nouvelle colonne dans le dataframe pour alléger le nombre de données
    df['combined'] = df['combined'].apply(_removeNonAscii)
    df['combined'] = df['combined'].apply(func=make_lower_case)
    df['combined'] = df['combined'].apply(func=remove_stop_words)
    df['combined'] = df['combined'].apply(func=remove_punctuation)
    df['combined'] = df['combined'].apply(func=remove_html)
    df['combined'] = df['combined'].apply(func=remove_letter_e_and_s_from_words)

    # print(f" Dataset des description nettoyées : \n\n {df['combined']}") # les accents sont aussi enlevé
    # print(f" Dataset des description nettoyées : \n\n {df['cleaned_desc']}") # les accents sont aussi enlevé

    # Doc de la fonction
    # https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

    stopwords = data['final_stopwords_list']
    tf = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 3),
        min_df=1,
        stop_words=stopwords,
        lowercase=True
    )
    # ngram_range = (2,2) ne renvoie que des 0 dans la matrice de similarité tf = TfidfVectorizer (analyzer='word',
    # ngram_range=(1, 3), min_df = 1, max_df = 0.7, lowercase = True) possibilité de mettre une valeur forte pour
    # max_df entre 0.7 et 1 à la place de la liste de stop_word

    data['tfidf_matrix'] = tf.fit_transform(df['combined'])
    return df


def TF_IDF_Word2Vec(df):
    # Applying all the functions and storing as a cleaned_desc le pre processing ici est moins stricte car le modèle
    # Word2vec permet d'avoir une certaine compréhension du contexte de la phrase donc il ne faut pas dénaturer les
    # textes de notre dataframe st.session_state['df']['combined'] = st.session_state['df']['combined'].apply(
    # _removeNonAscii)
    df['combined'] = df.combined.apply(func=make_lower_case)
    df['combined'] = df.combined.apply(func=remove_stop_words)
    df['combined'] = df.combined.apply(func=remove_punctuation)
    df['combined'] = df.combined.apply(func=remove_html)

    # Pas besoin de charger le model, il est chargé au lancement de l'appli

    # Building TFIDF model and calculate TFIDF score
    tfidf = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 3),
        min_df=2,
        stop_words=data['final_stopwords_list'],
        lowercase=True
    )
    # st.session_state['tfidf_matrix'] = tf.fit(st.session_state['df']['combined'])
    # tfidf.fit(st.session_state['df']['combined'])
    data['tfidf_matrix'] = tfidf.fit_transform(df['combined'])

    # Getting the words from the TF-IDF model

    tfidf_list = dict(zip(tfidf.get_feature_names_out(), list(tfidf.idf_)))
    tfidf_feature = tfidf.get_feature_names_out()  # tfidf words/col-names

    # detailed tutorial of the computation :
    # https://pub.towardsai.net/content-based-recommendation-system-using-word-embeddings-c1c15de1ef95

    # splitting the description into words
    corpus = []
    for words in df['combined']:
        corpus.append(words.split())

    len_dataframe = len(df['combined'])
    word_not_present_in_vocabulary = []
    # mots qui ne sont pas dans le vocabulaire du modèle, mais présent dans la requête utilisateur
    data['are_word_not_present_in_vocabulary'] = False
    # variable pour identifier si des mots de la requête utilisateur ne sont pas présent dans le vocabulaire du modèle

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
                # garanti que le score tf_idf est > à 1 pour que exp(tf_idf^2) augmente significativement le score
                # tf_idf
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
                # lorsque l'on traite la requête utilisateur, c'est-à-dire la dernière ligne du dataframe
                if i == len_dataframe - 1:
                    # Traitement du cas ou un mot de la requête utilisateur n'est pas reconnu (lorsque le mot
                    # n'est pas dans le vocabulaire du modèle) ce qui peut réduire grandement la précision des
                    # résultats

                    data['are_word_not_present_in_vocabulary'] = True

                    word_not_present_in_vocabulary.append(word)

                    # Si le mot n'est pas reconnu, obtiens le mot le plus proche mots_proches = st.session_state[
                    # 'model'].similar_by_word(word, topn=1) mots_proches = st.session_state[
                    # 'model'].similar_by_vector(st.session_state['model'][word] , topn=1) mot_proche, similarite =
                    # mots_proches[0] print(f"Le mot le plus proche du mot non reconnu '{word}' est '{mot_proche}'
                    # avec une similarité de {similarite}")

        if weight_sum != 0:
            sent_vec /= weight_sum
        else:  # si aucun mot n'a été reconnu
            print("aucun mot de la requête n'a été reconnu provoquant des résultats imprécis, précisez davantage "
                  "votre requête en ajoutant de nouveaux mots")
            print("weight sum = 0")
            from numpy import random
            sent_vec = random.randint(2, size=300)
            # le vecteur de la requete utilisateur est remplacé par un vecteur de nombre aléatoire

        tfidf_vectors.append(sent_vec)
        line += 1

    data['tfidf_matrix+Word2Vec'] = tfidf_vectors
    print(f"Mots de la requête utilisateur suivante :   ''' {df['combined'].iloc[-1]} '''   qui ne sont pas présent "
          f"dans le vocabulaire du modèle et donc ne sont pas pris en compte dans le calcul de la recommandation et "
          f"réduit la précision : {word_not_present_in_vocabulary}")

    return df


# Recommandation Calculate
def AISearch_part4(df):
    if data['similarity_method'] == "cosine":
        # Calculating the similarity measures based on Cosine Similarity
        tf_idf_similarity_matrix = cosine_similarity(data['tfidf_matrix'], data['tfidf_matrix'])
        # print(cosine_similarity[-1,-1])
        tf_idf_similarity_matrix[-1, -1] = 0
        # correspond à la similarité entre la requete utilisateur et elle meme qui vaut donc 1 et fausse le résultat
        # que l'on cherche en la mettant à 0 elle n'interfère plus avec le résultat de la meilleure recommandation
        # print (type(cosine_similarity)) print(cosine_similarity) print (cosine_similarity.shape)
        top_n_resultat = 1

        if data['method'] == 'TF-IDF+Word2Vec':
            tf_idf_Word2Vec_similarity_matrix = cosine_similarity(data['tfidf_matrix+Word2Vec'],
                                                                  data['tfidf_matrix+Word2Vec'])
            # print(cosine_similarity[-1,-1])
            tf_idf_Word2Vec_similarity_matrix[-1, -1] = 0
            top_n_resultat = 4  # nombre de ressources recommandée
    elif data['similarity_method'] == "euclidean":
        # Calcul de similarité avec la distance euclidienne (inverse de la similarité)
        tf_idf_similarity_matrix = 1 / (1 + euclidean_distances(data['tfidf_matrix'],
                                                                data['tfidf_matrix']))
        tf_idf_similarity_matrix[-1, -1] = 0
        # correspond à la similarité entre la requete utilisateur et elle meme qui vaut donc 1 et fausse le résultat
        # que l'on cherche
        top_n_resultat = 1

        if data['method'] == 'TF-IDF+Word2Vec':
            tf_idf_Word2Vec_similarity_matrix = 1 / (1 + euclidean_distances(data['tfidf_matrix+Word2Vec'],
                                                                             data['tfidf_matrix+Word2Vec']))
            # print(cosine_similarity[-1,-1])
            tf_idf_Word2Vec_similarity_matrix[-1, -1] = 0
            top_n_resultat = 4  # nombre de ressources recommandé
    elif data['similarity_method'] == "dot":
        # Calcul de similarité avec le produit scalaire
        tf_idf_similarity_matrix = linear_kernel(data['tfidf_matrix'], data['tfidf_matrix'])
        tf_idf_similarity_matrix[-1, -1] = 0
        # correspond à la similarité entre la requete utilisateur et elle meme qui vaut donc 1 et fausse le résultat
        # que l'on cherche
        top_n_resultat = 1

        if data['method'] == 'TF-IDF+Word2Vec':
            tf_idf_Word2Vec_similarity_matrix = linear_kernel(data['tfidf_matrix+Word2Vec'],
                                                              data['tfidf_matrix+Word2Vec'])
            # print(cosine_similarity[-1,-1])
            tf_idf_Word2Vec_similarity_matrix[-1, -1] = 0
            top_n_resultat = 4  # nombre de ressources recommandées

    # Affichage des résultats

    data['recommended_ressources'] = []  # liste des ressources déjà recommander

    if data['method'] == 'TF-IDF':
        return pick_recomandation(
            df,
            data['similarity_method'],
            tf_idf_similarity_matrix,
            5
        )

    if data['method'] == 'TF-IDF+Word2Vec':
        if 'are_word_not_present_in_vocabulary' in data:
            if data['are_word_not_present_in_vocabulary']:
                # affichage du meilleur résultat provenant de la méthode TF_IDF dans le cas ou des mots des
                # requêtes utilisateurs ne sont pas présent dans le vocabulaire du modèle NLP
                return pick_recomandation(
                    df,
                    data['similarity_method'],
                    tf_idf_similarity_matrix,
                    1
                )

            # affichage des autres résultats de la méthode Word2Vec+TF_IDF
            pick_recomandation(
                df,
                data['similarity_method'],
                tf_idf_Word2Vec_similarity_matrix,
                top_n_resultat
            )


def pick_recomandation(df, similarity_method, similarity_matrix, n_best_results):
    # Récupérez la dernière ligne du tableau
    last_row = similarity_matrix[-1]

    # Obtenez les indices triés de la dernière ligne
    # sorted_indices = np.argsort(last_row)
    # top_n = 5
    # #renvoie les n meilleure ressource
    sorted_indices = np.argsort(last_row)[::-1][:n_best_results + 5]
    # [: :-1] reversés the array returned by argsort() and [:n] gives that last n elements , n+5 au cas ou il soit
    # nécessaire d'avoir besoin de plus de ressource à recommander

    # Trie la dernière ligne en fonction des indices triés
    sorted_last_row = last_row[sorted_indices]

    print("Dernière ligne triée :", sorted_last_row)
    print("Indices d'origine triés :", sorted_indices)
    # print(sorted_last_row[0])

    print(f"Nom de la technique pour calculé la similarité utilisé : {similarity_method} \n ")

    print(f"similarity matric = {similarity_matrix}")
    print(f"Dernière ligne triée = {sorted_last_row}")
    print(f"liste indices trié = {sorted_indices}")

    recommended_ressources = []

    for i in range(n_best_results):

        # st.write(f"{i+1} ième ressource")

        n = i
        # si la ressource en question a déjà été recommandé, on propose celle qui suit dans la liste, si l'indice
        # correspond à la requête utilisateur dans la matrice de similarité, car la ligne est rempli de 0 (dans le
        # cas il n'y pas ou trop peu de résultat, parce que des mots ne sont pas reconnus) alors, on passe aussi à
        # l'indice suivant
        while ((n < (n_best_results + 5) and sorted_indices[n] in recommended_ressources)
               or (sorted_indices[n] == (df.shape[0] - 1))):
            n += 1
            # st.write(f"df shape = {st.session_state['df'].shape[0]}  et sorted_indices[{n}] = {sorted_indices[n]}")

        # print(f"TITRE : {df['title'][sorted_indices[n]]} \nDESCRIPTION : {df['description'][sorted_indices[n]]} \n")

        # si le projet n'est pas HUMANE alors, il sera nécessaire de changer dans la requête le nom du champ de donnée
        # 'n.ns0__identifier' et 'ns0__resource_identifier'

        # requête permettant de récupérer toutes les données de la ressource en question,
        # en particulier pour récupérer son URL renvoyant sur édubase pour le projet HUMANE

        # query = f"MATCH (n:{data['ressource']}) WHERE n.{data['ressource_id']} = '{df['ID'][sorted_indices[n]]}'\n"
        # query += f"RETURN "
        # for element in data['special_return']:
        #     query += f"n.{element}, "
        # query = query[:-2]
        #
        # with neo4j_driver.session() as session:
        #     url_ressource = session.run(query=query)

        # Affiche le lien hypertexte Streamlit Artifact
        # print(f"[Cliquez ici pour voir la fiche pédagogique]({url_ressource})")
        #
        # # les liens des images décrivant la fiche pédagogique sur le projet HUMANE
        # # sont de la forme https://thumbnails.appcraft.events/edubase/thumbnail/6551.jpg avec le
        # # nombre 6551 avant le .jpg correspondant à l'id de la ressource
        #
        # ID_ressource = re.search(r'\d+$', url_ressource.iat[0, 0])
        # # Utilise une expression régulière pour trouver le nombre à la fin du lien, Cette expression régulière (
        # # \d+$) recherche une séquence de chiffres (\d) à la fin ($) de la chaîne. Dans cet exemple, cela extrairait
        # # le nombre "6551" du lien
        # ID_ressource = ID_ressource.group()
        # # st.markdown(f"![Alt Text](https://thumbnails.appcraft.events/edubase/thumbnail/{ID_ressource}.jpg)")

        # print(f"![Image](https://thumbnails.appcraft.events/edubase/thumbnail/{ID_ressource}.jpg)")
        # print(f"=={n}===")
        recommended_ressources.append(sorted_indices[n])
        # print("=={n}===")
    find_recommended_ressources(df, recommended_ressources)


# Recover recommended ressources
def find_recommended_ressources(df, recommended_ressources_id_list):
    recommended_ressources = []
    for id in recommended_ressources_id_list:
        recommended_ressources.append(df.iloc[id])
    data['recommended_ressources'] = recommended_ressources
