from gensim.models import KeyedVectors
from environment import settings

# https://fasttext.cc/docs/en/crawl-vectors.html
# pre-trained word vectors for 157 languages, trained on Common Crawl and Wikipedia
# using fastText. These models were trained using CBOW with position-weights,
# in dimension 300, with character n-grams of length 5, a window of size 5 and 10 negatives.
# with three new word analogy datasets, for French, Hindi and Polish.

if settings.LOAD_LOCAL_MODEL:
    frWac_model_path = "file://"+"../Models/cc.fr.300.vec"
else:
    frWac_model_path = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.vec.gz"

# model = FastText.load_fasttext_format(model_path)  # ce modèle est trop gourmand en mémoire pour être utilisé

# ## vocabulaire limité à 50 000 mots, cela couvre une grande partie du vocabulaire courant et est souvent
# suffisant pour de nombreuses applications. La plupart des mots fréquemment utilisés dans le langage quotidien,
# ainsi que beaucoup de termes spécialisés, sont généralement inclus dans un modèle avec un vocabulaire de cette
# taille. Pour de nombreuses tâches, l'utilisation d'un modèle avec un vocabulaire plus important peut ne pas
# apporter d'amélioration significative, tout en augmentant les exigences en mémoire. possibilité d'augmenter ou
# réduire le nombre de mots dans le paramètre limit=50 000 ## model = KeyedVectors.load_word2vec_format(
# frWac_model_path, binary=False, limit=50000)
print("======================================\nloading model...\n======================================")
loaded_model = KeyedVectors.load_word2vec_format(frWac_model_path, binary=False, limit=50000)
# loaded_model = None
print("======================================\nmodel loaded\n======================================")

# ## la nécessité de ré entrainer le modèle avec notre dataset est à étudier car ce n'est pas forcément
# nécessaire cependant il n'est pas possible de ré-entrainer ce modèle en question avec notre dataset ,
# Si l'ensemble de données est assez différent du corpus sur lequel le modèle pré-entraîné a été construit,
# la ré-entraînement peut améliorer les performances du modèle pour votre tâche particulière un ré entrainement
# peut notamment ajouter de nouveaux mots si notre dataset contient des mots qui n'existent pas dans le modele

# pour réaliser un ré entrainement il faut aussi prendre en compte la taille de notre dataset, s'il est trop
# grand il faut prendre en compte les ressources nécessaires qui peuvent être gourmande pour le ré entrainer ,
# s'il est trop petit alors le modèle va se surajuster à notre dataset et ne généralisera pas bien à de nouvelle
# donnée => dans la mesure ou l'objectif est de réaliser un système de recommandation générique qui doit
# fonctionner sur plusieurs projets avec des données différentes entre chaque projet alors il ne semble pas
# nécessaire de ré entrainer le modèle et de simplement garder le modèle d'origine qui est plus général ##
