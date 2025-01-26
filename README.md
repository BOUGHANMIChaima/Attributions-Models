# Attributions Models using ChannelAttribution Package based on Markov Chain 

## Data
Cette base de données a suivi les parcours des clients du **1er juillet 2018** au **31 juillet 2018** et a enregistré en détail le moment où chaque publicité s'est affichée, quel client elle a atteint et si le client a réussi à se convertir.

Par exemple, pour le premier cookie, son parcours comprend quatre états : Instagram \> Online Display \> Onlin Display \> Online Display. Malheureusement, l'utilisateur n'a pas converti pendant la période observée, et donc la conversion et la valeur de conversion sont toutes deux égales à 0.

![apperçu Data](https://github.com/BOUGHANMIChaima/Attributions-Models/blob/main/data__.png)

## Preprocessing
Afin d’appliquer le modèle de chaîne de Markov, nous devons transformer nos données et créer une variable de chemin __path__
comme dans notre exemple précédent. Ici, j’ai utilisé data.table pour traiter nos données car il peut traiter les données beaucoup plus rapidement que data.frame.

![](https://github.com/BOUGHANMIChaima/Attributions-Models/blob/main/conversion_value.png)

## Markov Chain Modeling
R dispose d’un excellent package conçu pour l’attribution des canaux, appelé __ChannelAttribution__.
Il peut être utilisé pour construire des modèles basés sur des approches heuristiques et markoviennes, respectivement. Pour évaluer les résultats de la chaîne de Markov, j’exécute simultanément des modèles heuristiques et je les considère comme des modèles de base.

![](https://github.com/BOUGHANMIChaima/Attributions-Models/blob/main/transition_proba.png)
