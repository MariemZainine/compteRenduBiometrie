## Résultats et Expérimentations ##
1. Configuration expérimentale

Pour ce TP, nous avons utilisé :

Détection de visage : MTCNN

Modèle d’embedding : FaceNet (CNN pré-entraîné)

Dimension de l’embedding : 512

Mesures de similarité :

Distance Euclidienne

Similarité Cosinus

Chaque image est transformée en vecteur d’embedding représentant les caractéristiques du visage.
--- 
2. Comparaison : Distance Euclidienne vs Similarité Cosinus

Deux méthodes de comparaison ont été testées.

* Distance Euclidienne
==> Plus la distance est petite, plus les visages sont similaires.
* Similarité Cosinus
==>  Plus la valeur est proche de 1, plus les visages sont similaires.
## Observation

Les deux méthodes donnent des résultats cohérents.
Cependant, la distance euclidienne est plus intuitive car elle mesure directement l'écart entre deux embeddings.
--- 
3. Étude de l'effet du seuil

Nous avons testé plusieurs valeurs de seuil pour observer l'impact sur la décision.
Analyse

*  Un seuil trop faible augmente les faux rejets.

* Un seuil trop élevé peut provoquer des fausses acceptations

4. Faux rejets et fausses acceptations
## Faux rejet (False Rejection)

Un utilisateur légitime est rejeté.
cause: seuil trop faible
## Fausse acceptation (False Acceptance)

Un utilisateur non autorisé est accepté.
cause: seuil trop élevé.
## Conclusion

Les expériences montrent que :

FaceNet permet d'obtenir des embeddings robustes des visages.

La distance euclidienne est efficace pour comparer les visages.

Le choix du seuil est crucial pour éviter :

les faux rejets

les fausses acceptations.