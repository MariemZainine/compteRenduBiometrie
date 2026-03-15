# Questions d’Analyse
---
## Pourquoi PCA nécessite un bon alignement des visages ?
La méthode PCA (Analyse en Composantes Principales) fonctionne en représentant chaque image de visage comme un vecteur de pixels. Chaque pixel occupe une position fixe dans ce vecteur.

Pour que la comparaison entre les visages soit correcte, les caractéristiques du visage (yeux, nez, bouche) doivent apparaître aux mêmes positions dans toutes les images.

Si les visages ne sont pas bien alignés (par exemple un visage légèrement tourné ou décalé), les pixels ne correspondent plus aux mêmes parties du visage. Dans ce cas, la méthode PCA interprète ces différences comme des différences entre individus alors qu'il s'agit simplement d'un problème de position.

Ainsi, un mauvais alignement peut provoquer une augmentation artificielle de la distance entre deux images du même visage, ce qui dégrade la performance du système de reconnaissance.
---
## Que se passe-t-il si k est trop faible ?
Le paramètre 
𝑘
k correspond au nombre de composantes principales conservées dans la projection PCA.

Lorsque 
𝑘
k est trop faible, seules quelques composantes principales sont utilisées pour représenter les visages. Cela signifie que la réduction de dimension est très forte et que beaucoup d'informations importantes sur les visages sont perdues.

Les conséquences sont :

une représentation trop simplifiée des visages

une perte d'informations discriminantes entre les individus

une augmentation des erreurs de reconnaissance

Dans ce cas, plusieurs visages différents peuvent être projetés dans des positions proches dans l’espace PCA, ce qui diminue la capacité du système à distinguer les personnes.
---
## Que se passe-t-il si k est trop élevé ?
Lorsque 
𝑘
k est trop élevé, on conserve un grand nombre de composantes principales, parfois presque toutes les dimensions de l’espace initial.

Dans ce cas, la réduction de dimension devient moins efficace et le modèle peut également conserver :

du bruit présent dans les images

des variations dues à l’éclairage

des détails inutiles pour la reconnaissance

Cela peut conduire à un phénomène appelé surapprentissage (overfitting), où le modèle s'adapte trop aux images d'entraînement et devient moins performant sur de nouvelles images.

De plus, le coût de calcul augmente car l'espace de représentation devient plus grand.
---
## Pourquoi la distance Euclidienne est adaptée dans l’espace PCA ?
Après l'application de PCA, les visages sont projetés dans un nouvel espace appelé espace des Eigenfaces.

Dans cet espace :

les axes sont orthogonaux

les composantes sont non corrélées

les données sont représentées par des vecteurs de caractéristiques

La distance Euclidienne permet alors de mesurer la similarité entre deux visages projetés. Elle calcule la distance entre deux vecteurs dans cet espace.

Plus la distance Euclidienne est petite, plus les deux visages sont similaires.
À l’inverse, une grande distance indique que les visages sont probablement différents.

C’est pourquoi cette mesure est simple, efficace et largement utilisée pour la reconnaissance dans l’espace PCA.
---
## Quelles sont les limites d’Eigenfaces face aux variations d’illumination ?
L’une des principales limitations de la méthode Eigenfaces est sa sensibilité aux variations d’illumination.

En effet, PCA analyse les images en fonction des valeurs des pixels. Or, les conditions d’éclairage peuvent modifier fortement ces valeurs.

Par exemple :

une lumière venant d’un côté du visage peut créer des ombres

une lumière trop forte peut éclaircir certaines zones

un environnement sombre peut réduire le contraste

Ces variations peuvent être interprétées par le système comme des différences entre visages, même si les images appartiennent à la même personne.

Ainsi, deux images du même individu prises sous des éclairages différents peuvent produire des vecteurs très différents dans l’espace PCA, ce qui peut conduire à des erreurs de reconnaissance.

Pour réduire cet effet, certaines techniques de prétraitement peuvent être utilisées, comme la normalisation d’histogramme ou l’égalisation de contraste.
---
---

**La méthode Eigenfaces basée sur PCA est une approche efficace pour la reconnaissance faciale car elle permet de réduire la dimension des images tout en conservant les informations principales du visage. Toutefois, ses performances dépendent fortement de la qualité des images, notamment de l’alignement des visages, du choix du nombre de composantes principales et des conditions d’illumination. **