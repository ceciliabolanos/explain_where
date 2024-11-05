Escribamos lo que estamos haciendo: 


Estamos buscando explicar el por qué se predijo una determinada clase. Ese por qué, en nuestro caso, será encontrar los segmentos más importantes para la predicción de la clase.
Nuestra intuición nos dice que en el segmento más importante es donde deberia estar la clase predicha.

Para esto consideramos diferentes métodos para ordenar la importancia de cada segmento en la predicción final.

Para todos tomamos ventanas cada 100ms de tamaño 500ms.

1) Naive: Se considera la importancia de cada ventana como la diferencia entre el score real del audio vs el score al enmascarar (con 0s) esa ventana.

2) RandomForest: Se generan perturbaciones del audio de cantidad num_samples. Cada ventana puede estar o ser enmascarada. Luego se entrena un random forest con los features 1s y 0s (si se enmascaró o no) y 
el y es el score predicho para la clase para el audio perturbado. La importancia de cada feature (cada segmento) es la dada por el método feature_importance de RF.

3) Linear regression: Se generan perturbaciones del audio de cantidad num_samples. Cada ventana puede estar o ser enmascarada. Luego se entrena una regresión lineal con los features 1s y 0s (si se enmascaró o no) y 
el y es el score predicho para la clase para el audio perturbado. La importancia de cada feature (cada segmento) es el coeficiente que lo acompaña en la regresión.


