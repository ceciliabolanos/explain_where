Estamos buscando explicar el por qué se predijo una determinada clase. Ese por qué, en nuestro caso, será encontrar los segmentos más importantes para la predicción de la clase.
Nuestra intuición nos dice que en el segmento más importante es donde debería estar la clase predicha.

Para esto consideramos diferentes métodos para ordenar la importancia de cada segmento en la predicción final.

Para todos tomamos ventanas cada 100ms de tamaño 500ms.

1) Naive: Se considera la importancia de cada ventana como la diferencia entre el score real del audio vs el score al enmascarar (con 0s) esa ventana.

2) RandomForest: Se generan perturbaciones del audio de cantidad num_samples. Cada ventana puede estar o ser enmascarada. Luego se entrena un random forest con los features 1s y 0s (si se enmascaró o no) y 
el y es el score predicho para la clase para el audio perturbado. La importancia de cada feature (cada segmento) es la dada por el método feature_importance de RF.

3) Linear regression: Se generan perturbaciones del audio de cantidad num_samples. Cada ventana puede estar o ser enmascarada. Luego se entrena una regresión lineal con los features 1s y 0s (si se enmascaró o no) y 
el y es el score predicho para la clase para el audio perturbado. La importancia de cada feature (cada segmento) es el coeficiente que lo acompaña en la regresión.


First step:

Process the dataframe, we need a df with the audios to process and the labels we have segmented for each class. 
Here we have a problem: There's a child_id so, perhaps, the same audio was segmented with a different label but train with other (for example: speech and man speech). 
For that reason we leave a column 'positive_labels' and 'father_labels'. We should use only 'positive_labels'.

~/explain_where$ python preprocess/dataframe.py

-----------------------------------------------

Primer paso: carpeta preprocess/

Acá encontramos 2 scripts importantes.
 El primero que hay q realizar es dataframe.py aca hacemos un preprocesamiento del dataframe que tenemos segmentado en donde obtenemos el ID base, los labels que hay que predecir (En este caso los labels son todos movidos a un nivel 2 en el arbol de ontologia) y sacamos los labels nan, que no encontramos su label en nombre. Aca tambien tenemos la duracion de cada label para despues filtrar aquellos que duren mas de 20% de la senal. Path to save this csv: '/home/cbolanos/experiments/audioset/labels/labels_segments.csv'
y el segundo audios.py para poder guardarnos las predicciones de cada modelo (AST y Yamnet). El segundo

Despues tenemos run_everything para correr toda la generacion de data y random fores y run_Example que nos sirve cuando queremos el video de resultado entonces solo corremos una carpeta y no todo