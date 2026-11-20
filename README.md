# Dataset de uso de casco en motociclistas

Combinación de 6 datasets públicos unificados en **4 clases** para detectar motos y el estado del casco:

| id | clase | qué incluye |
|---|---|---|
| 0 | `motocicleta` | moto con conductor |
| 1 | `casco_correcto` | casco bien puesto (integral o medio casco) |
| 2 | `casco_mal_puesto` | casco mal puesto o inválido, casco en la mano, gorra, pañuelo o turbante en vez de casco |
| 3 | `sin_casco` | conductor o pasajero sin casco |

## Qué hay en este repo

```
descargar_datasets.py        # baja los datasets de Roboflow y Mendeley
unificar_etiquetas.py        # los une en las 4 clases y genera los recortes
datasets_casco/unificado/
├── data.yaml                # listo para entrenar YOLO (Ultralytics)
├── labels/{train,val,test}/ # etiquetas YOLO unificadas
├── recortes/{train,val,test}/<clase>/   # cabezas recortadas para entrenar un clasificador (CNN)
└── resumen.txt              # conteos por fuente y por clase
```

**Las imágenes y los videos no están en el repo**, porque pesan unos 5 GB. Para tenerlos:

```bash
python descargar_datasets.py --api-key TU_API_KEY_DE_ROBOFLOW
python unificar_etiquetas.py
```

- La API key de Roboflow es gratis: roboflow.com → Settings → API Keys.
- Solo hace falta Python 3.9+ y `pillow` (`pip install pillow`).
- `--con-mendeley-grande` agrega el dataset opcional Helmet Wearing (27,560 imágenes, unos 2.5 GB).

Después de correr los scripts, entrenar el detector es:

```bash
yolo detect train data=datasets_casco/unificado/data.yaml model=yolov8n.pt
```

## Números

13,303 imágenes (train 10,242 · val 1,944 · test 1,117).

| fuente | imágenes | motocicleta | casco_correcto | casco_mal_puesto | sin_casco |
|---|---:|---:|---:|---:|---:|
| EdgeVision | 2,392 | 3,793 | 1,671 | — | 2,810 |
| Motorcycle Helmet OD | 3,607 | — | 10,398 | — | 4,900 |
| Mapua | 3,749 | — | 3,987 | 564 | 1,318 |
| LDrago | 2,258 | — | 2,844 | 286 | 913 |
| Proper Helmet Usage | 950 | — | 434 | 457 | — |
| Helmet Violation | 347 | 70 | 362 | 1 | 181 |
| **Total cajas** | | **3,863** | **19,696** | **1,308** | **10,122** |

Recortes para el clasificador: casco_correcto 19,328 · casco_mal_puesto 1,296 · sin_casco 10,037.

Además se descargan los **32 videos de Zacatecas** (con y sin casco) en `datasets_casco/fuentes/zacatecas_videos/` para probar el modelo en tiempo real.

## Cómo se mapearon las etiquetas

| fuente | original → unificada |
|---|---|
| EdgeVision | BikeWithRider → motocicleta · Helmet → casco_correcto · NoHelmet → sin_casco |
| Motorcycle Helmet OD | With Helmet → casco_correcto · Without Helmet → sin_casco |
| Mapua | rider_full_face, rider_half_face → casco_correcto · rider_helmet_invalid → casco_mal_puesto · rider_no_helmet → sin_casco · rider → descartada |
| LDrago | Helmet-Valid → casco_correcto · Helmet-on-hand, Scarf, Turban → casco_mal_puesto · No-Helmet-on-Rider, No-Helmet-on-passenger → sin_casco |
| Proper Helmet Usage | Proper → casco_correcto · Improper → casco_mal_puesto |
| Helmet Violation | helmet → casco_correcto · cap → casco_mal_puesto · no_helmet → sin_casco · motorcycle → motocicleta · person → descartada |

Las etiquetas que venían como polígonos se convirtieron a la caja que los encierra.

## Limitaciones a tener en cuenta

- **Clases desbalanceadas:** `casco_mal_puesto` tiene 15 veces menos ejemplos que `casco_correcto`. Al entrenar conviene usar pesos por clase o sobremuestreo, y medir con matriz de confusión o F1 por clase, no solo con exactitud.
- **`motocicleta` solo viene etiquetada en EdgeVision y Helmet Violation.** En las demás fuentes aparecen motos sin etiquetar, y el detector las aprende como fondo. Una alternativa es detectar la moto con un YOLO preentrenado en COCO (clase `motorcycle`) y usar este dataset solo para el casco.
- **No todas las cajas encierran lo mismo.** EdgeVision, Motorcycle Helmet OD, Proper Helmet Usage y Helmet Violation encierran la cabeza; LDrago y Mapua encierran a la persona entera. Para los recortes de LDrago y Mapua se tomó el 40 % superior de la caja, donde suele estar la cabeza. Los de LDrago son los menos precisos, porque hay personas lejanas en el tráfico.
- El dataset "Helmet and no helmet rider" (avisentv6) no se incluyó: su autor no publicó ninguna versión descargable en Roboflow.

## Fuentes y licencias

Todos los datasets son **CC BY 4.0**: se pueden usar y compartir citando a sus autores.

- **EdgeVision Dataset**: Gajjar, K., Patel, N., Patel, R., Patel, K. Mendeley Data, doi:[10.17632/j82bnw7gsr.1](https://doi.org/10.17632/j82bnw7gsr.1)
- **Dataset for the Detection of Helmet Use in Motorcycle Drivers** (videos de Zacatecas): Mercado Reyna, J., Luna-García, H., Espino-Salas, C. H., Celaya Padilla, J. M., et al. Mendeley Data, doi:[10.17632/bmy35m25pw.1](https://doi.org/10.17632/bmy35m25pw.1)
- **Motorcycle Helmet Object Detection** (v1): [projectt en Roboflow Universe](https://universe.roboflow.com/projectt/motorcycle-helmet-object-detection)
- **Helmet** (v10): [ldrago en Roboflow Universe](https://universe.roboflow.com/ldrago/helmet-elr7r)
- **Proper Helmet Usage** (v6): [duusuuuu en Roboflow Universe](https://universe.roboflow.com/duusuuuu/proper-helmet-usage)
- **Motorcycle Helmet Detection Dataset** (v8): [Mapua University en Roboflow Universe](https://universe.roboflow.com/mapua-university/motorcycle-helmet-detection-dataset)
- **Helmet Violation Detection** (v1): [imageprocessingworkspace en Roboflow Universe](https://universe.roboflow.com/imageprocessingworkspace/helmet-violation-detetction-ihuwu)
