# READ ME
----
**API de Mapas**
----
# **Información sobre el proyecto/Como me conecto**

Usa python 3.12.11 y se utilizo conda para gestion de librerias.

Para mas informacion sobre librerias mirar <code>requirements.txt.

A fecha de 25/09/2025 la API esta activa con  [render](https://reto1-bridge-data.onrender.com). Puede tardar un rato en ponerse en marcha. 
  
/docs al url veras la documentación. 
De todas formas esta en [este enlace](https://reto1-bridge-data.onrender.com/docs).

Seguramente los mapas no se acaben implementando en el proyecto por lo que hasta que no se confirme implementación de estos no se generaran mapas mas mapas o mas complejos. 
Seguramente el url acabe cambiando, ya que se esta metiendo en un <code> docker compose.

Tambien tended cuidado:
Cada vez que alguien sube a main la API se reinicia. Esto puede romper la API y gastar los  500 min gratis que tenemos. Asegurarse de que las funciones funciones en local antes de commit en main. No se si se reinicia por cada cambio solo por los cambios que le a la API.


Se deberia de  borrar el .env y poner el .gitignore pero por facilidad de uso en desarrollo es visible.

<code>DB_URL=mongodb+srv://<dbuser>:<dbpassword>@cluster0.qdcfbed.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0
<code>DB_NAME=Db name

DB_NAME no se usa. Se que no deberia de la url de la base de datos aqui pero quiero assegurarme que el que lo lea le funcione.



## Run en local
decir  las palabras magicas:

<code>python main.py

DENTRO DE LA CARPETA  

o ...

<code>python api_mapas/main.py

en directorio principal
y revisar el localhost puerto 9000 /docs para ver la documentacion. Si da error autoriza el localhost9000. 

## Como funciona por detras/cosas para tener en cuenta en desarrollo de mas mapas 

Al iniciarse, se conecta y lee la base de datos (código basado en el ejemplo de MongoDB, aunque más actualizado y no del todo optimizado).

Esta comfigurado para que la base de datos este separado por 3 colecciones y cada función de lectura lee
Existen funciones específicas para cada tipo de dato:

* **leergas:** lee la base de datos de gasolineras y la transforma en un DataFrame limpio.

* **leerpeaje:** lee la base de datos de peajes y la transforma en un DataFrame limpio.

* **leerev:** lee la base de datos de cargadores eléctricos y la transforma en un DataFrame limpio


*gasolina*
| #   | Columna                | Tipo          |
|-----|------------------------|---------------|
| 0  | _id                     | object        |
| 1  | idTicket                | object        |
| 2  | idEmpresa               | object        |
| 3  | empresaNombre           | object        |
| 4  | idUsuario               | object        |
| 5  | fechaHora               | datetime64[ns]|
| 6  | metodoPago              | object        |
| 7  | baseImponible           | float64       |
| 8  | iva                     | float64       |
| 9  | total                   | float64       |
| 10 | moneda                  | object        |
| 11 | tipoDocumento           | object        |
| 12 | estacion.id             | object        |
| 13 | estacion.nombre         | object        |
| 14 | estacion.provincia      | object        |
| 15 | estacion.municipio      | object        |
| 16 | estacion.direccion      | object        |
| 17 | estacion.lat            | float64       |
| 18 | estacion.lon            | float64       |
| 19 | estacion.grupo          | object        |
| 20 | estacion.nifEmpresa     | object        |
| 21 | lineas.producto         | object        |
| 22 | lineas.litros           | float64       |
| 23 | lineas.precioUnitario   | float64       |
| 24 | lineas.importe          | float64       |



*Vehiculos electricos*

| #   | Columna                           | Tipo           |
|-----|----------------------------------|----------------|
| 0   | _id                               | object         |
| 1   | idTicket                          | object         |
| 2   | idEmpresa                         | object         |
| 3   | empresaNombre                     | object         |
| 4   | idUsuario                         | object         |
| 5   | fechaHora                         | datetime64[ns] |
| 6   | metodoPago                        | object         |
| 7   | baseImponible                     | float64        |
| 8   | iee                               | float64        |
| 9   | iva                               | float64        |
| 10  | total                             | float64        |
| 11  | moneda                            | object         |
| 12  | tipoDocumento                     | object         |
| 13  | estacion.id                       | object         |
| 14  | estacion.provincia                | object         |
| 15  | estacion.municipio                | object         |
| 16  | estacion.direccion                | object         |
| 17  | estacion.lat                      | float64        |
| 18  | estacion.lon                      | float64        |
| 19  | estacion.empresa                  | object         |
| 20  | estacion.nifEmpresa               | object         |
| 21  | estacion.potenciaMaxKW            | float64        |
| 22  | estacion.tarifa                   | object         |
| 23  | lineas.producto                   | object         |
| 24  | lineas.kwh                        | float64        |
| 25  | lineas.precioUnitarioSinImpuestos | float64        |
| 26  | lineas.precioUnitario             | float64        |
| 27  | lineas.importe                    | float64        |



*Peaje*
| #   | Columna              | Tipo   |
|-----|----------------------|--------|
| 0   | _id                  | object |
| 1   | tipoDocumento        | object |
| 2   | concesionaria        | object |
| 3   | autopista            | object |
| 4   | fechaHora            | object |
| 5   | categoriaVehiculo    | object |
| 6   | importe              | float  |
| 7   | ivaIncluido          | object |
| 8   | formaPago            | object |
| 9   | referencia           | object |
| 10  | idEmpresa            | object |
| 11  | empresaNombre        | object |
| 12  | idUsuario            | object |
| 13  | provincia            | object |
| 14  | localizacion.tramo   | object |
| 15  | localizacion.entrada | object |
| 16  | localizacion.salida  | object |

## Funciones de mapas

* **mapakwh:** 

*por detras*

Llama a una funcion que llama a la base de datos electico pasa a dataframe y se filtra con pandas.
Usa un bucle for.
Actualmente no el endpoint no filtra info especifica, esperare a la confirmacion de la implementación de mapas.

*Por delante*

Devuelve para todos los medios
Formato:<code> [[lat, lon, precio_medio], ...] 
     
* **mapagas:** 

*por detras*

Llama a una funcion que llama a la base de datos de vehiculos de combustible pasa a dataframe y se filtra con pandas. Estan todos juntos.

*Por delante*

Devuelve para todos los medios
Formato: <code>[[lat, lon, precio_medio], ...]


* **mapagas_concreto:** 

*por detrás*

Mismo que mapa gas pero puedes filtrar el combustible


Llama a una funcion que llama a la base de datos gas pasa a dataframe y se filtra con pandas
Actualmente la base de datos tiene 
<code>
['Gasóleo A', 'Gasolina 95 E5', 'Gasolina 98 E5', 'Gasóleo Premium']

Si se quieren filtrar varios (ej. Gasolina 98 E5 y Gasóleo Premium), se pasa como query param:



<code>
/mapagas_concreto?combustible=Gasolina%2098%20E5&combustible=Gas%C3%B3leo%20Premium

Esta funcion es mejor revisar en <code>/docs
*Por delante*

Devuelve para todos los medios
Formato: <code>[[lat, lon, precio_medio], ...]

Ahora mismo no hay forma de diferenciar cual es cual.
