# DESAFIO 

En la fase final del Bootcamp se nos encarga un desafio que debemos realizar junto con las demas clases del bootcamp; ciberseguridad, marketing,fullstack. 

## Problema especifico:

Dentro del desafio se hace necesario el reconocimiento de los tickets, ello se puede hacer mediante un programa OCR, no obstante, en los ultimos años han surgido soluciones que nos permiten realizar este mismo trabajo de forma mas rapida. dado el caracter de sprint del desafio se hace necesario el uso de herramientas de IA. 

En concreto, para este punto se utilizará el llamcloud.IA. [LlamaCloud](https://cloud.llamaindex.a)

## Datos que vamos a encontrar en todas las facturas:

Segu la Web de la hacienda tributaria son obligatorias las siguientes:
"Las facturas simplificadas deben contener al menos los siguientes datos:

- Número y, en su caso, serie. La numeración será correlativa.

- Fecha de expedición.

- Fecha en que se hayan efectuado las operaciones o se haya recibido el pago anticipado, siempre que sea distinta a la de expedición de la factura.

- Número de identificación fiscal, así como el nombre y apellidos, razón o denominación social del obligado a su expedición.

-  identificación del tipo de bienes entregados o de servicios prestados.

- Tipo impositivo aplicado y, opcionalmente también la expresión «IVA incluido».

- Contraprestación total.

-  el caso de facturas rectificativas, la referencia expresa e inequívoca de la factura rectificada y de las especificaciones que se modifican.

- Las menciones recogidas en las facturas ordinarias relativas a la aplicación de regímenes especiales y a determinadas operaciones (exentas, con inversión del sujeto pasivo, etc.)."

No obstante, no todo lo que aqui aparece se debe de recoger con el OCR.

## Recogeremos en la de gasolina: 
FACTURA GASOLINERA

Nombre de la empresa

nº factura

NIF/CIF


Dirección de la estación de servicio(
provincia
municipio
latitud  
longitud
código postal


productos comprados -> nombre del producto, precio por litro, litros, precio total.

(lo mismo para eléctrico)

IVA 

total

forma de pago

fecha

hora

## En la de peaje: 

nº de factura

nombre de concesionaria

autopista

localización-> tramo, entrada, salida.

fecha 
hora 
importe 
iva
forma de pago

categoría de 


# Otros parametros: 
llamcloud llama 

***

En este caso asumiremos que las facturas vienen con la dirección de establecimiento.
por otro lado tambien tendremos que recoger las facturas que vengan de los peajes a pesar de ser ambos tickets el contenido puede variar enormemente, 
incluso dentro de los tickets de peaje y los de gasolina la varianza entre los tickets es enorme, a pesar de que la legislacion 
obliga a un contenido minimo muchos no lo soportan como debieran.



API-key: llx-80200pQgQeS1QWleqJHsDUjIBFf4QCbQIOyqyPJbo6KnGNRq
nombre: LEER FACTURAS 1
id: 882591e8-1c77-4ae7-a494-77ce1a30d4ce

nombre: Facturas Peaje
id: a0a10b83-97b4-4024-b9ff-e09b2f0a64e6


# Tipos de archivos que soporta 
LlamaExtract supports the following file formats:

Documents: PDF (.pdf), Word (.docx)
Text files: Plain text (.txt), CSV (.csv), JSON (.json), HTML (.html, .htm), Markdown (.md)
Images: PNG (.png), JPEG (.jpg, .jpeg). 

el codigo recogido en este repositorio, recoge la forma que tendria un fastapi con la información, en el punto de entrada del archivo 
se puede meter cualquier formato que se quiera, la aplicacion lo transformara el bytes y despues lo enviará a llamacloud para procesar. 











