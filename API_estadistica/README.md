# READ ME
Kernel: Python 3.12.10

Para launchear: uvicorn main:app --reload en terminal
Acceso local: http://127.0.0.1:8000 

Para ver endpoints:

http://127.0.0.1:8000/kpis/combustible
http://127.0.0.1:8000/kpis/ev
http://127.0.0.1:8000/kpis/peaje

Para ver salud y colecciones:

http://127.0.0.1:8000/health
http://127.0.0.1:8000/debug/peek

Todos los combustibles (sin filtros):
http://127.0.0.1:8000/kpis/combustible

Combustible (rango de fechas):
http://127.0.0.1:8000/kpis/combustible?start_date=2025-01-01&end_date=2025-09-24

Combustible cambiar parametros (empresa + usuario + fechas):

http://127.0.0.1:8000/kpis/combustible?empresa=Transporte_01%20S.L.&idUsuario=EMP001-U1&start_date=2025-01-01&end_date=2025-09-24


Eléctricos:
http://127.0.0.1:8000/kpis/ev

Peajes (rango de fechas):
http://127.0.0.1:8000/kpis/peaje?start_date=2025-01-01&end_date=2025-09-24

Documentación interactiva (Swagger):
http://127.0.0.1:8000/docs

Health & debug:

http://127.0.0.1:8000/health

http://127.0.0.1:8000/debug/peek?n=2



