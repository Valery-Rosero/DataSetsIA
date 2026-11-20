"""
Descarga los datasets de casco/moto para el proyecto.

Uso:
    python descargar_datasets.py --api-key TU_API_KEY_DE_ROBOFLOW
    python descargar_datasets.py --api-key KEY --con-mendeley-grande   # + Helmet Wearing (~2.5 GB)
    python descargar_datasets.py --solo-mendeley                      # sin Roboflow

La API key de Roboflow es gratis: roboflow.com -> Settings -> API Keys.
Todo queda en ./datasets_casco/fuentes/<nombre>/
Solo usa la biblioteca estándar de Python (no hace falta pip install).
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

UA = {"User-Agent": "descargar-datasets-casco/1.0 (script academico)"}
BASE = Path(__file__).resolve().parent / "datasets_casco" / "fuentes"

ROBOFLOW = [
    # (carpeta, workspace, proyecto)
    ("motorcycle_helmet_od", "projectt", "motorcycle-helmet-object-detection"),
    ("helmet_no_helmet_rider", "avisentv6", "helmet-and-no-helmet-rider-detection-ahshg"),
    ("ldrago_helmet", "ldrago", "helmet-elr7r"),
    ("proper_helmet_usage", "duusuuuu", "proper-helmet-usage"),
    ("mapua_helmet", "mapua-university", "motorcycle-helmet-detection-dataset"),
    ("helmet_violation", "imageprocessingworkspace", "helmet-violation-detetction-ihuwu"),
]

MENDELEY = [
    # (carpeta, id, version, opcional)
    ("edgevision", "j82bnw7gsr", 1, False),
    ("zacatecas_videos", "bmy35m25pw", 1, False),
    ("helmet_wearing_27k", "tm72fkfxd5", 2, True),
]


def get_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def descargar(url, destino: Path):
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists() and destino.stat().st_size > 0:
        print(f"    ya existe: {destino.name}")
        return
    tmp = destino.with_suffix(destino.suffix + ".part")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        hecho = 0
        while True:
            bloque = r.read(1 << 20)
            if not bloque:
                break
            f.write(bloque)
            hecho += len(bloque)
            if total:
                print(f"\r    {destino.name}: {hecho/1e6:.0f}/{total/1e6:.0f} MB", end="")
    print()
    tmp.replace(destino)


def descomprimir(zip_path: Path, carpeta: Path):
    print(f"    descomprimiendo {zip_path.name}")
    with zipfile.ZipFile(zip_path) as z:
        raiz = carpeta.resolve()
        for m in z.infolist():
            # evita rutas que escapen de la carpeta destino
            if not (carpeta / m.filename).resolve().is_relative_to(raiz):
                print(f"    [omitido, ruta insegura] {m.filename}")
                continue
            z.extract(m, carpeta)
    zip_path.unlink()


# ---------------------------------------------------------------- Roboflow
def roboflow(api_key, carpeta, ws, proyecto, formato="yolov8"):
    destino = BASE / carpeta
    if (destino / "data.yaml").exists():
        print(f"[roboflow] {carpeta}: ya descargado")
        return True
    print(f"[roboflow] {ws}/{proyecto}")
    q = urllib.parse.urlencode({"api_key": api_key})
    info = get_json(f"https://api.roboflow.com/{ws}/{proyecto}?{q}")
    versiones = info.get("versions") or []
    if not versiones:
        print("    sin versiones publicadas")
        return False
    # la versión más reciente
    ultima = max(versiones, key=lambda v: int(str(v["id"]).rsplit("/", 1)[-1]))
    num = str(ultima["id"]).rsplit("/", 1)[-1]
    print(f"    versión {num}, {ultima.get('images', '?')} imágenes, clases: {list((info.get('project') or {}).get('classes', {}).keys())}")
    url = f"https://api.roboflow.com/{ws}/{proyecto}/{num}/{formato}?{q}"
    for _ in range(60):  # Roboflow puede tardar en generar el export
        r = get_json(url)
        link = (r.get("export") or {}).get("link")
        if link:
            break
        print("    generando export en Roboflow...")
        time.sleep(5)
    else:
        print("    el export no estuvo listo a tiempo, vuelve a correr el script")
        return False
    zip_path = destino / "roboflow.zip"
    descargar(link, zip_path)
    descomprimir(zip_path, destino)
    return True


# ---------------------------------------------------------------- Mendeley
def mendeley(carpeta, ds, ver):
    destino = BASE / carpeta
    if destino.exists() and any(destino.iterdir()):
        print(f"[mendeley] {carpeta}: ya descargado")
        return True
    print(f"[mendeley] {ds} v{ver} -> {destino}")
    # Mendeley arma un zip con todo el dataset ("Download All"): mucho más rápido
    # que bajar miles de archivos uno por uno.
    zip_path = BASE.parent / "zips" / f"{carpeta}.zip"
    descargar(f"https://data.mendeley.com/public-api/zip/{ds}/download/{ver}", zip_path)
    destino.mkdir(parents=True, exist_ok=True)
    descomprimir(zip_path, destino)
    # zips anidados dentro del dataset
    for interno in list(destino.rglob("*.zip")):
        descomprimir(interno, interno.parent)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY"))
    ap.add_argument("--con-mendeley-grande", action="store_true")
    ap.add_argument("--solo-mendeley", action="store_true")
    ap.add_argument("--solo-roboflow", action="store_true")
    a = ap.parse_args()

    fallos = []
    if not a.solo_roboflow:
        for carpeta, ds, ver, opcional in MENDELEY:
            if opcional and not a.con_mendeley_grande:
                continue
            try:
                mendeley(carpeta, ds, ver)
            except Exception as e:
                fallos.append((carpeta, f"https://data.mendeley.com/datasets/{ds}/{ver}", e))

    if not a.solo_mendeley:
        if not a.api_key:
            print("\n[!] Falta --api-key: se omiten los datasets de Roboflow")
        else:
            for carpeta, ws, proyecto in ROBOFLOW:
                try:
                    roboflow(a.api_key, carpeta, ws, proyecto)
                except Exception as e:
                    fallos.append((carpeta, f"https://universe.roboflow.com/{ws}/{proyecto}", e))

    print("\n================ RESUMEN ================")
    for d in sorted(BASE.glob("*")):
        n = sum(1 for _ in d.rglob("*") if _.is_file())
        print(f"  {d.name:28s} {n} archivos")
    if fallos:
        print("\nFallaron (descárgalos a mano y descomprímelos en datasets_casco/fuentes/<carpeta>/):")
        for carpeta, url, e in fallos:
            print(f"  - {carpeta}: {url}\n      error: {e}")
    print("\nHELMET Myanmar (910 videos, opcional): bájalo a mano desde https://osf.io/4pwj8/")
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
