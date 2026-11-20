"""
Une los datasets descargados en datasets_casco/fuentes/ en un solo dataset YOLO
con 4 clases, y genera recortes por clase para entrenar un clasificador (CNN).

    0 motocicleta
    1 casco_correcto
    2 casco_mal_puesto
    3 sin_casco

Uso:
    python unificar_etiquetas.py              # detector + recortes
    python unificar_etiquetas.py --sin-recortes

Salida (en datasets_casco/unificado/):
    images/{train,val,test}/, labels/{train,val,test}/, data.yaml
    recortes/{train,val,test}/<clase>/*.jpg
    resumen.txt
"""
import argparse
import hashlib
import os
import re
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

RAIZ = Path(__file__).resolve().parent / "datasets_casco"
FUENTES = RAIZ / "fuentes"
SALIDA = RAIZ / "unificado"

CLASES = ["motocicleta", "casco_correcto", "casco_mal_puesto", "sin_casco"]
MOTO, OK, MAL, SIN = range(4)

# Mapeo: nombre de clase original -> clase unificada (None = descartar)
# "caja" indica qué encierra la caja original: "cabeza" o "persona" (jinete entero).
# Para recortes de cajas tipo "persona" se toma solo la parte superior (la cabeza).
FUENTES_YOLO = {
    "motorcycle_helmet_od": {
        "caja": "cabeza",
        "mapa": {"With Helmet": OK, "Without Helmet": SIN},
    },
    "ldrago_helmet": {
        "caja": "persona",
        "mapa": {
            "Helmet-Valid": OK,
            "Helmet-on-hand": MAL,        # casco en la mano en vez de puesto
            "Scarf": MAL,                 # pañuelo en vez de casco
            "Turban": MAL,                # turbante en vez de casco
            "No-Helmet-on-Rider": SIN,
            "No-Helmet-on-passenger": SIN,
            "shirt": None,
        },
        "caja_completa": {"Helmet-on-hand"},  # esta caja no es una persona
    },
    "mapua_helmet": {
        "caja": "persona",
        "mapa": {
            "rider_full_face": OK,
            "rider_half_face": OK,
            "rider_helmet_invalid": MAL,
            "rider_no_helmet": SIN,
            "rider": None,
        },
    },
    "proper_helmet_usage": {
        "caja": "cabeza",
        "mapa": {"Proper": OK, "Improper": MAL},
    },
    "helmet_violation": {
        "caja": "cabeza",
        "mapa": {
            "helmet": OK,
            "cap": MAL,                   # gorra en vez de casco
            "no_helmet": SIN,
            "motorcycle": MOTO,
            "person": None,
        },
    },
}

FUENTES_VOC = {
    "edgevision": {
        "caja": "cabeza",
        "mapa": {"BikeWithRider": MOTO, "Helmet": OK, "NoHelmet": SIN},
    },
}

SPLITS = {"train": "train", "valid": "val", "val": "val", "test": "test"}
EXT_IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FRACCION_CABEZA = 0.40  # parte superior de una caja "persona" que se usa como recorte


def leer_nombres(data_yaml: Path):
    txt = data_yaml.read_text(encoding="utf-8")
    m = re.search(r"^names:\s*(\[.*?\])", txt, re.M | re.S)
    if m:
        return [s.strip().strip("'\"") for s in m.group(1)[1:-1].split(",")]
    # formato en lista con guiones
    m = re.search(r"^names:\s*\n((?:\s+-.*\n?)+)", txt, re.M)
    return [l.strip()[1:].strip().strip("'\"") for l in m.group(1).splitlines()]


def split_por_hash(nombre):
    h = int(hashlib.md5(nombre.encode()).hexdigest(), 16) % 100
    return "train" if h < 80 else ("val" if h < 90 else "test")


def enlazar(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)  # no ocupa espacio extra si está en el mismo disco
    except OSError:
        shutil.copy2(src, dst)


def a_caja(valores):
    """YOLO: 4 números = caja; más = polígono -> caja envolvente."""
    if len(valores) == 4:
        return valores
    xs, ys = valores[0::2], valores[1::2]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]


class Unificador:
    def __init__(self, recortes: bool):
        self.recortes = recortes
        self.conteo = defaultdict(Counter)       # fuente -> clase unificada -> n
        self.descartadas = defaultdict(Counter)  # fuente -> clase original -> n
        self.imagenes = Counter()
        self.recortes_n = Counter()

    def guardar(self, fuente, split, img: Path, cajas, tipo_caja):
        """cajas: lista de (clase, cx, cy, w, h, caja_completa) normalizadas."""
        nombre = f"{fuente}__{img.stem}"
        enlazar(img, SALIDA / "images" / split / f"{nombre}{img.suffix.lower()}")
        lineas = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h, _ in cajas]
        (SALIDA / "labels" / split).mkdir(parents=True, exist_ok=True)
        (SALIDA / "labels" / split / f"{nombre}.txt").write_text("\n".join(lineas) + ("\n" if lineas else ""))
        self.imagenes[fuente] += 1
        for c, *_ in cajas:
            self.conteo[fuente][CLASES[c]] += 1
        if self.recortes:
            self.recortar(fuente, split, img, nombre, cajas, tipo_caja)

    def recortar(self, fuente, split, img, nombre, cajas, tipo_caja):
        objetivo = [b for b in cajas if b[0] != MOTO]
        if not objetivo:
            return
        try:
            im = Image.open(img).convert("RGB")
        except Exception as e:
            print(f"    [no se pudo abrir] {img.name}: {e}")
            return
        W, H = im.size
        for i, (c, cx, cy, w, h, completa) in enumerate(objetivo):
            x0, x1 = (cx - w / 2) * W, (cx + w / 2) * W
            y0, y1 = (cy - h / 2) * H, (cy + h / 2) * H
            if tipo_caja == "persona" and not completa:
                y1 = y0 + (y1 - y0) * FRACCION_CABEZA
                # la cabeza suele ser más angosta que el cuerpo
                m = (x1 - x0) * 0.15
                x0, x1 = x0 + m, x1 - m
            # margen del 10 %
            mx, my = (x1 - x0) * 0.1, (y1 - y0) * 0.1
            caja = (max(0, x0 - mx), max(0, y0 - my), min(W, x1 + mx), min(H, y1 + my))
            if caja[2] - caja[0] < 12 or caja[3] - caja[1] < 12:
                continue  # demasiado pequeño para ser útil
            d = SALIDA / "recortes" / split / CLASES[c]
            d.mkdir(parents=True, exist_ok=True)
            im.crop(caja).save(d / f"{nombre}_{i}.jpg", quality=92)
            self.recortes_n[CLASES[c]] += 1

    # ------------------------------------------------------------ YOLO
    def fuente_yolo(self, fuente, cfg):
        base = FUENTES / fuente
        if not (base / "data.yaml").exists():
            print(f"[omitido] {fuente}: no está descargado")
            return
        nombres = leer_nombres(base / "data.yaml")
        print(f"[yolo] {fuente}: {nombres}")
        mapa = cfg["mapa"]
        completas = cfg.get("caja_completa", set())
        for carpeta_split in sorted(base.iterdir()):
            split = SPLITS.get(carpeta_split.name)
            if not split or not (carpeta_split / "images").is_dir():
                continue
            for img in sorted((carpeta_split / "images").iterdir()):
                if img.suffix.lower() not in EXT_IMG:
                    continue
                lbl = carpeta_split / "labels" / f"{img.stem}.txt"
                cajas = []
                if lbl.exists():
                    for linea in lbl.read_text().splitlines():
                        v = linea.split()
                        if len(v) < 5:
                            continue
                        idx = int(v[0])
                        original = nombres[idx] if idx < len(nombres) else str(idx)
                        nueva = mapa.get(original)
                        if nueva is None:
                            self.descartadas[fuente][original] += 1
                            continue
                        cx, cy, w, h = a_caja([float(x) for x in v[1:]])
                        if w <= 0 or h <= 0:
                            continue
                        cajas.append((nueva, cx, cy, w, h, original in completas))
                self.guardar(fuente, split, img, cajas, cfg["caja"])

    # ------------------------------------------------------------ VOC
    def fuente_voc(self, fuente, cfg):
        base = FUENTES / fuente
        if not base.exists():
            print(f"[omitido] {fuente}: no está descargado")
            return
        xmls = list(base.rglob("*.xml"))
        imgs = {p.name.lower(): p for p in base.rglob("*") if p.suffix.lower() in EXT_IMG}
        print(f"[voc]  {fuente}: {len(xmls)} xml, {len(imgs)} imágenes")
        sin_imagen = 0
        for x in sorted(xmls):
            raiz = ET.parse(x).getroot()
            archivo = (raiz.findtext("filename") or f"{x.stem}.jpg").strip()
            img = imgs.get(Path(archivo).name.lower()) or imgs.get(f"{x.stem}.jpg".lower())
            if img is None:
                sin_imagen += 1
                continue
            W = float(raiz.findtext("size/width") or 0)
            H = float(raiz.findtext("size/height") or 0)
            if not W or not H:
                with Image.open(img) as im:
                    W, H = im.size
            cajas = []
            for obj in raiz.iter("object"):
                original = (obj.findtext("name") or "").strip()
                nueva = cfg["mapa"].get(original)
                if nueva is None:
                    self.descartadas[fuente][original] += 1
                    continue
                bb = obj.find("bndbox")
                x0, y0, x1, y1 = (float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
                x0, x1 = max(0, x0), min(W, x1)
                y0, y1 = max(0, y0), min(H, y1)
                if x1 <= x0 or y1 <= y0:
                    continue
                cajas.append((nueva, (x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H, False))
            self.guardar(fuente, split_por_hash(img.name), img, cajas, cfg["caja"])
        if sin_imagen:
            print(f"    {sin_imagen} xml sin imagen correspondiente")

    def resumen(self):
        lineas = ["RESUMEN DE UNIFICACIÓN", "=" * 60, "", "Cajas por fuente (detector):"]
        total = Counter()
        for f in sorted(self.conteo):
            c = self.conteo[f]
            total.update(c)
            detalle = ", ".join(f"{k}={c[k]}" for k in CLASES if c[k])
            lineas.append(f"  {f:24s} {self.imagenes[f]:5d} imgs | {detalle}")
        lineas += ["", "TOTAL cajas: " + ", ".join(f"{k}={total[k]}" for k in CLASES),
                   f"TOTAL imágenes: {sum(self.imagenes.values())}", ""]
        for split in ("train", "val", "test"):
            n = len(list((SALIDA / "images" / split).glob("*"))) if (SALIDA / "images" / split).exists() else 0
            lineas.append(f"  {split:5s}: {n} imágenes")
        if self.recortes:
            lineas += ["", "Recortes para el clasificador:"]
            lineas += [f"  {k:18s} {self.recortes_n[k]}" for k in CLASES[1:]]
        lineas += ["", "Etiquetas descartadas:"]
        for f in sorted(self.descartadas):
            lineas.append(f"  {f}: " + ", ".join(f"{k}={v}" for k, v in self.descartadas[f].items()))
        texto = "\n".join(lineas)
        print("\n" + texto)
        (SALIDA / "resumen.txt").write_text(texto, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sin-recortes", action="store_true")
    a = ap.parse_args()

    if SALIDA.exists():
        print(f"Borrando salida anterior: {SALIDA}")
        shutil.rmtree(SALIDA)
    u = Unificador(recortes=not a.sin_recortes)
    for f, cfg in FUENTES_YOLO.items():
        u.fuente_yolo(f, cfg)
    for f, cfg in FUENTES_VOC.items():
        u.fuente_voc(f, cfg)

    # sin "path": Ultralytics usa la carpeta donde está este data.yaml
    (SALIDA / "data.yaml").write_text(
        "train: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(CLASES)}\nnames: {CLASES}\n",
        encoding="utf-8",
    )
    u.resumen()


if __name__ == "__main__":
    main()
