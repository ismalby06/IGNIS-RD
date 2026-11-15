"""Datos SIMULADOS con forma de FIRMS, para probar el pipeline sin MAP_KEY.

No representan incendios reales en República Dominicana. Imitan lo que hace
el problema interesante:
- un mismo conjunto de fuegos "verdaderos" visto por dos sensores;
- MODIS (1 km) solo ve fuegos más grandes y da ~1 píxel por fuego;
- VIIRS (375 m, desde 2012) ve fuegos pequeños y parte los grandes en varios píxeles;
- nubes, temporada seca, un año extremo y una fuente persistente (vertedero/industria).
Así el salto de 2012 aparece igual que en los datos reales.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.clean import inside_dr

# (lon, lat, dispersión km, peso) de zonas con quemas en la simulación
CLUSTERS = [
    (-70.70, 18.85, 14, 3.0),   # Cordillera Central
    (-70.95, 19.02, 10, 2.0),
    (-71.45, 18.12, 10, 2.0),   # Sierra de Bahoruco
    (-68.95, 18.58, 16, 3.0),   # Este, zona cañera
    (-69.35, 18.68, 12, 2.0),
    (-71.35, 19.62, 14, 2.0),   # Línea Noroeste
    (-70.45, 19.28, 18, 2.0),   # Cibao
    (-71.20, 18.82, 10, 1.5),   # Valle de San Juan
    (-70.70, 18.45, 12, 1.5),   # Azua
]
PERSISTENT_SITE = (-69.90, 18.52)

SEASON = np.array([0.9, 1.3, 2.0, 1.8, 0.9, 0.5, 0.6, 0.6, 0.4, 0.3, 0.3, 0.5])
CLEAR_SKY = np.array([.66, .68, .65, .58, .45, .42, .48, .45, .40, .42, .50, .60])

VIIRS_START = pd.Timestamp("2012-01-20")
AQUA_START = pd.Timestamp("2002-07-04")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _sample_points(rng, n):
    w = np.array([c[3] for c in CLUSTERS]); w = w / w.sum()
    idx = rng.choice(len(CLUSTERS), size=n * 2, p=w)
    lon0 = np.array([CLUSTERS[i][0] for i in idx]); lat0 = np.array([CLUSTERS[i][1] for i in idx])
    sd = np.array([CLUSTERS[i][2] for i in idx])
    lat = lat0 + rng.normal(0, 1, n * 2) * sd / 110.6
    lon = lon0 + rng.normal(0, 1, n * 2) * sd / (111.3 * np.cos(np.radians(lat0)))
    ok = inside_dr(lat, lon)
    return lat[ok][:n], lon[ok][:n]


def simulate(start_year: int = 2001, end_year: int = 2025, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    years = np.arange(start_year, end_year + 1)
    year_factor = rng.lognormal(0, 0.28, len(years)) * (1 - 0.008 * (years - start_year))
    extreme = rng.choice(years[years >= 2013])            # un año extremo dentro del solapamiento
    year_factor[years == extreme] *= 2.1
    pre = rng.choice(years[years <= 2009])                # y otro antes de VIIRS
    year_factor[years == pre] *= 1.8

    modis_rows, viirs_rows = [], []
    for y, f in zip(years, year_factor):
        for m in range(1, 13):
            lam = 120 * f * SEASON[m - 1]
            n_fires = rng.poisson(lam)
            if n_fires == 0:
                continue
            days_in = pd.Period(f"{y}-{m:02d}").days_in_month
            lat, lon = _sample_points(rng, n_fires)
            n_fires = len(lat)
            frp = rng.lognormal(1.2, 1.1, n_fires)            # MW, mediana ~3.3
            dur = rng.geometric(0.55, n_fires)
            start_day = rng.integers(1, days_in + 1, n_fires)
            for i in range(n_fires):
                for d in range(dur[i]):
                    day = pd.Timestamp(y, m, 1) + pd.Timedelta(days=int(start_day[i] - 1 + d))
                    clear = rng.random() < CLEAR_SKY[day.month - 1]
                    if not clear:
                        continue
                    f_today = frp[i] * rng.lognormal(0, 0.3)
                    # MODIS: necesita fuegos más grandes; antes de Aqua solo Terra
                    p_m = _sigmoid((np.log(f_today) - np.log(9)) / 0.55)
                    if day < AQUA_START:
                        p_m *= 0.62
                    if rng.random() < p_m:
                        npx = 1 + rng.poisson(f_today / 70)
                        for _ in range(npx):
                            modis_rows.append((
                                lat[i] + rng.normal(0, .004), lon[i] + rng.normal(0, .004),
                                day, rng.choice([1520, 1800, 300, 230]),
                                "Aqua" if (day >= AQUA_START and rng.random() < .5) else "Terra",
                                int(np.clip(rng.normal(55 + 10 * np.log(f_today), 15), 0, 100)),
                                round(f_today / npx * rng.lognormal(0, .2), 1)))
                    # VIIRS: ve fuegos pequeños y divide los grandes en varios píxeles
                    if day >= VIIRS_START:
                        p_v = _sigmoid((np.log(f_today) - np.log(1.3)) / 0.55)
                        if rng.random() < p_v:
                            npx = 1 + rng.poisson(f_today / 6)
                            for _ in range(npx):
                                viirs_rows.append((
                                    lat[i] + rng.normal(0, .0028), lon[i] + rng.normal(0, .0028),
                                    day, rng.choice([1745, 1800, 545, 610]), "N",
                                    rng.choice(["l", "n", "h"], p=[.08, .70, .22]),
                                    round(f_today / npx * rng.lognormal(0, .25), 1)))

    # Fuente persistente: aparece en ~35 % de los meses, todo el año
    for y in years:
        for m in range(1, 13):
            if rng.random() < 0.35:
                day = pd.Timestamp(y, m, int(rng.integers(1, 28)))
                modis_rows.append((PERSISTENT_SITE[1], PERSISTENT_SITE[0], day, 1520,
                                   "Terra", 70, 12.0))
                if day >= VIIRS_START:
                    for _ in range(2):
                        viirs_rows.append((PERSISTENT_SITE[1] + rng.normal(0, .001),
                                           PERSISTENT_SITE[0] + rng.normal(0, .001),
                                           day, 1745, "N", "n", 4.0))

    cols = ["latitude", "longitude", "acq_date", "acq_time", "satellite", "confidence", "frp"]
    mod = pd.DataFrame(modis_rows, columns=cols)
    mod["instrument"] = "MODIS"; mod["daynight"] = np.where(mod["acq_time"] > 1000, "D", "N")
    vii = pd.DataFrame(viirs_rows, columns=cols)
    vii["instrument"] = "VIIRS"; vii["daynight"] = np.where(vii["acq_time"] > 1000, "D", "N")
    for df in (mod, vii):
        df["acq_date"] = df["acq_date"].dt.strftime("%Y-%m-%d")
        df["type"] = 0
    return mod, vii
