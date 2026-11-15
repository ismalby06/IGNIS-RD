#!/usr/bin/env bash
# Script de inicio rápido para FuegoRD / IGNIS-RD (Streamlit)
# NASA Space Apps Challenge 2026

set -e
cd "$(dirname "$0")"

echo "=========================================================="
echo "🔥 FuegoRD · IGNIS-RD: Calendario Armonizado de Quemas"
echo "NASA Space Apps Challenge 2026 · República Dominicana 🇩🇴"
echo "=========================================================="

if [ ! -d ".venv" ]; then
    echo "📦 Creando entorno virtual (.venv)..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "⬇️ Instalando dependencias de requirements.txt..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    source .venv/bin/activate
fi

echo "🚀 Iniciando servidor Streamlit en http://localhost:8501..."
streamlit run app.py --server.port 8501
