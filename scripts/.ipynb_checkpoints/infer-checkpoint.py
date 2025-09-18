#!/usr/bin/env python3
# infer.py — CLI de inferência para o pipeline COP30

import argparse
import sys
import numpy as np
import pandas as pd
import joblib

def main():
    parser = argparse.ArgumentParser(
        description="Inferência com pipeline salvo (COP30)."
    )
    parser.add_argument("--model", required=True, help="Caminho do .joblib salvo")
    parser.add_argument("--input", required=True, help="CSV de entrada (novos dados)")
    parser.add_argument("--output", default="predicoes_saida.csv", help="CSV de saída")
    parser.add_argument("--threshold", type=float, default=0.5, help="Limiar p/ classe 1 (default=0.5)")
    args = parser.parse_args()

    # 1) Carrega pipeline
    try:
        pipe = joblib.load(args.model)
    except Exception as e:
        print(f"[ERRO] Falha ao carregar modelo: {e}", file=sys.stderr)
        sys.exit(1)

    # 2) Lê CSV novo
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"[ERRO] Falha ao ler CSV de entrada: {e}", file=sys.stderr)
        sys.exit(1)

    # 3) Verifica colunas esperadas (usamos feature_names_in_ do preprocessor)
    try:
        expected = list(pipe.named_steps["prep"].feature_names_in_)
    except Exception:
        expected = df.columns.tolist()  # fallback simples

    missing = set(expected) - set(df.columns)
    if missing:
        print(f"[ERRO] Colunas faltantes no CSV de entrada: {missing}", file=sys.stderr)
        sys.exit(1)

    # 4) Probabilidades da classe positiva
    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        prob_pos = pipe.predict_proba(df)[:, 1]
    elif hasattr(pipe.named_steps["clf"], "decision_function"):
        scores = pipe.decision_function(df)
        prob_pos = 1 / (1 + np.exp(-scores))  # mapeamento logístico simples (fallback)
    else:
        preds = pipe.predict(df)
        prob_pos = preds.astype(float)

    # 5) Predições padrão e por limiar
    pred_default = (prob_pos >= 0.5).astype(int)
    pred_thresh  = (prob_pos >= args.threshold).astype(int)

    # 6) Salva resultado
    out = df.copy()
    out["prob_pos"] = prob_pos
    out["pred_padrao_0_1"] = pred_default
    out["pred_thresh_0_1"] = pred_thresh
    out.to_csv(args.output, index=False)

    print(f"[OK] Predições salvas em: {args.output}")

if __name__ == "__main__":
    main()
