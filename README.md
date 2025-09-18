# Projeto COP30 — Classificação de Risco Socioambiental

### Contexto
Este projeto foi desenvolvido no âmbito da **COP30 em Belém (2025)**, com o objetivo de apoiar a tomada de decisão sobre projetos na Amazônia.  
Utilizamos algoritmos de **classificação supervisionada** para identificar se projetos apresentam **alto risco socioambiental**, considerando variáveis como:  

- 🌳 Percentual de desmatamento  
- 💧 Nível de poluição em água (mg/L)  
- 🗺️ Distância até áreas protegidas (km)  
- 🌍 Emissão de CO₂ (toneladas)  

---

### 📂 Estrutura do Repositório
```bash
projeto-cop30/
├── data/               # Datasets e saídas de predição
│   ├── dataset_socioambiental_60.csv
│   ├── novos_projetos.csv
│   ├── predicoes_novos_projetos.csv
│   └── top5_alto_risco.csv
├── models/             # Modelos treinados e metadados
│   ├── cop30_svm_pipeline.joblib
│   └── cop30_svm_metadata.json
├── scripts/            # Scripts de execução
│   └── infer.py
├── notebooks/          # Notebooks de modelagem e avaliação
│   └── 01_modelagem_e_avaliacao.ipynb
├── governanca.json     # Regras de threshold e revisão
└── README.md           # Este arquivo


## Como Usar
1. Treino e Avaliação

Abra o notebook principal em notebooks/01_modelagem_e_avaliacao.ipynb e execute as etapas de pré-processamento, modelagem e avaliação.

2. Inferência em Novos Projetos
Execute o script em linha de comando:
python scripts/infer.py \
  --model models/cop30_svm_pipeline.joblib \
  --input data/novos_projetos.csv \
  --output data/predicoes_novos_projetos.csv \
  --threshold 0.5

3. Governança

O arquivo governanca.json define:

- Threshold oficial: 0.5
- Datas de revisão periódica
- Critérios máximos de FP/FN
- Responsáveis pela validação


4. Resultados
Validação Cruzada (25 folds)

1. k-NN → F1 = 0.986 | ROC-AUC = 0.999
2. SVM → F1 = 0.982 | ROC-AUC = 1.000
3. Árvore de Decisão → F1 = 0.947 | ROC-AUC = 0.744

- Modelo selecionado: SVM (maior robustez e estabilidade



5. Predições Operacionais
- Arquivo avaliado: predicoes_novos_projetos.csv
- Limiar oficial (governança): 0.50
- Projetos avaliados (N): 5
- Projetos sinalizados como alto risco: 5 (100.0%)
- Mediana das probabilidades: 0.971
- P90 das probabilidades: 0.993

Exportado top5_alto_risco.csv com os 5 projetos mais críticos para revisão humana.

6. Conclusão

O pipeline de classificação foi treinado, validado, salvo e aplicado em novos projetos simulados.
O modelo SVM alcançou alto desempenho (F1 ≈ 0.99) e classificou todos os projetos de teste como de alto risco socioambiental.

7. Próximos Passos

- Monitorar deriva de dados em produção
- Reavaliar limiar de decisão a cada revisão de governança
- Expandir variáveis socioeconômicas e de circularidade