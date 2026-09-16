# Hasil - Prediksi Kartu Pokemon TCG Ultra Rare

Data: 14,497 kartu | ultra rare 2,179 (15.0%)

## Perbandingan Model

|                     |   accuracy |   precision |   recall |    f1 |   roc_auc |
|:--------------------|-----------:|------------:|---------:|------:|----------:|
| Logistic Regression |      0.957 |       0.888 |    0.815 | 0.85  |     0.969 |
| Stacking            |      0.955 |       0.89  |    0.799 | 0.842 |     0.977 |
| Voting (soft)       |      0.951 |       0.89  |    0.769 | 0.825 |     0.972 |
| Random Forest       |      0.948 |       0.877 |    0.765 | 0.817 |     0.97  |
| K-Nearest Neighbors |      0.931 |       0.818 |    0.693 | 0.75  |     0.921 |
| RF tuned + SMOTE    |      0.958 |       0.87  |    0.848 | 0.859 |     0.98  |

## Best Params

```
{'clf__class_weight': 'balanced', 'clf__max_depth': 25, 'clf__max_features': 0.5, 'clf__min_samples_split': 5, 'clf__n_estimators': 300}
```

## Cek Kebocoran (AUC satu kolom)

|          |     0 |
|:---------|------:|
| subtypes | 0.902 |
| rules    | 0.887 |
| hp       | 0.882 |
| artist   | 0.865 |
| set      | 0.754 |

Kolom dibuang: ['subtypes', 'rules']

## Top 10 Fitur

|                            |      0 |
|:---------------------------|-------:|
| hp                         | 0.3416 |
| punya_flavor               | 0.1278 |
| set_Shiny Vault            | 0.0951 |
| tahun                      | 0.048  |
| max_damage                 | 0.0425 |
| artist_5ban Graphics       | 0.0264 |
| artist_Ryo Ueda            | 0.0249 |
| set_SWSH Black Star Promos | 0.02   |
| artist_infrequent_sklearn  | 0.0147 |
| set_SM Black Star Promos   | 0.0131 |
