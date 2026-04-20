# Kurumsal Karbon Stres Testi (CBAM) - ARIMAX + MLP Hibrit Pipeline

Bu repo, **EREGL.IS** için CBAM şok etkisini modellemek amacıyla istenen yeni mimariye göre sıfırdan kurulmuştur.

## Hedef

Son 5 yıllık günlük veride şu yapıyı kurar:

- **Y:** `EREGL.IS` (Erdemir kapanış fiyatı)
- **X1:** `KRBN` (global carbon ETF proxy)
- **X2:** `TIO=F` (demir cevheri vadeli)

Model:

1. **ARIMAX** (Y ~ X2)
2. **MLP** ile ARIMAX residual tahmini (X1 + residual lag1)
3. **Hibrit tahmin** = ARIMAX tahmini + MLP residual tahmini

## Notebook Yapısı

Notebook/Colab akışı 6 modül içerir:

1. Veri çekme, ffill/bfill, MinMax, ADF/diff
2. ARIMAX eğitimi ve test tahmini
3. Eğitim residual çıkarımı
4. TensorFlow/Keras MLP eğitimi
5. Benchmark modelleri (Baseline + XGBoost), hibrit birleşim ve RMSE/MAE/MAPE karşılaştırması
6. Akademik görselleştirme (ayrı figürler: RMSE bar chart, test tahmin çizgileri, residual dağılımı, parity plot, residual zaman serisi)

## Kurulum

```bash
pip install -r requirements.txt
```

## Çalıştırma

```bash
python main.py
```

Notebook sürümü:

- `notebooks/cbam_hybrid_pipeline.ipynb`
