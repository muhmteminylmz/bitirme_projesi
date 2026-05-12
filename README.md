# Kurumsal Karbon Stres Testi (CBAM) - ARIMAX + MLP Hibrit Pipeline

Bu repo, **EREGL.IS** için CBAM şok etkisini modellemek amacıyla istenen yeni mimariye göre sıfırdan kurulmuştur.

## Hedef

Son 5 yıllık günlük veride şu yapıyı kurar:

- **Y:** `EREGL.IS` (Erdemir kapanış fiyatı, `USDTRY=X` ile USD standardize)
- **X1:** `KEUA` (global carbon ETF proxy)
- **X2:** `TIO=F` (demir cevheri vadeli)
- **X3:** `USDTRY=X` (kur etkisi feature)

Model:

1. **ARIMAX** (Y ~ X2, order otomatik AIC seçimi)
2. **MLP** ile ARIMAX residual tahmini (X1/X2, residual lagleri, fiyat lagleri, değişim lagleri, rolling residual istatistikleri)
3. **Hibrit tahmin** = validation üzerinde öğrenilen lineer meta-birleştirici (ARIMAX + MLP residual)

## Notebook Yapısı

Notebook/Colab akışı 6 modül içerir:

1. Veri çekme, USD standardizasyonu, log-getiri dönüşümü, ffill/bfill, **leakage-safe ölçekleme** (yalnızca train fit), ADF/diff (train referanslı)
2. ARIMAX eğitimi ve model seçimi (p,d,q + exog adayları; validation RMSE + residual whiteness)
3. Eğitim/validation residual çıkarımı
4. TensorFlow/Keras MLP residual modeli (genişletilmiş özellik seti + kontrollü HP/seed araması)
5. Benchmark modelleri (Baseline + XGBoost tuning), öğrenilebilir hibrit birleşim ve RMSE/MAE karşılaştırması
6. **Rolling backtest** özeti (RMSE mean/std/wins) + başarı kriteri pass/fail çıktısı
7. Akademik görselleştirme (RMSE bar chart, test tahmin çizgileri, residual dağılımı, stres testi fan chart)

Veri bölme stratejisi: **%70 eğitim / %15 doğrulama / %15 test** (zaman sıralaması korunur) + expanding-window rolling backtest.

## Başarı Kriteri

Pipeline aşağıdaki hedefleri otomatik pass/fail olarak raporlar:

- Tek split test RMSE’de hibrit modelin ikinci en iyi modele karşı minimum iyileşme eşiği
- Rolling backtest pencerelerinde hibrit modelin kazanma oranı eşiği
- Tek split ve rolling hibrit RMSE için stabil bant üst sınırı kontrolü
- Tek split iyi, rolling kötü ise otomatik **FAIL** (katı karar kuralı)

Ek olarak pipeline artık:

- Data quality gate (eksik veri, indeks bütünlüğü, anomali sıçrama oranı) uygular
- Seed tekrarlı diagnostik baseline üretir (RMSE/MAE dağılımı)
- Modül bazlı RMSE ayrıştırması ve ilk kırılma noktası raporu üretir
- Ablation karşılaştırma tablosu üretir

## Kurulum

```bash
pip install -r requirements.txt
```

## Çalıştırma

```bash
python main.py
```

Testler:

```bash
pytest -q
```

Notebook sürümü:

- `notebooks/cbam_hybrid_pipeline.ipynb`
