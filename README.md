# article_41524_pearl
日本発の衛星データプラットフォーム Tellus のオウンドメディア「宙畑」の記事、https://sorabatake.jp/41524 で利用しているコードです。

GCOM-Cの海面水温情報を用いて、沿岸部の海面温の可視化を行います。

# 構成
level2_sst.py
GCOM-Cの海面温度のL2データを可視化するコード。
海面水温（SST）やクロロフィルa濃度といった具体的な物理量が計算され、ピクセル単位での科学的分析が可能になります。

8days_sst.py
GCOM-Cの海面温度の8日平均（L3）のデータを可視化するコード。L2データに比べて雲や欠測の影響を受けにくく、より広範囲の水温分布を安定して得られます。

# ライセンス、利用規約
ソースコードのライセンスは CC0-1.0（Creative Commons Zero v1.0 Universal）ライセンスです。
今回コード内で GCOM-C データを用いております。利用ポリシーは以下をご参考下さい。 https://www.tellusxdp.com/ja/catalog/data/gcom-c_chla_nrt.html ※サイトの閲覧にはTellusへのログインが必要です。
