import pandas as pd
import numpy as np
from scipy.stats import shapiro, bartlett, ttest_ind, mannwhitneyu
import sys

# コマンドライン引数でCSVファイルのパスを取得
if len(sys.argv) < 2:
    print("Usage: python script_name.py <path_to_csv_file>")
    sys.exit()

csv_file_path = sys.argv[1]

# Shift-JISエンコーディングでCSVファイルを読み込む
data_shiftjis = pd.read_csv(csv_file_path, encoding='shift-jis')

# 欠損値を持つ行を削除
data_cleaned = data_shiftjis.dropna()

# シャピロ・ウィルク検定を適用
shapiro_test_pole = shapiro(data_cleaned["pole"])
shapiro_test_no_pole = shapiro(data_cleaned["no_pole"])

print("シャピロ・ウィルク検定結果")
print("pole グループ：", shapiro_test_pole)
print("no_pole グループ：", shapiro_test_no_pole)
print("-" * 50)

# Bartlettの検定を適用
bartlett_test_result = bartlett(data_cleaned["pole"], data_cleaned["no_pole"])

print("Bartlettの検定結果：", bartlett_test_result)
print("-" * 50)

# t検定を適用
ttest_result = ttest_ind(data_cleaned["pole"], data_cleaned["no_pole"])

print("t検定結果：", ttest_result)

# Welchのt検定を適用
welch_ttest_result = ttest_ind(data_cleaned["pole"], data_cleaned["no_pole"], equal_var=False)

print("Welchのt検定結果：", welch_ttest_result)

# マン・ホイットニーU検定を適用
mannwhitneyu_result = mannwhitneyu(data_cleaned["pole"], data_cleaned["no_pole"])

print("マン・ホイットニーU検定結果：", mannwhitneyu_result)

# 自由度の計算
df_bartlett = 2 - 1  # Bartlettの検定の自由度

# t検定の自由度
n1 = len(data_cleaned["pole"])
n2 = len(data_cleaned["no_pole"])
df_ttest = n1 + n2 - 2

# Welchのt検定の自由度
s1 = np.std(data_cleaned["pole"], ddof=1)
s2 = np.std(data_cleaned["no_pole"], ddof=1)
df_welch = ((s1**2 / n1 + s2**2 / n2)**2) / (((s1**2 / n1)**2 / (n1 - 1)) + ((s2**2 / n2)**2 / (n2 - 1)))

print("Bartlettの検定の自由度：", df_bartlett)
print("t検定の自由度：", df_ttest)
print("Welchのt検定の自由度：", df_welch)


# 平均値と中央値を計算
mean_pole = np.mean(data_cleaned["pole"])
mean_no_pole = np.mean(data_cleaned["no_pole"])
median_pole = np.median(data_cleaned["pole"])
median_no_pole = np.median(data_cleaned["no_pole"])

# 結果を出力
print("-" * 50)
print(f"Mean of pole group: {mean_pole}")
print(f"Mean of no_pole group: {mean_no_pole}")
print(f"Median of pole group: {median_pole}")
print(f"Median of no_pole group: {median_no_pole}")
print("-" * 50)