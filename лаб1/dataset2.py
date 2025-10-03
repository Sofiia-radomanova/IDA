import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
import matplotlib.pyplot as plt

#налаштування для відображення в пандасі
pd.set_option("display.max_columns", None)   # показувати всі колонки
pd.set_option("display.max_rows", None)      # показувати всі рядки (обережно з великими датафреймами)
pd.set_option("display.width", None)         # прибирає перенесення рядків
pd.set_option("display.max_colwidth", None)  # повна довжина назв колонок
pd.set_option("display.float_format", '{:.3f}'.format)  # завжди показувати float у нормальному форматі




# Завантаження
file_path = r"E:\3 курс\1 сем\IDA\datasets\HR_Data_MNC_Data Science Lovers.csv"
df = pd.read_csv(file_path)

# Видаляємо непотрібні колонки
df = df.drop(columns=['Unnamed: 0'], errors='ignore')

print("Перші 5 рядків датасету:")
print(df.head())

# Категоризація цільової змінної Performance_Rating
df['target_group'] = pd.qcut(df['Performance_Rating'], q=3, labels=['Low', 'Medium', 'High'])

# Категоризована гістограма для Experience_Years
plt.figure(figsize=(10, 6))
for label in df['target_group'].unique():
    subset = df[df['target_group'] == label]
    plt.hist(subset['Experience_Years'], bins=15, alpha=0.6, label=label)
plt.title('Гістограма досвіду роботи (Experience_Years) за категоріями Performance_Rating')
plt.xlabel('Experience_Years')
plt.ylabel('Кількість')
plt.legend()
plt.grid(True)
plt.show()

from sklearn.preprocessing import MinMaxScaler

# Радіальна діаграма середніх значень ознак за групами
numeric_columns = df.select_dtypes(include=[np.number]).columns
features = numeric_columns.drop('Performance_Rating', errors='ignore')

# Обчислюємо середні значення ознак для кожної групи
grouped = df.groupby('target_group')[features].mean()

# Нормалізуємо (0-1), щоб усі ознаки були порівнянні
scaler = MinMaxScaler()
grouped_scaled = pd.DataFrame(scaler.fit_transform(grouped),
                              columns=grouped.columns,
                              index=grouped.index)

# Кути для діаграми
angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
angles += angles[:1]

# Побудова графіка
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for group_name, row in grouped_scaled.iterrows():
    values = row.tolist()
    values += values[:1]  # замикаємо контур
    ax.plot(angles, values, label=f'{group_name}')
    ax.fill(angles, values, alpha=0.25)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(features, fontsize=9)
ax.set_title("Радіальна діаграма середніх значень ознак за групами Performance_Rating")
ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
plt.show()

# Описова статистика по всіх числових змінних
desc_stats = pd.DataFrame({
    'mean': df.mean(numeric_only=True),
    'median': df.median(numeric_only=True),
    'std': df.std(numeric_only=True),
    'variance': df.var(numeric_only=True),
    'skewness': df.skew(numeric_only=True),
    'kurtosis': df.kurtosis(numeric_only=True),
    'min': df.min(numeric_only=True),
    'max': df.max(numeric_only=True)
})

print("\nОписова статистика по кожній числовій змінній:")
print(desc_stats.round(3))

# Висновок по рівню розсіянню, асиметрії
print("\nАналіз рівня, розсіяння та асиметрії:")
for column in numeric_columns:
    print(f"\nОзнака: {column}")
    print(f"  Рівень (середнє): {df[column].mean():.3f}")
    print(f"  Розсіяння (ст. відхилення): {df[column].std():.3f}")
    print(f"  Асиметрія: {skew(df[column]):.3f}")