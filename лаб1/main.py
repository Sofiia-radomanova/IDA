import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

# Завантаження
data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target


print("Перші 5 рядків датасету:")
print(df.head())

#Категоризація цільової змінної
df['target_group'] = pd.qcut(df['target'], q=3, labels=['Low', 'Medium', 'High'])

#Категоризована гістограма
plt.figure(figsize=(10, 6))
for label in df['target_group'].unique():
    subset = df[df['target_group'] == label]
    plt.hist(subset['bmi'], bins=15, alpha=0.6, label=label)
plt.title('Гістограма індексу маси тіла (BMI) за категоріями цільової змінної')
plt.xlabel('BMI')
plt.ylabel('Кількість')
plt.legend()
plt.grid(True)
plt.show()

#Радіальна діаграма
# Групування по категоріях
grouped = df.groupby('target_group').mean(numeric_only=True)

features = data.feature_names
angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for group_name, row in grouped.iterrows():
    values = row[features].tolist()
    values += values[:1]
    ax.plot(angles, values, label=f'{group_name}')
    ax.fill(angles, values, alpha=0.2)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(features)
ax.set_title("Радіальна діаграма середніх значень ознак за групами")
ax.legend()
plt.show()

# Описова статистика по всіх змінних
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

print("\nОписова статистика по кожній змінній:")
print(desc_stats.round(3))

#Висновок по рівню, розсіянню, асиметрії
print("\nАналіз рівня, розсіяння та асиметрії:")
for column in data.feature_names:
    print(f"\nОзнака: {column}")
    print(f"  Рівень (середнє): {df[column].mean():.3f}")
    print(f"  Розсіяння (ст. відхилення): {df[column].std():.3f}")
    print(f"  Асиметрія: {skew(df[column]):.3f}")
