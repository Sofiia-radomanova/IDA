
from sklearn.datasets import fetch_olivetti_faces
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

data = fetch_olivetti_faces()
X = data.data       # ознаки (4096 пікселів = 64x64)
y = data.target     # мітки класів (40 людей)

# Розбивка на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=5)
#stratify гарантує рівномірне розбиття класів даних в тренувальній і тестовій вибірках

clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=7)
clf.fit(X_train, y_train) # навчання на 100 деревах
yy = clf.predict(X_test) #передбачення результатів на тестовій вибірці

fig, axes = plt.subplots(3, 5, figsize=(10,6))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(64,64), cmap='gray')
    ax.set_title(f"Реал: {y_test[i]}\nПред: {yy[i]}")
    ax.axis('off')
plt.tight_layout()
plt.show()


# Підрахунок скільки разів кожен клас зустрівся в тестовій вибірці
classes = np.arange(40)
real_counts = np.bincount(y_test, minlength=40)
pred_counts = np.bincount(yy, minlength=40)

plt.figure(figsize=(12,6))
width = 0.4

plt.bar(classes - width/2, real_counts, width=width, label='Реальні')
plt.bar(classes + width/2, pred_counts, width=width, label='Передбачені')
plt.xlabel('Клас (особа)')
plt.ylabel('Кількість зображень')
plt.title('Порівняння реальних та передбачених класів')
plt.legend()
plt.grid(axis='y')
plt.show()