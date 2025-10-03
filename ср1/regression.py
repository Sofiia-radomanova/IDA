
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

data = fetch_california_housing()

X = data.data       # ознаки (4096 пікселів = 64x64)
y = data.target
# Розбивка на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
#stratify гарантує рівномірне розбиття класів даних в тренувальній і тестовій вибірках

clf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=7)
clf.fit(X_train, y_train) # навчання на 100 деревах
yy = clf.predict(X_test)

plt.figure(figsize=(8,6))
plt.scatter(y_test, yy, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # ідеальна лінія
plt.xlabel("Реальні ціни")
plt.ylabel("Передбачені ціни")
plt.title("Передбачені ціни на нерухомість vs Реальні")
plt.grid(True)
plt.show()
