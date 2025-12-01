import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/large_twitch_features.csv", encoding="utf-8")

print("Data:")
print(df.head(10), "\n")

print("Column:")
print(df.info(), "\n")

# --- Check dublication ---
duplicates = df.duplicated().sum()
print(f"Dublication: {duplicates}")
if duplicates > 0:
    df = df.drop_duplicates()

# --- Check null ---
missing_values = df.isna().sum()
print("Brakujące wartości w każdej kolumnie:\n", missing_values, "\n")

df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
df['updated_at'] = pd.to_datetime(df['updated_at'], errors='coerce')

invalid_dates = df[df['created_at'] > df['updated_at']]
print(f"Liczba błędnych rekordów (created_at > updated_at): {len(invalid_dates)}")
if len(invalid_dates) > 0:
    print("Nieprawidłowe wiersze:")
    print(invalid_dates)
    df = df[df['created_at'] <= df['updated_at']]

negative_views = df[df['views'] < 0]
print(f"Liczba rekordów z views < 0: {len(negative_views)}")
if len(negative_views) > 0:
    df = df[df['views'] >= 0]

print("\nDigital statistic:")
print(df[['views', 'life_time']].describe())

print("\nUnikalne wartości w kolumnach kategorycznych:")
for col in ['language', 'affiliate', 'mature', 'dead_account']:
    print(f"{col}: {df[col].unique()}")

# --- views ---
plt.figure(figsize=(10, 6))
sns.histplot(df["views"], bins=50, log_scale=(True, False), kde=True, color="royalblue", alpha=0.7)

mean_views = df["views"].mean()
median_views = df["views"].median()
plt.axvline(mean_views, color="orange", linestyle="--", linewidth=2, label=f"Średnia: {mean_views:.0f}")
plt.axvline(median_views, color="red", linestyle="--", linewidth=2, label=f"Mediana: {median_views:.0f}")

plt.title("Rozkład liczby wyświetleń (views)", fontsize=16, weight="bold")
plt.xlabel("Liczba wyświetleń (logarytmiczna skala)", fontsize=12)
plt.ylabel("Liczba użytkowników", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- life_time ---
plt.figure(figsize=(10, 6))
sns.histplot(df["life_time"], bins=40, kde=True, color="seagreen", alpha=0.6)

mean_life = df["life_time"].mean()
median_life = df["life_time"].median()
plt.axvline(mean_life, color="orange", linestyle="--", linewidth=2, label=f"Średnia: {mean_life:.0f} dni")
plt.axvline(median_life, color="red", linestyle="--", linewidth=2, label=f"Mediana: {median_life:.0f} dni")

plt.title("Rozkład czasu istnienia konta (life_time)", fontsize=16, weight="bold")
plt.xlabel("Czas (w dniach)", fontsize=12)
plt.ylabel("Liczba użytkowników", fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
