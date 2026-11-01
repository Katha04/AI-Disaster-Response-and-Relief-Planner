import pandas as pd

# Load existing dataset
df = pd.read_csv("data/disaster_dataset.csv")

# Additional synthetic disaster examples
new_data = [

    ["Flood", "Severe flooding affected many homes and roads.", "Barisal", 4500, 14, "High", "Rainy", "Yes", "Critical"],
    ["Flood", "Heavy rain caused widespread flooding in low-lying areas.", "Cumilla", 2800, 10, "High", "Rainy", "Yes", "High"],
    ["Flood", "Moderate flooding affected several streets.", "Mymensingh", 900, 7, "Medium", "Rainy", "No", "Medium"],
    ["Flood", "Minor flooding affected a few local roads.", "Rajshahi", 200, 3, "Low", "Rainy", "No", "Low"],

    ["Earthquake", "A powerful earthquake caused major structural damage.", "Dhaka", 6000, 3, "High", "Clear", "Yes", "Critical"],
    ["Earthquake", "Earthquake damaged several buildings and injured residents.", "Sylhet", 2500, 2, "High", "Clear", "Yes", "High"],
    ["Earthquake", "Moderate shaking caused cracks in buildings.", "Rajshahi", 700, 1, "Medium", "Clear", "No", "Medium"],
    ["Earthquake", "Minor earthquake caused no major structural damage.", "Khulna", 100, 1, "Low", "Clear", "No", "Low"],

    ["Fire", "A large industrial fire spread to nearby buildings.", "Chittagong", 1800, 6, "High", "Clear", "Yes", "Critical"],
    ["Fire", "A major building fire required emergency evacuation.", "Dhaka", 1200, 5, "High", "Clear", "Yes", "High"],
    ["Fire", "A moderate fire damaged part of a commercial building.", "Khulna", 400, 3, "Medium", "Clear", "No", "Medium"],
    ["Fire", "A small kitchen fire was quickly contained.", "Sylhet", 30, 1, "Low", "Clear", "No", "Low"],

    ["Landslide", "A massive landslide blocked roads and damaged houses.", "Rangamati", 3500, 12, "High", "Rainy", "Yes", "Critical"],
    ["Landslide", "A large landslide damaged several homes and roads.", "Bandarban", 1800, 8, "High", "Rainy", "Yes", "High"],
    ["Landslide", "A moderate landslide blocked a local road.", "Sylhet", 500, 5, "Medium", "Rainy", "No", "Medium"],
    ["Landslide", "A small landslide affected a rural pathway.", "Moulvibazar", 80, 2, "Low", "Rainy", "No", "Low"],

    ["Cyclone", "A severe cyclone caused widespread coastal destruction.", "Barisal", 15000, 20, "High", "Storm", "Yes", "Critical"],
    ["Cyclone", "A strong cyclone damaged houses and coastal infrastructure.", "Patuakhali", 8000, 16, "High", "Storm", "Yes", "Critical"],
    ["Cyclone", "A moderate cyclone affected coastal roads and homes.", "Bhola", 2500, 12, "Medium", "Storm", "Yes", "High"],
    ["Cyclone", "A weak cyclone caused limited local damage.", "Khulna", 500, 6, "Low", "Storm", "No", "Medium"],

    ["Flood", "River overflow caused extensive flooding and displacement.", "Faridpur", 7000, 16, "High", "Rainy", "Yes", "Critical"],
    ["Flood", "Continuous rainfall flooded several neighborhoods.", "Narayanganj", 3200, 11, "High", "Rainy", "Yes", "High"],
    ["Flood", "Moderate water levels affected local markets.", "Tangail", 800, 6, "Medium", "Rainy", "No", "Medium"],
    ["Flood", "Short rainfall caused minor waterlogging.", "Gazipur", 120, 2, "Low", "Rainy", "No", "Low"],

    ["Earthquake", "A major earthquake severely damaged infrastructure.", "Chittagong", 5000, 4, "High", "Clear", "Yes", "Critical"],
    ["Earthquake", "Strong shaking damaged older buildings.", "Dhaka", 2200, 2, "High", "Clear", "Yes", "High"],
    ["Earthquake", "Moderate shaking caused minor structural cracks.", "Sylhet", 500, 1, "Medium", "Clear", "No", "Medium"],
    ["Earthquake", "Light earthquake caused no serious damage.", "Rajshahi", 70, 1, "Low", "Clear", "No", "Low"],

    ["Fire", "A large warehouse fire caused extensive property damage.", "Narayanganj", 2000, 7, "High", "Clear", "Yes", "Critical"],
    ["Fire", "A commercial fire required evacuation of nearby residents.", "Chittagong", 900, 4, "High", "Clear", "Yes", "High"],
    ["Fire", "A moderate shop fire caused limited damage.", "Rajshahi", 250, 2, "Medium", "Clear", "No", "Medium"],
    ["Fire", "A small electrical fire was contained quickly.", "Dhaka", 40, 1, "Low", "Clear", "No", "Low"],

    ["Landslide", "Heavy rainfall triggered a major landslide.", "Bandarban", 4200, 14, "High", "Rainy", "Yes", "Critical"],
    ["Landslide", "A landslide damaged several houses and blocked roads.", "Rangamati", 1600, 9, "High", "Rainy", "Yes", "High"],
    ["Landslide", "Moderate soil movement blocked a village road.", "Sylhet", 400, 4, "Medium", "Rainy", "No", "Medium"],
    ["Landslide", "Small soil movement affected a rural road.", "Moulvibazar", 60, 2, "Low", "Rainy", "No", "Low"],

    ["Cyclone", "A powerful cyclone destroyed coastal infrastructure.", "Patuakhali", 11000, 18, "High", "Storm", "Yes", "Critical"],
    ["Cyclone", "Strong winds damaged houses across coastal areas.", "Bhola", 5000, 14, "High", "Storm", "Yes", "High"],
    ["Cyclone", "Moderate storm affected local roads.", "Khulna", 1200, 8, "Medium", "Storm", "No", "Medium"],
    ["Cyclone", "A weak storm caused minor property damage.", "Barisal", 250, 4, "Low", "Storm", "No", "Low"]
]

columns = [
    "disaster_type",
    "description",
    "location",
    "people_affected",
    "duration_hours",
    "infrastructure_damage",
    "weather_condition",
    "evacuation_required",
    "severity"
]

new_df = pd.DataFrame(new_data, columns=columns)

# Combine original and synthetic data
combined_df = pd.concat([df, new_df], ignore_index=True)

# Save expanded dataset
combined_df.to_csv("data/disaster_dataset.csv", index=False)

print("Dataset expanded successfully!")
print("Original records:", len(df))
print("New records:", len(new_df))
print("Total records:", len(combined_df))
print("\nSeverity distribution:")
print(combined_df["severity"].value_counts())