import joblib
import numpy as np

print("=== Marketing Response Predictor Test ===")

try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")

    print("✅ Model loaded successfully")

    # Test prediction
    test_input = np.zeros(len(feature_names))
    test_input[feature_names.index("Income")] = (
        60000 if "Income" in feature_names else 0
    )
    test_input[feature_names.index("Age")] = 45 if "Age" in feature_names else 0
    test_input[feature_names.index("Total_Spending")] = (
        500 if "Total_Spending" in feature_names else 0
    )

    test_scaled = scaler.transform([test_input])
    prob = model.predict_proba(test_scaled)[0][1]

    result = "Likely to Respond" if prob > 0.5 else "Not Likely to Respond"
    print(f"Test prob: {prob:.1%} → {result}")
    print("✅ Test passed!")

except Exception as e:
    print(f"❌ Error: {e}")
