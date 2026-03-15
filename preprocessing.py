import pandas as pd


def preprocess_transactions(input_csv, output_csv):

    df = pd.read_csv(input_csv)

    df = df.dropna(how="all")

    df["value_date"] = pd.to_datetime(df["value_date"], errors="coerce")
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")

    df = df.dropna(subset=["value_date"])

    df["withdrawal"] = pd.to_numeric(df["withdrawal"], errors="coerce").fillna(0)
    df["deposit"] = pd.to_numeric(df["deposit"], errors="coerce").fillna(0)
    df["balance"] = pd.to_numeric(df["balance"], errors="coerce")

    df["amount"] = df["deposit"] - df["withdrawal"]

    df["month"] = df["value_date"].dt.month
    df["weekday"] = df["value_date"].dt.weekday

    df["remarks"] = (
        df["remarks"]
        .astype(str)
        .str.replace("\n", " ", regex=False)
        .str.replace("/", " ", regex=False)
        .str.lower()
        .str.strip()
    )

    # -----------------------------
    # Merchant extraction
    # -----------------------------
    def extract_merchant(text):

        if "swiggy" in text:
            return "swiggy"
        if "zomato" in text:
            return "zomato"

        if "amazon" in text:
            return "amazon"
        if "flipkart" in text:
            return "flipkart"
        if "myntra" in text:
            return "myntra"

        if "uber" in text:
            return "uber"
        if "ola" in text:
            return "ola"
        if "petrol" in text or "fuel" in text:
            return "fuel"

        if "recharge" in text:
            return "recharge"

        if "electricity" in text or "water" in text or "gas" in text:
            return "utilities"

        if "bajaj" in text or "tata" in text:
            return "utilities"

        if "netflix" in text or "spotify" in text:
            return "entertainment"

        if "hospital" in text or "medical" in text or "pharmacy" in text:
            return "medical"

        if "school" in text or "college" in text:
            return "education"

        if "loan" in text or "emi" in text:
            return "loan"

        if "upi" in text or "neft" in text or "imps" in text:
            return "bank_transfer"

        if "atm" in text:
            return "atm"

        return "other"

    df["merchant"] = df["remarks"].apply(extract_merchant)

    # -----------------------------
    # Category mapping
    # -----------------------------
    category_map = {

        "swiggy": "food",
        "zomato": "food",

        "amazon": "shopping",
        "flipkart": "shopping",
        "myntra": "shopping",

        "uber": "transport",
        "ola": "transport",
        "fuel": "transport",

        "recharge": "recharge",

        "utilities": "utilities",

        "entertainment": "entertainment",

        "medical": "health",

        "education": "education",

        "bank_transfer": "bank",

        "atm": "cash",

        "loan": "loan",

        "other": "other"
    }

    df["category"] = df["merchant"].map(category_map)

    df["category"] = df["category"].fillna("other")

    df.to_csv(output_csv, index=False)

    return df