import streamlit as st
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("SERP_API_KEY")


# ---------------- UI ----------------
st.title("🛒 Price Comparison App")
product = st.text_input("Enter product name")

# ---------------- BUTTON ----------------
if st.button("Compare"):

    if not product.strip():
        st.warning("Please enter a product name")
        st.stop()

    # ---------------- API CALL ----------------
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_shopping",
        "q": product,
        "gl": "in",
        "hl": "en",
        "api_key": API_KEY
    }

    indian_sites = [
        "flipkart", "amazon", "croma", "reliance",
        "tatacliq", "vijaysales", "snapdeal", "jiomart"
    ]

    response = requests.get(url, params=params)
    data = response.json()

    if "shopping_results" not in data:
        st.error("No data received from API")
        st.stop()

    # ---------------- EXTRACT DATA ----------------
    products = []
    for item in data["shopping_results"]:
        if "price" in item and "title" in item and "source" in item:
            products.append({
                "title": item["title"],
                "price": item["price"],
                "website": item["source"]
            })

    df = pd.DataFrame(products)

    if df.empty:
        st.warning("No products found")
        st.stop()

    # ---------------- CLEAN PRICE ----------------
    df["price"] = df["price"].str.replace(r"[^\d.]", "", regex=True)
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"])

    # ---------------- FILTER: INDIAN WEBSITES ----------------
    df = df[df["website"].str.lower().str.contains("|".join(indian_sites))]

    # ---------------- FILTER: USED / REFURBISHED ----------------
    exclude_keywords = ["restored", "refurbished", "used", "renewed"]
    df = df[~df["title"].str.lower().str.contains("|".join(exclude_keywords))]

    # ---------------- FILTER: ACCESSORIES ----------------
    accessory_keywords = [
        "cover", "case", "flip", "back cover", "tempered",
        "glass", "protector", "skin", "charger", "cable"
    ]
    df = df[~df["title"].str.lower().str.contains("|".join(accessory_keywords))]

    # ---------------- SAFETY CHECK BEFORE ML ----------------
    if df.empty:
        st.warning("No valid products found")
        st.info("New or unreleased products may not be listed yet")
        st.stop()

    # ---------------- ML SIMILARITY ----------------
    query = product.lower()

    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(df["title"].str.lower())
    query_vec = vectorizer.transform([query])

    similarity_scores = cosine_similarity(query_vec, vectors)[0]
    df["similarity"] = similarity_scores   # ✅ similarity created ONCE

    # ---------------- SMART FILTERING ----------------
    filtered_df = df.copy()

    # Enforce model number (iphone 12 / 13 / etc.)
    query_tokens = product.lower().split()
    model_numbers = [t for t in query_tokens if t.isdigit()]

    if model_numbers:
        filtered_df = filtered_df[
            filtered_df["title"].str.lower().apply(
                lambda x: any(num in x for num in model_numbers)
            )
        ]

    # Fallback if model filtering removes all
    if filtered_df.empty:
        filtered_df = df.copy()

    # Take top relevant products
    filtered_df = (
        filtered_df
        .sort_values(by="similarity", ascending=False)
        .head(5)
    )

    if filtered_df.empty:
        st.warning("No matching products found")
        st.stop()

    # ---------------- BEST DEAL (RELEVANCE FIRST) ----------------
    best_deal = (
        filtered_df
        .sort_values(by=["similarity", "price"], ascending=[False, True])
        .iloc[0]
    )

    # ---------------- OUTPUT ----------------
    st.subheader("✅ Best Deal")
    st.write(best_deal)

    st.metric("Lowest Price", f"₹{best_deal['price']:,.0f}")
    st.metric("Website", best_deal["website"])

    st.subheader("🔍 All Matched Products")
    st.dataframe(filtered_df[["title", "price", "website", "similarity"]])


