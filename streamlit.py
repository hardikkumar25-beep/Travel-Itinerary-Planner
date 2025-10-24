import streamlit as st
from ta import Itinerary, city_info, Hotels, summarize, add, sub, multiply, divide, calculator 
from pprint import pprint

st.set_page_config(page_title="Travel Agent", page_icon="🗺️", layout="wide")

st.title("🧳 AI-Powered Travel Itinerary Planner")
st.markdown("Plan your perfect trip using AI! 🌍")

city = st.text_input("🏙️ Enter a city name:", placeholder="e.g. Delhi")

if st.button("Generate Itinerary"):
    with st.spinner("✨ Planning your trip..."):
        try:
            plan = Itinerary(city, city_info, Hotels)
            attr_data = plan.attractions()
            hotel_data = plan.hotels()
            try:
                weather_data = plan.weather()
            except Exception as e:
                st.warning(f"⚠️ Weather data not available for {city}. Skipping... ({e})")
                weather_data = {}

            merged = attr_data.copy()
            merged["hotels"] = hotel_data["hotels"]
            merged["weather"] = weather_data
            final=summarize(merged)
            st.subheader(f"🌆 5-Day Itinerary for {city}")
            st.markdown(final)

            
            st.subheader("🏨 If you want to explore other hotels,Here is the list:")
            for i, h in enumerate(merged["hotels"], start=1):
                st.markdown(f"**{i}. {h['name']}** — ₹{h['cost_per_night']}/night ({h['location']})")

            avg_cost = sum(h["cost_per_night"] for h in merged["hotels"]) / len(merged["hotels"])
            st.info(f"💰 Average hotel cost: ₹{avg_cost:.0f}/night")

        except Exception as e:
            st.error(f"⚠️ Something went wrong: {e}")
st.divider()
st.title("🧮 Calculator ")
st.markdown("Calculate your budget")
query = st.text_input("🧮 Enter your query:", placeholder="e.g. Add 10000 to 35000")

if st.button("Calculate"):
    with st.spinner("Calculating your query..."):
        try:
            calc = calculator()
            answer = calc.invoke({"messages": [query]})
            result = answer["messages"][-1]
            st.markdown(result.content if hasattr(result, "content") else str(result))
        except Exception as e:
            st.error(f"⚠️ Something went wrong: {e}")


