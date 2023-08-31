import streamlit as st
import speech_recognition as sr
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import xgboost as xgb

import time

from datetime import datetime

df = pd.read_csv('weather.csv')

X = df[['Humidity', 'Wind Speed (km/h)', 'Visibility (km)', 'Pressure (millibars)']]
y = df['Temperature (C)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = xgb.XGBRegressor(
    objective='reg:squarederror',  # For regression
    max_depth=3,
    learning_rate=0.1,
    n_estimators=100
)

# Training the model
model.fit(X_train, y_train)






st.title("Simple chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])




# Define the chatbot responses
greetings = {
    "hello": "Hello! How can I assist you?",
    "hi": "Hi there! How can I help you?",
    "hey": "Hey! What can I do for you?",
    "how are you": "I'm just a chatbot, but I'm here to help.",
    "how's it going": "Hi! How can I assist you today?",
    "default": "I'm sorry, I didn't quite catch that as a greeting."
}

weather = {
    "can you tell me about the weather today": "Sure, can you fill  these parameters first?",
    "what's the temperature for today": "Sure, can you fill in these parameters first?",
    "default": "I'm sorry, I didn't quite catch that as a greeting."
}
pleasure = {
    "thanks": "You're welcome! If you have any more questions in the future, feel free to ask. Have a great day!",
    "thank you": "You're welcome! If you have any more questions in the future, feel free to ask. Have a great day!",
    "you helped me a lot ": "I'm glad I could help! If you ever need assistance again, don't hesitate to reach out. Have a wonderful day and happy coding!",
    "goodbye": "Goodbye! Have a great day!"
}

responses = {
    **greetings,
    **weather,
    **pleasure,
    "default": "I'm sorry, I didn't quite understand that."
}

# Initialize speech recognizer
recognizer = sr.Recognizer()

if st.button("Start Speech"):
    with sr.Microphone() as source:
        st.write("Listening... Speak now!")
        audio = recognizer.listen(source, timeout=10)

        try:
            user_input = recognizer.recognize_google(audio).lower()
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.chat_message("user")
            st.markdown(user_input)
        except sr.UnknownValueError:
            st.write("Sorry, I couldn't understand what you said.")
            user_input = None

if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process user input
    user_input = prompt.lower()

    # Determine assistant's response based on user input
    if user_input in responses:
        assistant_response = responses[user_input]
    else:
        assistant_response = responses["default"]

    # Handle weather-related responses
    if "weather" in user_input:
        assistant_response = weather[user_input]
        st.session_state.expecting_weather_parameters = True

    else:
        st.session_state.expecting_weather_parameters = False

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Collect weather parameters if expecting them
if hasattr(st.session_state, "expecting_weather_parameters") and st.session_state.expecting_weather_parameters:
    humidity = st.slider("Humidity", min_value=0, max_value=100, step=1)
    wind_speed = st.slider("Wind Speed (km/h)", min_value=0, max_value=100, step=1)
    visibility = st.slider("Visibility (km)", min_value=0, max_value=100, step=1)
    pressure = st.slider("Pressure (millibars)", min_value=0, max_value=2000, step=1)

    # Store the collected parameters in the session state
    st.session_state.collected_parameters = {
        "humidity": humidity,
        "wind_speed": wind_speed,
        "visibility": visibility,
        "pressure": pressure
    }

    # Use the collected parameters for prediction
    if st.button("Predict Temperature"):
        model_input = pd.DataFrame({
            "Humidity": [humidity],
            "Wind Speed (km/h)": [wind_speed],
            "Visibility (km)": [visibility],
            "Pressure (millibars)": [pressure]
        })

        # Make a prediction using the XGBoost model
        predicted_temperature = model.predict(model_input)[0]

        # Display the predicted temperature to the user
        st.write(f"Predicted Temperature: {predicted_temperature:.2f} °C")
