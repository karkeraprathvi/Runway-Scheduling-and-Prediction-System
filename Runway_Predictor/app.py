from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from decimal import Decimal
import datetime

app = Flask(__name__)

# Load and preprocess dataset
df = pd.read_csv('runway_dataset.csv')
df['Original_Aircraft_Type'] = df['Aircraft_Type']  # Keep original names for reference

le_aircraft = LabelEncoder()
df['Aircraft_Type'] = le_aircraft.fit_transform(df['Aircraft_Type'])

le_runway = LabelEncoder()
df['Runway_Assigned'] = le_runway.fit_transform(df['Runway_Assigned'])

X = df[['Aircraft_Type', 'Wind_Direction', 'Arrival_Hour']]
y = df['Runway_Assigned']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)

# Visualization plots
os.makedirs("static/plots", exist_ok=True)
sns.countplot(x='Runway_Assigned', data=df)
plt.savefig('static/plots/runway_count.png')
plt.clf()

sns.barplot(x='Runway_Assigned', y='Arrival_Hour', data=df, estimator='mean')
plt.savefig('static/plots/arrival_hour_avg.png')
plt.clf()

sns.boxplot(x='Runway_Assigned', y='Wind_Direction', data=df)
plt.savefig('static/plots/wind_vs_runway.png')
plt.clf()

# Runway queue state
runway_queue_state = {}

# Convert existing dataset to schedule map (runway -> list of time blocks)
df_schedule = {}
for _, row in df.iterrows():
    runway = le_runway.inverse_transform([row['Runway_Assigned']])[0]
    hour = row['Arrival_Hour']
    df_schedule.setdefault(runway, []).append(hour)

def calculate_wait_time(predicted_runway, arrival_hour, user_input):
    scheduled_times = sorted(df_schedule.get(predicted_runway, []))
    queue = runway_queue_state.get(predicted_runway, [])

    # Combine dataset times and queue times, rounded
    occupied_times = [round(t, 2) for t in scheduled_times]
    occupied_times += [round(entry['arrival_hour'], 2) for entry in queue]
    occupied_times = sorted(set(occupied_times))

    proposed_time = round(float(arrival_hour), 2)

    # Avoid conflicts by checking rounded times
    while round(proposed_time, 2) in occupied_times:
        proposed_time = round(proposed_time + 0.05, 2)  # move by ~3 minutes


    
    decimal_time = Decimal(str(proposed_time))
    eta_hour = int(decimal_time)
    eta_minute_decimal = (decimal_time - eta_hour) * 100  # Interpret .30 as 30 mins
    eta_minute = int(eta_minute_decimal)

# Ensure minute stays valid (in case someone enters 9.75 → 75 minutes)
    if eta_minute >= 60:
        eta_hour += eta_minute // 60
        eta_minute = eta_minute % 60


    now = datetime.datetime.now()
    eta = now.replace(hour=eta_hour, minute=eta_minute, second=0, microsecond=0)

    queue_entry = {
        "runway": predicted_runway,
        "aircraft": user_input['aircraft'],
        "eta": eta.strftime("%H:%M"),
        "arrival_hour": round(proposed_time, 2)
    }
    runway_queue_state.setdefault(predicted_runway, []).append(queue_entry)
    wait_time_minutes = int((proposed_time - float(arrival_hour)) * 60)
    return wait_time_minutes, runway_queue_state[predicted_runway]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        aircraft = request.form['aircraft'].strip().upper()
        wind = float(request.form['wind'])
        hour = float(request.form['hour'])
        user_input = {'aircraft': aircraft, 'wind': wind, 'hour': hour}

        if aircraft not in le_aircraft.classes_:
            return render_template('index.html', accuracy=accuracy, prediction="Unknown Aircraft Type", 
                                   user_input=user_input, wait_time=None, runway_queue=[])

        aircraft_encoded = le_aircraft.transform([aircraft])[0]
        input_data = [[aircraft_encoded, wind, hour]]
        prediction_encoded = clf.predict(input_data)[0]
        predicted_runway = le_runway.inverse_transform([prediction_encoded])[0]

        wait_time, runway_queue = calculate_wait_time(predicted_runway, hour, user_input)

        return render_template('index.html', accuracy=accuracy, prediction=predicted_runway,
                               user_input=user_input, wait_time=wait_time, runway_queue=runway_queue)

    return render_template('index.html', accuracy=accuracy, prediction=None, 
                           user_input={'aircraft': '', 'wind': '', 'hour': ''}, 
                           wait_time=None, runway_queue=[])

if __name__ == '__main__':
    app.run(debug=True)