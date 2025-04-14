# Introduction
Modern airports handle large volumes of air traffic and must manage runway assignments efficiently to ensure safety, reduce delays, and maintain smooth operations. This project presents a machine learning-powered system that predicts optimal runway assignments for arriving aircraft based on factors like aircraft type, wind direction, and arrival time. It also includes a real-time scheduling mechanism to simulate runway availability and calculate wait times, thereby preventing overlap in runway usage.

# Technologies Used
•	Python
•	Flask (Web framework)
•	Scikit-learn (Machine Learning)
•	Pandas & NumPy (Data processing)
•	Matplotlib & Seaborn (Visualization)
•	HTML & CSS (Frontend)
Problem Definition
Conventional runway assignment does not dynamically account for current traffic, leading to delays or conflicts when two aircraft are assigned the same runway at the same time. To address this, the system should learn from historical runway data to suggest likely runways, while also checking for conflicts using a real-time scheduling queue.

# Implementation
Dataset:
The synthetic dataset includes columns such as Aircraft_Type, Wind_Direction, Arrival_Hour, and Runway_Assigned. Example entries:
A320, 270°, 14.5h → 27L
B737, 90°, 9.0h → 09R
B747, 180°, 7.5h → 18L

# Methodology:
• Label encoding is applied to aircraft and runway names.
• A Decision Tree Classifier is trained using features: Aircraft_Type, Wind_Direction, and Arrival_Hour.
• The trained model predicts the runway based on new inputs.
• A real-time queue checks both dataset history and current simulation to find the next available safe time slot, adding 3-minute intervals to avoid overlaps.

# Results
Sample Prediction:
Input: A320 at 9.0h with wind 90°
→ Predicted Runway: 09R
→ Assigned ETA: 09:05
→ Wait Time: 5 minutes
Queue Snapshot:
09R — B737 — 09:00
09R — A320 — 09:05
09R — A380 — 09:10
