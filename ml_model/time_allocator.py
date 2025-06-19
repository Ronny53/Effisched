import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from ml_model.priority_predictor_task_sorter import Event, EventPriorityQueue
import os

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
# Loading Priority Predictor Pipeline
with open(os.path.join(MODEL_DIR, "priority_model.pkl"), "rb") as f:
    priority_pipeline = pickle.load(f)

# Loading Time Allocator Pipeline
with open(os.path.join(MODEL_DIR, "alloc_hours_model.pkl"), "rb") as f:
    time_pipeline = pickle.load(f)

time_model   = time_pipeline['model']
time_scaler  = time_pipeline['scaler']
time_encoder = time_pipeline['encoder']

def allocate_hours(tasks, free_hours):
    queue = EventPriorityQueue()
    for t in tasks:
        evt = Event(
            event_name=t['event_name'],
            deadline=t['deadline'],
            duration=t['duration'],
            priority=t.get('priority')
        )
        queue.add_event(evt)
    allocations = []
    remaining_hours = free_hours
    while remaining_hours > 0 and len(queue) > 0:
        event = queue.pop_event()
        if event is None:
            break
        # Build features for time allocation
        feat = event.to_feature_dict()
        if event.event_name in time_encoder.classes_:
            task_code = int(time_encoder.transform([event.event_name])[0])
        else:
            task_code = 0
        df_time = pd.DataFrame([{
            'duration':       feat['duration'],
            'priority':       event.priority,
            'hours_left':     feat['hours_remaining'],
            'days_remaining': max(1, int(feat['hours_remaining'] // 24)),
            'task_enc':       task_code
        }])
        numeric_cols = ['duration', 'priority', 'hours_left', 'days_remaining']
        num_scaled = time_scaler.transform(df_time[numeric_cols])
        task_enc_arr = df_time[['task_enc']].to_numpy().reshape(-1, 1)
        X_scaled = np.hstack([num_scaled, task_enc_arr])
        df_scaled = pd.DataFrame(
            time_scaler.transform(df_time[numeric_cols]),
            columns=numeric_cols
        )
        df_scaled['task_enc'] = df_time['task_enc']
        alloc = int(round(time_model.predict(df_scaled)[0]))
        alloc = max(1, min(alloc, remaining_hours, int(event.duration)))
        allocations.append({'event_name': event.event_name, 'allocated_hours': alloc})
        event.duration -= alloc
        remaining_hours -= alloc
        if event.duration > 0:
            new_feat = event.to_feature_dict()
            df_prio = pd.DataFrame([{
                'duration':        new_feat['duration'],
                'hours_remaining': new_feat['hours_remaining'],
                'time_pressure':   new_feat['time_pressure']
            }])
            new_prio = int(round(priority_pipeline.predict(df_prio)[0]))
            event.priority = new_prio
            queue.add_event(event)
    return allocations

if __name__ == "__main__":
    # Example usage for testing
    sample_tasks = [
        {"event_name": "Online Course", "deadline": "2025-06-27T12:00:00", "duration": 3},
        {"event_name": "Project Work", "deadline": "2025-06-20T18:00:00", "duration": 4},
        {"event_name": "Assignnt", "deadline": "2025-06-30T09:00:00", "duration": 8},
    ]
    free_hours = float(input("Enter your free hours available today: "))
    result = allocate_hours(sample_tasks, free_hours)
    print(result)
