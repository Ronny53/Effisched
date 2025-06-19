import pickle
import heapq
import pandas as pd
from datetime import datetime
import os

CURRENT_TIME = datetime.now()

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(MODEL_DIR, "priority_model.pkl"), "rb") as f:
    loaded_pipeline = pickle.load(f)

class Event:
    def __init__(self, event_name, start_time=None, end_time=None,
                 deadline=None, duration=None, priority=None):
        self.event_name = event_name
        self.start_time = datetime.fromisoformat(start_time) if start_time else None
        self.end_time = datetime.fromisoformat(end_time) if end_time else None
        self.deadline = datetime.fromisoformat(deadline) if deadline else None
        self.duration = float(duration) if duration is not None else None
        self.priority = priority

    def to_feature_dict(self):
        if not self.deadline or self.duration is None:
            raise ValueError("Cannot compute features without deadline and duration")

        hours_remain = (self.deadline - CURRENT_TIME).total_seconds() / 3600
        hours_remain = max(1, hours_remain)

        features = {
            'duration': self.duration,
            'hours_remaining': hours_remain,
            'time_pressure': self.duration / hours_remain
        }
        return features

    def __repr__(self):
        return (f"Event(name={self.event_name}, start={self.start_time}, "
                f"end={self.end_time}, deadline={self.deadline}, "
                f"duration={self.duration}, priority={self.priority})")

class EventPriorityQueue:
    def __init__(self):
        self._heap = []
        self._counter = 0

    def add_event(self, event: Event):
        # Checking if the event is flexible
        if event.start_time is None and event.end_time is None:
            if event.priority is None:
                features = event.to_feature_dict()
                X = pd.DataFrame([features], columns=["duration", "hours_remaining", "time_pressure"])
                pred = loaded_pipeline.predict(X)[0]
                event.priority = int(round(pred))
                print(f"Predicted priority for flexible event '{event.event_name}': {event.priority}")

            # Pushing flexible event into priority queue
            heapq.heappush(self._heap, (event.priority, self._counter, event))
            self._counter += 1
            print(f"Enqueued: {event}")
        
        else:
            # Fixed Event
            print(f"Fixed-time event '{event.event_name}' will not be added to the priority queue.")

    def pop_event(self):
        if not self._heap:
            return None
        _, _, event = heapq.heappop(self._heap)
        return event

    def get_all_events(self):
        # Return a List of Dictionary
        events = [item[2] for item in sorted(self._heap)]
        return [
            {
                "event_name": e.event_name,
                "priority": e.priority,
                "duration": e.duration,
                "deadline": e.deadline.isoformat() if e.deadline else None
            } for e in events
        ]

    def __len__(self):
        return len(self._heap)

# Example usage
if __name__ == "__main__":
    queue = EventPriorityQueue()

    inputs = [
        {"event_name": "Task A", "deadline": "2025-05-27T12:00:00", "duration": 2},
        {"event_name": "Meeting B", "start_time": "2025-05-23T09:00:00", "end_time": "2025-05-28T10:00:00"},
        {"event_name": "Task C", "deadline": "2025-05-30T18:00:00", "duration": 1, "priority": 1},
    ]

    for inp in inputs:
        evt = Event(**inp)
        queue.add_event(evt)

# Sorted List of Dictionary of Event Information
    # print("\nExported Event Data for ML Model:")
    # print(queue.get_all_events())

    # print("\nDequeuing flexible events in priority order:")
    # while len(queue) > 0:
    #     evt = queue.pop_event()
    #     print(evt)
