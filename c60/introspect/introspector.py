# PipelineIntrospector class and related explainability logic

class PipelineIntrospector:
    """
    Centralized engine for logging, tracing, and explaining pipeline evolution events.
    Logs population initialization, mutation, evaluation, selection, and agent actions.
    """
    def __init__(self):
        self.history = []

    def log(self, event_type, pipeline_id, details=None):
        self.history.append({
            "event_type": event_type,
            "pipeline_id": pipeline_id,
            "details": details or {},
        })

    def get_history(self, pipeline_id=None):
        if pipeline_id is None:
            return self.history
        return [e for e in self.history if e["pipeline_id"] == pipeline_id]

    def explain(self, pipeline_id):
        events = self.get_history(pipeline_id)
        return f"Explanation for {pipeline_id}:\n" + "\n".join(str(e) for e in events)
