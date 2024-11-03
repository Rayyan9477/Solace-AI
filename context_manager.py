import time

class ContextManager:
    def __init__(self, context_timeout=600):  # 10 minutes default timeout
        self.context = {}
        self.context_timeout = context_timeout

    def add_to_context(self, user_id, key, value):
        if user_id not in self.context:
            self.context[user_id] = {}
        self.context[user_id][key] = {
            'value': value,
            'timestamp': time.time()
        }

    def get_from_context(self, user_id, key):
        if user_id in self.context and key in self.context[user_id]:
            if time.time() - self.context[user_id][key]['timestamp'] < self.context_timeout:
                return self.context[user_id][key]['value']
            else:
                del self.context[user_id][key]
        return None

    def clear_context(self, user_id):
        if user_id in self.context:
            del self.context[user_id]

    def get_full_context(self, user_id):
        if user_id in self.context:
            return {k: v['value'] for k, v in self.context[user_id].items() 
                    if time.time() - v['timestamp'] < self.context_timeout}
        return {}