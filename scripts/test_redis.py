import redis

r = redis.Redis(host='localhost', port=6379, db=0)
ping = r.ping()
print(f"Ping should return True -- Ping: {ping}")
r.set('test_key', 'test_value')
retrieve = r.get('test_key').decode('utf-8')
print(f"If returns test_value, then key-value creation worked. Result: {retrieve}")

