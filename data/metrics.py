ObjectId = lambda s: s
metrics = [{'_id': ObjectId('6612c451986264f1fe43f8f0'), 'experiment_name': 'demoday-specs', 'kind': 'manager', 'timestamp': 1712505937.2152088, 'total_samples': 10000, 'worker_count': 20}, {'_id': ObjectId('6612c58e986264f1fe43f8f1'), 'experiment_name': 'demoday-specs', 'kind': 'manager', 'timestamp': 1712506254.697387, 'total_samples': 15000, 'worker_count': 20}, {'_id': ObjectId('6612c6c8986264f1fe43f8f2'), 'experiment_name': 'demoday-specs', 'kind': 'manager', 'timestamp': 1712506568.0183766, 'total_samples': 20000, 'worker_count': 20}, {'_id': ObjectId('6612d105986264f1fe43f8f3'), 'experiment_name': 'demoday-specs', 'kind': 'manager', 'timestamp': 1712509189.337134, 'total_samples': 25000, 'worker_count': 20}, {'_id': ObjectId('6612d23b986264f1fe43f8f4'), 'experiment_name': 'demoday-specs', 'kind': 'manager', 'timestamp': 1712509499.1484194, 'total_samples': 30000, 'worker_count': 20}, {'_id': ObjectId('6612d371986264f1fe43f8f5'), 'experiment_name': 'demoday-specs', 'kind': 'manager', 'timestamp': 1712509809.3048828, 'total_samples': 35000, 'worker_count': 20}, {'_id': ObjectId('6612d4aa986264f1fe43f8f6'), 'experiment_name': 'demoday-specs', 'kind': 'manager', 'timestamp': 1712510122.7552216, 'total_samples': 40000, 'worker_count': 20}, {'_id': ObjectId('6612d61e986264f1fe43f8f7'), 'experiment_name': 'demoday-specs', 'kind': 'manager', 'timestamp': 1712510494.0862901, 'total_samples': 45000, 'worker_count': 20}, {'_id': ObjectId('6612d753986264f1fe43f8f8'), 'experiment_name': 'demoday-specs', 'kind': 'manager', 'timestamp': 1712510803.255227, 'total_samples': 50000, 'worker_count': 20}, {'_id': ObjectId('6612d88a986264f1fe43f8f9'), 'experiment_name': 'demoday-specs', 'kind': 'manager', 'timestamp': 1712511114.8663626, 'total_samples': 55000, 'worker_count': 20}, {'_id': ObjectId('6612d9c1986264f1fe43f8fa'), 'experiment_name': 'demoday-specs', 'kind': 'manager', 'timestamp': 1712511425.3801365, 'total_samples': 60000, 'worker_count': 20}, {'_id': ObjectId('6612dafd986264f1fe43f8fb'), 'experiment_name': 'demoday-specs', 'kind': 'manager', 'timestamp': 1712511741.5973132, 'total_samples': 65000, 'worker_count': 20}, {'_id': ObjectId('6612dc33986264f1fe43f8fc'), 'experiment_name': 'demoday-specs', 'kind': 'manager', 'timestamp': 1712512051.9791348, 'total_samples': 70000, 'worker_count': 20}, {'_id': ObjectId('6612dd6b986264f1fe43f8fd'), 'experiment_name': 'demoday-specs', 'kind': 'manager', 'timestamp': 1712512363.6584167, 'total_samples': 75000, 'worker_count': 20}, {'_id': ObjectId('6612dea4986264f1fe43f8fe'), 'experiment_name': 'demoday-specs', 'kind': 'manager', 'timestamp': 1712512676.1205606, 'total_samples': 80000, 'worker_count': 20}, {'_id': ObjectId('6612dfdc986264f1fe43f8ff'), 'experiment_name': 'demoday-specs', 'kind': 'manager', 'timestamp': 1712512988.9147084, 'total_samples': 85000, 'worker_count': 20}, {'_id': ObjectId('6612e113986264f1fe43f900'), 'experiment_name': 'demoday-specs', 'kind': 'manager', 'timestamp': 1712513299.6691587, 'total_samples': 90000, 'worker_count': 20}, {'_id': ObjectId('6612e24e986264f1fe43f901'), 'experiment_name': 'demoday-specs', 'kind': 'manager', 'timestamp': 1712513614.4343631, 'total_samples': 95000, 'worker_count': 20}]

samples_per_swap = 5000
timestamps = []
counts = []

for record in metrics:
    timestamps.append(record['timestamp'])
    counts.append(record['total_samples'])

rate_of_increase = []
for i in range(1, len(counts)):
    delta_count = counts[i] - counts[i-1]
    delta_time = timestamps[i] - timestamps[i-1]
    rate = delta_count / delta_time
    rate_of_increase.append(rate)

average_rate = sum(rate_of_increase)/len(rate_of_increase)
print(f"Average samples/s: {average_rate}")
print(f"Time per 10000 samples: {10000/average_rate}")