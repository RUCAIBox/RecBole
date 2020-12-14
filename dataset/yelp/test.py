with open('yelp.inter', 'r') as fp:
	fp.readline()
	max_time, min_time = 0, 9999999999
	for line in fp:
		l = line.strip().split('\t')
		t = int(l[-1])
		max_time = max(t, max_time)
		min_time = min(t, min_time)
print(max_time)
print(min_time)
