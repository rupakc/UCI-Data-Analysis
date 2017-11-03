from datasketch import MinHashLSHForest, MinHash

data1 = ['minhash', 'is', 'a', 'probabilistic', 'data', 'structure', 'for',
        'estimating', 'the', 'similarity', 'between', 'datasets']
data2 = ['minhash', 'is', 'a', 'probability', 'data', 'structure', 'for',
        'estimating', 'the', 'similarity', 'between', 'documents']
data3 = ['minhash', 'is', 'probability', 'data', 'structure', 'for',
        'estimating', 'the', 'similarity', 'between', 'documents']

# Create MinHash objects
m1 = MinHash(num_perm=128)
m2 = MinHash(num_perm=128)
m3 = MinHash(num_perm=128)
for d in data1:
    m1.update(d.encode('utf8'))
for d in data2:
    m2.update(d.encode('utf8'))
for d in data3:
    m3.update(d.encode('utf8'))

# Create a MinHash LSH Forest with the same num_perm parameter
forest = MinHashLSHForest(num_perm=128)

# Add m2 and m3 into the index
forest.add("m2", m2)
forest.add("m3", m3)

# IMPORTANT: must call index() otherwise the keys won't be searchable
forest.index()

# Check for membership using the key
print("m2" in forest)
print("m3" in forest)

# Using m1 as the query, retrieve top 2 keys that have the higest Jaccard
result = forest.query(m1, 1)
print("Top 2 candidates", result)

from datasketch import MinHashLSHEnsemble, MinHash

set1 = set(["cat", "dog", "fish", "cow"])
set2 = set(["cat", "dog", "fish", "cow", "pig", "elephant", "lion", "tiger",
             "wolf", "bird", "human"])
set3 = set(["cat", "dog", "car", "van", "train", "plane", "ship", "submarine",
             "rocket", "bike", "scooter", "motorcyle", "SUV", "jet", "horse"])

# Create MinHash objects
m1 = MinHash(num_perm=128)
m2 = MinHash(num_perm=128)
m3 = MinHash(num_perm=128)
for d in set1:
    m1.update(d.encode('utf8'))
for d in set2:
    m2.update(d.encode('utf8'))
for d in set3:
    m3.update(d.encode('utf8'))

# Create an LSH Ensemble index with a threshold
lshensemble = MinHashLSHEnsemble(threshold=0.8, num_perm=128)

# Index takes an iterable of (key, minhash, size)
lshensemble.index([("m2", m2, len(set2)), ("m3", m3, len(set3))])

# Check for membership using the key
print("m2" in lshensemble)
print("m3" in lshensemble)

# Using m1 as the query, get an result iterator
print("Sets with containment > 0.2:")
for key in lshensemble.query(m1, len(set1)):
    print(key)

from datasketch import HyperLogLog,HyperLogLogPlusPlus

data1 = ['hyperloglog', 'is', 'a', 'probabilistic', 'data', 'structure', 'for',
'estimating', 'the', 'cardinality', 'of', 'dataset', 'dataset', 'a']

h = HyperLogLogPlusPlus(p=12)
for d in data1:
  h.update(d.encode('utf8'))
print("Estimated cardinality is", h.count())

s1 = set(data1)
print("Actual cardinality is", len(s1))
