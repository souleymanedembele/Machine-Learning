# import packages
import numpy as np
from sklearn import preprocessing

# Define saome sample data
input_data = ['seattle', 'paris', 'abidjan', 'ouagadougou', 'paris', 'tokyo', 'amsterdam']

# create the label encoder object
encoder = preprocessing.LabelEncoder()

# train it
encoder.fit(input_data)

# print the mapping between words and numbers
print("\nLabel mapping:")
for i, item in enumerate(encoder.classes_):
    print(item, '-->', i)

# encode set of random
test_labels = ['ouagadougou', 'amsterdam', 'seattle']
encoded_values = encoder.transform(test_labels)
print("\nLabels = ", test_labels)
print("Encoded values =", list(encoded_values))

# decode a random set of numbers
encoded_values = [3, 0, 4, 1, 2]
decoded_list = encoder.inverse_transform(encoded_values)
print("\nEncoded values =", encoded_values)
print("\n decode labels =", list(decoded_list))
