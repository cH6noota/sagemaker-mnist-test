
import boto3
import json
import numpy as np
import random
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
num_samples = 5
indices = random.sample(range(x_test.shape[0] - 1), num_samples)
images, labels = x_test[indices]/255, y_test[indices]

endpoint = "sagemaker-tensorflow-scriptmode-2021-07-28-05-30-35-308"
client = boto3.client('sagemaker-runtime')

response = client.invoke_endpoint(
    EndpointName=endpoint,
        Body=json.dumps({
            "instances": images.reshape(num_samples, 28, 28, 1).tolist()
        }),
        ContentType='application/json'
)

body = response['Body']
prediction = json.load(body)['predictions']

prediction = np.array(prediction)
predicted_label = prediction.argmax(axis=1)
print('The predicted labels are: {}'.format(predicted_label))
