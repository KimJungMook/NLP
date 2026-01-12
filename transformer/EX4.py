import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
import time
import tensorflow as tf
import tensorflow_datasets as tfds
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

urllib.request.urlretrieve("https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv",
                           filename="ChatBotData.csv")

train_data = pd.read_csv("ChatBotData.csv")
print(train_data.head())