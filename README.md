# Advanced Classification of Disaster-Related Tweets Using Deep Learning 

In this project, we will build a deep learning model using Keras to classify tweets as real or fake in the context of disasters. 🎯 This task is inspired by the "NLP with Disaster Tweets" challenge and enriched with additional data to improve model performance and insights. 📈 The dataset provides a fascinating opportunity to explore Natural Language Processing (NLP) techniques on real-world data. 🌎

<p align="center">
   <img src="https://img.shields.io/badge/pypi-3775A9?style=for-the-badge&logo=pypi&logoColor=white" />
   <img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue" />
   <img src="https://img.shields.io/badge/Keras-FF0000?style=for-the-badge&logo=keras&logoColor=white" />
   <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
   <img src="https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white" />
   <img src="https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white" />
   <img src="https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white" />
</p>

🙏 I would like to extend my heartfelt gratitude to [Santiago Hernández, an expert in Cybersecurity and Artificial Intelligence](https://www.udemy.com/user/shramos/). His incredible course on Deep Learning, available at Udemy, was instrumental in shaping the development of this project. The insights and techniques learned from his course were crucial in crafting the neural network architecture used in this classifier.

The dataset contains over 11,000 tweets associated with disaster-related keywords such as "crash," "quarantine," and "bush fires." 🔥 The data structure is based on the original "Disasters on social media" dataset. It includes:

* **Tweets:** The text of the tweet. 💬
* **Keywords:** Specific disaster-related keywords. 🚨
* **Location:** The geographical information provided in the tweets. 📍

These tweets were collected on January 14th, 2020, and cover major events including:

* The eruption of Taal Volcano in Batangas, Philippines. 🌋
* The emerging outbreak of Coronavirus (COVID-19). 🦠
* The devastating Bushfires in Australia. 🔥
* The Iranian downing of flight PS752. ✈️

The dataset contains text that may include profane, vulgar, or offensive language. ⚠️ Please approach with caution during analysis.

## Project Goals

### Inspiration**

The primary goal of this project is to develop a machine learning model capable of identifying whether a tweet is genuinely related to a disaster or not. This involves:

* Enriching the already available data with newly collected, manually classified tweets. ✍️
* Leveraging state-of-the-art deep learning methods to extract meaningful insights. 🧠
* Applying NLP techniques to preprocess, clean, and tokenize the tweets for model training. 🛠️

This notebook will walk through the process of preparing the dataset, building a deep learning model, and evaluating its performance. By the end, we aim to achieve a robust model that can classify disaster tweets with high accuracy. 💯

### Why It Matters

Effective classification of disaster-related tweets has numerous practical applications:

* **Emergency Response:** Helps organizations identify critical information in real time. 🚨
* **Resource Allocation:** Facilitates better planning by focusing on real disasters. 🗺️
* **Misinformation Control:** Mitigates the spread of false information during crises. 🚫

https://www.kaggle.com/datasets/vstepanenko/disaster-tweets
