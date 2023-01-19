# opinion_sentiment_classification

Adviser: Pu Wei

This thesis focuses on how to use neural network algorithms for the purpose of sentiment analysis of social opinion.

Firstly, the theories and techniques involved in the project are organized and studied, and the shortcomings of traditional recurrent neural networks in terms of gradient computation are discussed. In order to avoid the problems of gradient disappearance and explosion, and to retain the feature information of the context as much as possible, a bidirectional LSTM is chosen as the initial model of the system. Based on a large amount of Weibo user comment data, the model is trained, validated, and evaluated, and the optimal model obtained during the training process is used as the final prediction model of the sentiment analysis system.

In order to grasp real-time social opinion information, a python-based web crawler code is written to get the latest Weibo posts and corresponding user comments. After cleaning them by certain methods, they are input into the optimal model obtained in the previous step, and the sentiment prediction results are calculated and stored in the Mysql database.

In order to visualize all the data previously stored in the Mysql database to the user, a front and back-end separated platform based on Vue + SpringBoot is built, the Element UI component library is used to render web pages, and the JavaScript-based ECharts visual chart library is introduced to enhance the data visualization.

By training the model based on large-scale data, it has good performance in both the training set, validation set and test set, and the predicted sentiment results of the crawled real-time Weibo comments are basically in line with human intuition, which is of certain use and reference value in monitoring the orientation of public opinion.
