# People interactions machine learning

Simple machine learning model to predict investors interactions in meeting.

Given the position and the characteristics of some investors in a meeting every 5 minutes, the goal is to predict what type of person a new investor will naturally want to speak with.

Here is the preprocessing part which establish a label matrix of zeros and ones when two investors spoke to each other during the meeting (looking if they were close enough for more than 5 minutes)

The proposed model is a uncommon "supervised" kmeans and it should arrive in a month.
