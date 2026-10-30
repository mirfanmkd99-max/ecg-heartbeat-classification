# ECG Heartbeat Classification

Classification of ECG heartbeats with a random forest and a 1D CNN.



Data Source: Physionet's MIT-BIH Arrhythmia Dataset

License: Unknown



ECG or EKG: Measures electrical activity of the heart muscles and print it on a paper in the form of waves how the heart muscles behave. 

P wave => shows activity of the upper heart chambers (atria).

QRS complex => movement of electrical impulses through the lower heart chambers (ventricles)

ST segment => Ventricle is contracting but no electricity is flowing through it. Straight line between QRS complex and T wave.

T wave => When lower heart chambers are resetting electrically and preparing for their next muscle contraction.



Every beat is classified into 5 classes: 

0: normal beat

1: supraventricular ectopic beat

2: ventricular ectopic beat

3: fusion beat

4: unknown/unclassified beat



After that, I trained Random Forest Classifier from scikit-learn library on my training data. and see all the classification report, Macro F1 score (considered all the classes equally) and Confusion matrix.



I came through the result that class 'fusion beat' class was the hardest one because the F1 score of this class was the lowest of all. Total samples were 96 about 30% were misclassified/confused with normal beats and 0.08% with ventricular beats. 



I tried with 3 different random\_state values and the ranking changed for 20 and was same for 42 and 60.

Ranking was F S V Q N for 42 and 60 and S F V Q N for 20.





|experiment|macro\_f1\_val|f1\_N|f1\_S|f1\_V|f1\_F|f1\_Q|train\_seconds|
|-|-:|-:|-:|-:|-:|-:|-:|
|1a. Random forest, no class\_weight|0.824|0.982|0.8|0.859|0.5|0.978|3.734|
|1b. Random forest, class\_weight=balanced\_subsample|0.853|0.978|0.8|0.822|0.706|0.96|2.271|
|2. Logistic regression + StandardScaler|0.633|0.95|0.556|0.386|0.353|0.921|1.589|
|3. HistGradientBoostingClassifier|0.87|0.986|0.8|0.921|0.667|0.978|5.766|
|4. Random forest, no class\_weight, oversampled train|0.869|0.98|0.821|0.848|0.737|0.96|17.089|







1D CNN Model training.



Trained CNN Model => 3 Convolution layers, Batch Normalization layers, ReLU activation function and MaxPooling layers.



Flatten, Dropout of 0.3 and Linear layers of 64.



Name my model ECGNet with balanced class weights. Cross Entropy Loss and Adam Optimizer with learning rate of 0.003.



Then predict through my model evaluation.



Then train for 20 epochs. Lastly, print Macro F1 score and loss after each epoch and save the model weights with best F1.













