# Unsupervised-and-Supervised-Machine-Learning-on-rs-fMRI

In this project, meaningful details and inferences were extracted from rs-fMRI scans (Spatial ICs) of 100 patients. rs-fMRI is a functional magnetic resonance imaging that is done to evaluate the functional connectivity in the brain networks when the patient is in a resting state.
• In the first phase, spatial ICs were divided into uniform slices, then the brain boundaries were extracted from each slice using contour detection. Then DBSCAN clustering technique was used to cluster the noise and resting state part of the brain and the total number of such clusters was calculated for each patient.
• In the second phase, rs-fMRI data scans of multiple patients were used to train a model using CNN supervised machine learning technique. This model predicted whether the patient’s brain was in a resting state or in a noisy state after evaluating the blood consumption activities in different regions of the brain with an accuracy of 84 %.