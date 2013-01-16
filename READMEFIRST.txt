1. For Variable Selection (Using R):
   
   just run corrBasedFS.R in folder /VariableSelection
   
   Copy the output (train1749.csv and test1749.csv) to /BlendingModel/EnsembleLearningToolbox/Data

2. For building blending model:
   
   run blend.py (using python) in folder /BlendingModel/EnsembleLearningToolbox/Benchmarks
   
   run ourBlend.m (using matlab) in folder /BlendingModel/DeepLearningToolbox/tests
   
   mix together both the output of blend.py and ourBlend.m into both .csv for training and for testing.
   example (mix dataset_blend_train1749_rf500_3.csv and blendtrain1749v2)
	   (mix dataset_blend_test1749_rf500_3.csv and blendtest1749v2) 
   
   Put the mix final data to folder /FinalModel
   example (blendtrain50v2_1749rf500_3 and blendtest50v2_1749rf500_3)

3. For building final model:
   
   run FinalModel.py (using python)
   The final output is in Folder /FinalModel/Submissions