cd FakeRealNews
cd PySpark

#STOP HADOOP
sh script/stopHadoop.sh

#RUN HADOOP
sh script/runHadoop.sh

#RUN SPARK AND BUILD MODEL FOR PREDICTIONS
sh script/runSpark.sh

