
MY_PTH="/home/bigdata2022/FakeRealNews/PySpark"

rm -r $MY_PTH/output

$HADOOP_DIR/bin/hdfs dfsadmin -safemode leave
$HADOOP_DIR/bin/hdfs dfs -rm -r output
$HADOOP_DIR/bin/hdfs dfs -rm -r input


$HADOOP_DIR/bin/hdfs dfs -put $MY_PTH/input input


$SPARK_HOME/bin/spark-submit --master yarn  --deploy-mode client  --driver-memory 2g  --executor-memory 2g  --executor-cores 1  --queue default $MY_PTH/script/FakeRealNews.py


$HADOOP_DIR/bin/hdfs dfs -get output $MY_PTH/output
