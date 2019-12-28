HADOOP_CMD="hadoop" #我的hadoop位置
STREAM_JAR_PATH="/usr/local/hadoop/share/hadoop/tools/lib/hadoop-streaming-2.8.5.jar" #streaming这个jar包的位置
INPUT_FILE_PATH_1="/join/input/*" #测试文件在hdfs中的位置。所以需要先将文件传入hdfs中
OUTPUT_PATH="/join/output" #文件输出目录（运行mr前一定不能存在，mr自己会创建）
 
$HADOOP_CMD fs -rmr -skipTrash $OUTPUT_PATH #删除原有的输出文件夹
 
#step 1.下面代码就是使用streaming框架的命令，具体参数就不解释了
$HADOOP_CMD jar $STREAM_JAR_PATH \
	-D mapred.map.tasks=3 \
	-D mapred.job.name="join_test" \
	-partitioner org.apache.hadoop.mapred.lib.KeyFieldBasedPartitioner \
        -input $INPUT_FILE_PATH_1 \
        -output $OUTPUT_PATH \
        -mapper "python3 mapper.py" \
        -reducer "python3 reducer.py" \
        -file ./mapper.py \
        -file ./reducer.py
