## 五子棋AI

我们使用了强化学习+威胁序列+自定义的策略（我们称为”气“）来产生我们的五子棋AI，最终版本(zip文件)在固定开局下能够以13:7的优势打败pisq7，4：16的劣势逊于Noesis. 

![vspisq7](vspisq7.png)

![VSnoesis](VSnoesis.JPG)

带 Eval 开头的部分代码是使用已经训练好的权重来生成模型的部分，因此最终我们的版本由以下命令生成
pyinstaller.exe .\EvalExample.py pisqpipe.py --name pbrain-eval-log3.exe --onefile
没有带 Eval的部分是用于训练神经网络的部分代码

五子棋manager piskvork详见 [Gomocup websites](http://gomocup.org)