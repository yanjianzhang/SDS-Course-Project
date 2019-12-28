## 五子棋AI

Shining points:

1. Used Temporal-Difference Learning in taking step
2. Threat-space was used for must-win strategy 
3.  Implemented the Numpy numerical operation on our own To compact the size of executive file (.exe)
4. Overpower pisq-7 (ranked 43/58 in published [Elo Ratings](<https://gomocup.org/elo-ratings/>)) with a score of 13:7 and ranked 5/30 in the class  

![vspisq7](vspisq7.png)

Files start without "Eval"  is for train

Files start with "Eval" is the file that used trained weight

The executive file can be generate by the following command:

pyinstaller.exe .\EvalExample.py pisqpipe.py --name pbrain-eval-log3.exe --onefile

Manager piskvork can be download from [Gomocup websites](http://gomocup.org)