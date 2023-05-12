#第一个问题 enumerate 的作用
import torch.nn as nn
lista = ["aa","bb","cc"]

#这里的0代表的是指定的序列号
#也就是说enumerate的作用，就是可以打印出序列号，还有具体的值

para = enumerate(lista,0)
for index,paratmp in para:
    print("%s:%s"%(index,paratmp))

#第二个问题，我们就是关于交叉熵损失函数，logsoftmax 和nlllos的关系

#交叉熵损失函数，其实就是  nn.CrossEntropyLoss,他是包含的是logsoftmax + nllloss的

"""
nllloss就是负对数损失函数，而logsoftmax的作用就是，把值进行归一化，归一化到所有的值的和为0的一个过程；

什么意思呢？就是说，我们输入一张图片，经过分类过程以后，就会得到一堆概率的值；这堆概率代表的就是，当前这张图片，是每一个分类的概率；

然后呢？这些概率现在加在一起不是1；我们经过logsoftmax以后，概率和就是1了

然后的话，再去算损失；

要注意的是，我们输入有两个值，一个是output，一个是label，这个label是分类的索引，这个output是概率值；一定要注意，这两个不是同一类，不是都是概率或者都是标签的索引；

这里说的索引，举例比如有5个分类，猫狗鸡鸭鹅；那么索引就是0，1，2，3，4
"""

