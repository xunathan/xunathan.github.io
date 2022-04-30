---
title: "基于Ｐython的会计后续教育半自动听课脚本"
date: 2022-04-30
draft: false
---
会计后续教育听课会弹出问题，如果不解答视频就不会听下去，经观察发现问题的位置一般比较固定，所以可以开发一个简单的脚本辅助听课。  
- 运行环境:python3  
### 首先开发一个小脚本获取点击答案的鼠标位置

```
import time
import win32api
import win32con

def setMouseLession():
    while True:
        pos = win32api.GetCursorPos()
        print(pos[0],pos[1])
        time.sleep(1)
```

### 获取播放暂时时题目的位置，按照单选和多选，点击答案和提交答案获取鼠标位置，把这些位置记录下来，定时点击即可
```
#!/usr/bin/env python

from pymouse import PyMouse
import time
from PIL import Image
from PIL import ImageGrab

def get_RGB(pixelX, pixelY):
    im = ImageGrab.grab((0,0,1366,768))
    im.save('screen.jpg','jpeg')
    img_src = Image.open('screen.jpg')
    img_src = img_src.convert('RGBA')
    str_strlist = img_src.load()
    RGBA = str_strlist[pixelX, pixelY]
    img_src.close()
    return RGBA

def printTime():
    print("begin work" + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

#这是在我的屏幕上获取的答案的位置
A=(277,384)
B=(277,414)
C=(277,444)
D=(277,474)
E=(277,504)
S=(667,597)
Err_S=(674,456)
allRst = (A,B,C,D,E)
multiAllRst = ((A,C),(A,C),(A,D),(B,C),(B,D),(C,D),(A,B,C),(A,B,D),(B,C,D),(A,B,C,D))
m = PyMouse()

def isSame():
    im1 = ImageGrab.grab((0,0,1366,768))
    time.sleep(10)
    im2 = ImageGrab.grab((0,0,1366,768))

    if im1 == im2:
        m.click(A[0],A[1])

def clickOption(location,len = 0):
    m.click(location[0],location[1]+len)
    time.sleep(2)
    m.click(location[0]+60,location[1]+len)
    time.sleep(2)

def clickSure(location):
    m.click(location[0]-120,location[1])
    time.sleep(2)
    m.click(location[0],location[1])
    time.sleep(2)

def clickSingleOptions(options,isSure,len = 0):
    for idx, option in enumerate(options):
    print("single Idx:", idx)
    clickOption(option,len)
    if isSure:
    print("Sure key")
    clickSure(S)
    clickSure(Err_S)

def clearOptions(options,len = 0):
    for option in options:
        clickOption(option,len)

def clickMuti(multiAllRst,len = 0):
    for idx,rst in enumerate(multiAllRst):
        print("Multi Idx:", idx)
        clickSingleOptions(rst, False,len)
        clickSure(S)
        clickSure(Err_S)
        clearOptions(rst,len)

def begin_work():
    while True:
        printTime()
        if get_RGB(S[0],S[1]) == (250, 209, 177, 255):
        print("Enter single options")
        clickSingleOptions(allRst,True)

        if get_RGB(S[0],S[1]) == (250, 209, 177, 255)：      
            print("Enter single options")
            clickSingleOptions(allRst,True,16)

        if get_RGB(S[0],S[1]) == (250, 209, 177, 255):
            print("Enter Multi options")
            clickMuti(multiAllRst)

        if get_RGB(S[0],S[1]) == (250, 209, 177, 255):
            print("Enter Multi options")
            clickMuti(multiAllRst,16)

        isSame()
        time.sleep(90)

if name == "main":
    time.sleep(10)
    begin_work()
```
