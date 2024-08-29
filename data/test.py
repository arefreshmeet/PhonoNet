import os
class testclass:
    def __init__(self):
        print('hello')
        current_path = os.getcwd()
        print("当前工作目录是:", current_path)