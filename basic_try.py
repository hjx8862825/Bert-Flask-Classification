import re

if __name__ == '__main__':
    line = "你好，世界！ ||| 世界，你好！"
    m = re.match(r"^(.*) \|\|\| (.*)$", line)
    a = m.group(1)
    b = m.group(2)
    print(a)
    print(b)