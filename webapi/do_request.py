from requests import put,get

if __name__ == '__main__':
    a = put('http://localhost:5000/todo1', data={'data': 'hellow!'}).json()
    print(a)