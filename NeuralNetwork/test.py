class MyClass:
    def __init__(self) -> None:
        self.x = 33

    def __call__(self, arg) -> None:
        print(self.x + arg)


# mine = MyClass()
# mine(2)

x = [1,2,3,4,5]

zs = [x]

zs.append([1,2,3,4])

print(zs)