class Parent:
    def __init__(self, name):
        self.name = name

    def print_name(self):
        self.__print_f__(self.name)

    def __print_f__(self, name):
        print(name)

    def pprint_f(self, name):
        print(name)



class Child(Parent):
    def __init__(self, name):
        super().__init__(name)
        
    def call(self):
        self.print_name()

    def __print_f__(self, name):
        print("I am a child class " + name)

def print_f_outsize(self, name):
    print("I am a child class outside " + name)

# Parent.pprint_f = print_f_outsize

child = Child("Alice")

child.call()