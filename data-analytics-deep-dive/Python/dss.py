
num = 1234

def disp1(msg):
    print('disp1', msg)

def disp2(msg):
    print('disp2', msg)
    
class Calc:
    def plus(self, *args):
        return sum(args)
