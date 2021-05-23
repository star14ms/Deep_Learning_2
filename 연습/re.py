import re

def space_to_s(space):
    space = str(space.group())
    return 's'
    
a = '      ....'
b = re.sub(' |.', space_to_s, a)
print(b)