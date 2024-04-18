class Mode:
    def __init__(self, name, num):
        self.name = name
        self.num = num

modes = [
    Mode("Camera", 0),
    Mode("ShortCut", 1),
    Mode("GUI", 2),
    Mode("GoogleMap", 3),
    Mode("GoogleSerachEngine", 4),
    Mode("ChatGPT", 5),
]

print(modes[0].name, modes[0].num)

for i in range(len(modes)):
    if modes[i].name == "ChatGPT":
        print(modes[i].num)