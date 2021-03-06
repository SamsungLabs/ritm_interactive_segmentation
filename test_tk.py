try:
    # python 2.x
    import Tkinter as tk
except ImportError:
    # python 3.x
    import tkinter as tk

class Example(tk.Frame):
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)

        self.popupMenu = tk.Menu(self, tearoff=0)
        self.popupMenu.add_command(label="One", command=self.menu_one)
        self.popupMenu.add_command(label="Two", command=self.menu_two)
        self.popupMenu.add_command(label="Three", command=self.menu_three)

        self.bind("<Button-2>", self.popup)

    def menu_one(self):
        print ("one...")

    def menu_two(self):
        print ("two...")

    def menu_three(self):
        print ("three...")

    def popup(self, event):
        self.popupMenu.post(event.x_root, event.y_root)

if __name__ == "__main__":
    root =tk.Tk()
    frame = Example(root, width=200, height=200)
    frame.pack(fill="both", expand=True)
    root.mainloop()