import tkinter as tk
from tkinter import messagebox, ttk


class BoundedNumericalEntry(tk.Entry):
    def __init__(self, master=None, min_value=None, max_value=None, variable=None,
                 vartype=float, width=7, allow_inf=False,  **kwargs):
        if variable is None:
            if vartype == float:
                self.var = tk.DoubleVar()
            elif vartype == int:
                self.var = tk.IntVar()
            else:
                self.var = tk.StringVar()
        else:
            self.var = variable

        self.fake_var = tk.StringVar(value=self.var.get())
        self.vartype = vartype
        self.old_value = self.var.get()
        self.allow_inf = allow_inf

        self.min_value, self.max_value = min_value, max_value
        self.get, self.set = self.fake_var.get, self.fake_var.set

        self.validate_command = master.register(self._check_bounds)
        tk.Entry.__init__(self, master, textvariable=self.fake_var, validate="focus", width=width,
                          vcmd=(self.validate_command, '%P', '%d'), **kwargs)

    def _check_bounds(self, instr, action_type):
        if self.allow_inf and instr == 'INF':
            self.fake_var.set('INF')
            return True

        if action_type == '-1':
            try:
                new_value = self.vartype(instr)
            except ValueError:
                pass
            else:
                if (self.min_value is None or new_value >= self.min_value) and \
                        (self.max_value is None or new_value <= self.max_value):
                    if new_value != self.old_value:
                        self.old_value = self.vartype(self.fake_var.get())
                        self.delete(0, tk.END)
                        self.insert(0, str(self.old_value))
                        self.var.set(self.old_value)
                    return True
        self.delete(0, tk.END)
        self.insert(0, str(self.old_value))
        mn = '-inf' if self.min_value is None else str(self.min_value)
        mx = '+inf' if self.max_value is None else str(self.max_value)
        messagebox.showwarning("Incorrect value in input field", f"Value for {self._name} should be in "
                               f"[{mn}; {mx}] and of type {self.vartype.__name__}")

        return False


class FocusHorizontalScale(tk.Scale):
    def __init__(self, *args, highlightthickness=0, sliderrelief=tk.GROOVE, resolution=0.01,
                 sliderlength=20, length=200, **kwargs):
        tk.Scale.__init__(self, *args, orient=tk.HORIZONTAL, highlightthickness=highlightthickness,
                          sliderrelief=sliderrelief, resolution=resolution,
                          sliderlength=sliderlength, length=length, **kwargs)
        self.bind("<1>", lambda event: self.focus_set())


class FocusCheckButton(tk.Checkbutton):
    def __init__(self, *args, highlightthickness=0, **kwargs):
        tk.Checkbutton.__init__(self, *args, highlightthickness=highlightthickness, **kwargs)
        self.bind("<1>", lambda event: self.focus_set())


class FocusButton(tk.Button):
    def __init__(self, *args, highlightthickness=0, **kwargs):
        tk.Button.__init__(self, *args, highlightthickness=highlightthickness, **kwargs)
        self.bind("<1>", lambda event: self.focus_set())


class FocusLabelFrame(ttk.LabelFrame):
    def __init__(self, *args, highlightthickness=0, relief=tk.RIDGE, borderwidth=2, **kwargs):
        tk.LabelFrame.__init__(self, *args, highlightthickness=highlightthickness, relief=relief,
                               borderwidth=borderwidth, **kwargs)
        self.bind("<1>", lambda event: self.focus_set())

    def set_frame_state(self, state):
        def set_widget_state(widget, state):
            if widget.winfo_children is not None:
                for w in widget.winfo_children():
                    w.configure(state=state)

        set_widget_state(self, state)
