import re

class PointGroup():
    def __init__(self, s, family, n, subfamily):
        self.str = s
        self.family = family
        self.n = n
        if self.n == 0:
            self.is_linear = True
        else:
            self.is_linear = False
        self.subfamily = subfamily
        self.dumb_pg()

    @classmethod
    def from_string(cls, s):
        regex = r"([A-Z]+)(\d+)?([a-z]+)?"
        m = re.match(regex, s)
        family, n, subfamily = m.groups()
        if n is not None:
            n = int(n)
        if subfamily is not None:
            subfamily = str(subfamily)
        family = str(family)
        return cls(s, family, n, subfamily)

    def __str__(self):
        nstr = self.n
        sfstr = self.subfamily
        if self.n is None:
            nstr = ""
        elif self.n == 0:
            nstr = "inf"
        else:
            nstr = str(self.n)
        if self.subfamily is None:
            sfstr = ""
        return self.family + nstr + sfstr

    def __repr__(self) -> str:
        return self.__str__()

    def dumb_pg(self):
        # Check if a dumb point group has been made (e.g. D1h, D0v, C2i)
        argstr = f"You have generated a dumb point group: {self.str}. Family {self.family}, n {self.n}, subfamily {self.subfamily}. We aren't sure how you managed to do this but we aren't paid enough to proceed with any calculations. If you have any questions, feel free to email the CFOUR listserv."
        if self.n is None:
            if self.family == "C":
                allowed = ["s", "i"]
                if self.subfamily in allowed:
                    return 0
            elif self.family == "T":
                allowed = [None, "h", "d"]
                if self.subfamily in allowed:
                    return 0
            elif self.family == "O" or self.family == "I":
                allowed = [None, "h"]
                if self.subfamily in allowed:
                    return 0
        elif self.n == 0:
            if self.family == "D" and self.subfamily == "h":
                return 0
            elif self.family == "C" and self.subfamily == "v":
                return 0
        elif self.n == 1:
            if self.family == "C" and self.subfamily is None:
                return 0
        elif self.n >= 2:
            if self.family == "C":
                allowed = [None, "v", "h"]
                if self.subfamily in allowed:
                    return 0
            elif self.family == "D":
                allowed = [None, "d", "h"]
                if self.subfamily in allowed:
                    return 0
            elif self.family == "S":
                if self.subfamily is None and self.n % 2 == 0:
                    return 0
        raise Exception(argstr)